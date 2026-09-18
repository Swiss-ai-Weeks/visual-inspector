import os
import random
import re
import uuid
from pathlib import Path

from flask import (
    Flask, abort, flash, redirect, render_template, request,
    send_from_directory, url_for,
)
from werkzeug.utils import secure_filename

from overlay import ensure_mp4, probe_duration, probe_video, render_overlay
from process_video import ask_agent, delete_video, upload_video
from tracking import TrackingUnavailable, fetch_tracks, wait_for_cv

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXT = {"mp4", "mov", "avi", "mkv", "webm"}
MAX_MB = 50

OVERLAY_SUFFIX = "_overlay.mp4"
KEEP_OVERLAYS = 10  # rendered overlays are served after the request, so prune by age

ACTIONS = [
    "raise your hand",
    "scratch your scalp like a monkey",
]

ACTION_EMOJIS = {
    "raise your hand": "🙌",
    "scratch your scalp like a monkey": "🐒",
}

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-change-me")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def pick_two_actions() -> tuple[str, str]:
    if len(ACTIONS) >= 2:
        a1, a2 = random.sample(ACTIONS, 2)
    else:
        a1 = a2 = ACTIONS[0]
    return a1, a2


def prune_overlays() -> None:
    """Keep only the most recent overlays; they outlive their request."""
    renders = sorted(
        UPLOAD_DIR.glob(f"*{OVERLAY_SUFFIX}"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for stale in renders[KEEP_OVERLAYS:]:
        stale.unlink(missing_ok=True)


def build_overlay(mp4_path: Path, video_name: str) -> str:
    """Fetch this clip's CV tracks and render them over it.

    Keyed by the video name rather than the sensor id: VSS records CV frames
    under `sensorId = <uploaded filename without extension>`, not the VST UUID.
    That name comes from `upload_video()`, which registers the clip under a
    name of its own — it is not this file's stem.

    Returns the overlay's filename, used as the token in the /overlay route.
    """
    _, width, height = probe_video(mp4_path)

    # RTVI-CV writes frames asynchronously and slower than real time. Reading
    # early truncates the timeline, so wait for it to cover the clip first.
    wait_for_cv(video_name, probe_duration(mp4_path))

    tracks = fetch_tracks(video_name, (width, height))

    overlay_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{OVERLAY_SUFFIX}"
    render_overlay(mp4_path, tracks, overlay_path)

    app.logger.info(
        "overlay rendered: %d track(s) over %s", len(tracks), mp4_path.name
    )
    prune_overlays()
    return overlay_path.name


def _ctx(action1, action2, **kwargs):
    return dict(
        action1=action1,
        action2=action2,
        emoji1=ACTION_EMOJIS.get(action1, "🎯"),
        emoji2=ACTION_EMOJIS.get(action2, "🎯"),
        **kwargs,
    )


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    winner = None
    overlay_token = None
    action1, action2 = pick_two_actions()

    if request.method == "POST":
        action1 = request.form.get("action1", action1)
        action2 = request.form.get("action2", action2)
        video = request.files.get("video")

        if not video or video.filename == "":
            flash("Please select a video file.", "error")
            return render_template("index.html", **_ctx(action1, action2))

        if not allowed_file(video.filename):
            flash(f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED_EXT))}", "error")
            return render_template("index.html", **_ctx(action1, action2))

        safe_name = secure_filename(video.filename)
        unique_name = f"{uuid.uuid4().hex}_{safe_name}"
        save_path = UPLOAD_DIR / unique_name
        video.save(save_path)

        prompt = (
            "Assign a number to each person from left to right.\n"
            "Return the number of the person who is first to execute BOTH of the following actions:\n"
            f"1. {action1}\n"
            f"2. {action2}"
        )

        mp4_path = save_path
        sensor_id = None
        video_name = ""
        cv_ready = False
        try:
            mp4_path = ensure_mp4(save_path)
            sensor_id, video_name, cv_ready = upload_video(str(mp4_path))
            result = ask_agent(sensor_id, prompt, mp4_path.name)
            match = re.search(r"\b(\d+)\b", result or "")
            winner = match.group(1) if match else None
        except Exception as exc:
            app.logger.exception("Agent query failed")
            flash(f"Processing error: {exc}", "error")

        # Tracking is additive: a failure here must not cost us the game result.
        if sensor_id:
            try:
                if not cv_ready:
                    raise TrackingUnavailable(
                        "the CV pipeline did not run for this clip (/complete failed)"
                    )
                overlay_token = build_overlay(mp4_path, video_name)
            except Exception as exc:
                app.logger.exception("Tracking overlay failed")
                flash(f"Tracking unavailable: {exc}", "error")

        # Deleting the sensor also drops its CV frames from Elasticsearch, so
        # the per-person tracking data goes with it. Opt in only — the next
        # stage (action recognition) needs those positions to stay put.
        if sensor_id and os.environ.get("DELETE_SENSORS"):
            delete_video(sensor_id)
        if not os.environ.get("KEEP_UPLOADS"):
            save_path.unlink(missing_ok=True)
            if mp4_path != save_path:
                mp4_path.unlink(missing_ok=True)

    return render_template(
        "index.html",
        **_ctx(action1, action2, result=result, winner=winner, overlay_token=overlay_token),
    )


@app.route("/overlay/<token>")
def overlay_video(token: str):
    """Serve a rendered tracking overlay."""
    name = secure_filename(token)
    if not name.endswith(OVERLAY_SUFFIX):
        abort(404)
    return send_from_directory(UPLOAD_DIR, name, mimetype="video/mp4")


@app.errorhandler(413)
def too_large(_):
    flash(f"File too large (max {MAX_MB} MB).", "error")
    return redirect(url_for("index")), 413
