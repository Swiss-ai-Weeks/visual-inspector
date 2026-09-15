import os
import random
import re
import subprocess
import uuid
from pathlib import Path

from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename

import imageio_ffmpeg
from process_video import query_vss_agent_video

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXT = {"mp4", "mov", "avi", "mkv", "webm"}
MAX_MB = 50

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


def ensure_mp4(src: Path) -> Path:
    """Re-encode to H.264/AAC MP4 for VST compatibility."""
    dst = src.with_name(src.stem + "_vst.mp4")
    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-i", str(src),
            "-c:v", "libx264",
            "-profile:v", "baseline",
            "-level", "3.1",
            "-pix_fmt", "yuv420p",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-ar", "44100",
            "-movflags", "+faststart",
            str(dst),
        ],
        check=True,
        capture_output=True,
    )
    app.logger.info("ffmpeg OK: %s → %s", src.name, dst.name)
    return dst


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
    action1, action2 = pick_two_actions()

    if request.method == "POST":
        action1 = request.form.get("action1", action1)
        action2 = request.form.get("action2", action2)
        video = request.files.get("video")

        if not video or video.filename == "":
            flash("Please select a video file.", "error")
            return render_template("index.html", **_ctx(action1, action2, result=None, winner=None))

        if not allowed_file(video.filename):
            flash(f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED_EXT))}", "error")
            return render_template("index.html", **_ctx(action1, action2, result=None, winner=None))

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
        try:
            mp4_path = ensure_mp4(save_path)
            result = query_vss_agent_video(str(mp4_path), prompt)
            match = re.search(r"\b(\d+)\b", result or "")
            winner = match.group(1) if match else None
        except Exception as exc:
            app.logger.exception("Agent query failed")
            flash(f"Processing error: {exc}", "error")
        finally:
            save_path.unlink(missing_ok=True)
            if mp4_path != save_path:
                mp4_path.unlink(missing_ok=True)

    return render_template("index.html", **_ctx(action1, action2, result=result, winner=winner))


@app.errorhandler(413)
def too_large(_):
    flash(f"File too large (max {MAX_MB} MB).", "error")
    return redirect(url_for("index")), 413
