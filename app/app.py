import os
import random
import re
import uuid
from pathlib import Path

from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename

from process_video import query_vss_agent_video

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXT = {"mp4", "mov", "avi", "mkv", "webm"}
MAX_MB = 50

ACTIONS = [
    "raise your hands to the sky",
    "do the ninja",
    "scratch your scalp like a monkey",
    "rub your stomach",
    "hide your eyes",
    "turn around",
]

ACTION_EMOJIS = {
    "raise your hands to the sky": "🙌",
    "do the ninja": "🥷",
    "scratch your scalp like a monkey": "🐒",
    "rub your stomach": "🫃",
    "hide your eyes": "🙈",
    "turn around": "🔄",
}

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-change-me")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    winner = None
    action = random.choice(ACTIONS)

    if request.method == "POST":
        action = request.form.get("action", action)
        video = request.files.get("video")

        if not video or video.filename == "":
            flash("Please select a video file.", "error")
            return render_template(
                "index.html",
                result=None,
                winner=None,
                action=action,
                emoji=ACTION_EMOJIS.get(action, "🎬"),
            )

        if not allowed_file(video.filename):
            flash(
                f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED_EXT))}",
                "error",
            )
            return render_template(
                "index.html",
                result=None,
                winner=None,
                action=action,
                emoji=ACTION_EMOJIS.get(action, "🎬"),
            )

        safe_name = secure_filename(video.filename)
        unique_name = f"{uuid.uuid4().hex}_{safe_name}"
        save_path = UPLOAD_DIR / unique_name
        video.save(save_path)

        prompt = (
            f"Assign a number to each person from left to right.\n"
            f"Return the number of the person who is first to execute the following action : {action}."
        )

        try:
            result = query_vss_agent_video(str(save_path), prompt)
            match = re.search(r"\b(\d+)\b", result or "")
            winner = match.group(1) if match else None
        except Exception as exc:
            app.logger.exception("Agent query failed")
            flash(f"Processing error: {exc}", "error")
        finally:
            save_path.unlink(missing_ok=True)

    return render_template(
        "index.html",
        result=result,
        winner=winner,
        action=action,
        emoji=ACTION_EMOJIS.get(action, "🎬"),
    )


@app.errorhandler(413)
def too_large(_):
    flash(f"File too large (max {MAX_MB} MB).", "error")
    return redirect(url_for("index")), 413
