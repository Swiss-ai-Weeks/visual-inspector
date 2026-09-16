"""Challenge Master (Agent 1) -- Flask orchestrator.

Generates sequences, triggers capture, routes through VSS perception,
runs deterministic validation, and stores results. No LLM in this layer.
"""

import json
import logging
import os
import random
import sqlite3
import subprocess
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, render_template, request, flash, redirect, url_for, g
from werkzeug.utils import secure_filename

from models import (
    ACTION_EMOJIS,
    ACTION_VOCAB,
    SEQUENCE_ACTIONS,
    ActionEvent,
    GameResult,
    PlayerResult,
)
from perception import extract_events, extract_events_via_agent
from validator import validate, needs_adjudication
from adjudicator import adjudicate

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
DB_PATH = BASE_DIR / "movematch.db"

ALLOWED_EXT = {"mp4", "mov", "avi", "mkv", "webm"}
MAX_MB = 50
SEQUENCE_LENGTH = 3
TIME_LIMIT = 5.0

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-change-me")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024


# -- Database -----------------------------------------------------------------

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(str(DB_PATH))
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    db = sqlite3.connect(str(DB_PATH))
    db.execute("""
        CREATE TABLE IF NOT EXISTS games (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            played_at       TEXT    NOT NULL,
            challenge       TEXT    NOT NULL,
            winner_name     TEXT,
            total_players   INTEGER DEFAULT 0,
            results_json    TEXT,
            events_json     TEXT,
            adjudicated     INTEGER DEFAULT 0
        )
    """)
    db.commit()
    db.close()


init_db()


# -- Helpers ------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def generate_sequence() -> list[str]:
    return random.sample(SEQUENCE_ACTIONS, min(SEQUENCE_LENGTH, len(SEQUENCE_ACTIONS)))


def ensure_mp4(src: Path) -> Path:
    """Re-encode to H.264/AAC MP4 for VST compatibility."""
    dst = src.with_name(src.stem + "_vst.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
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
    return dst


# -- Pipeline -----------------------------------------------------------------

def run_pipeline(
    video_path: str,
    expected: list[str],
    player_names: list[str],
    lenient: bool = False,
) -> GameResult:
    """Full MoveMatch pipeline: upload -> perceive -> validate -> adjudicate."""
    from vss_client import upload_video, delete_video

    sensor_id = upload_video(video_path)

    try:
        try:
            events = extract_events(sensor_id)
        except Exception:
            log.info("CV pipeline /summarize failed, falling back to agent")
            events = extract_events_via_agent(
                sensor_id, Path(video_path).name, expected,
            )

        adjudicated = False
        uncertain = needs_adjudication(events)
        if uncertain:
            try:
                refined = adjudicate(sensor_id, uncertain)
                event_map = {(e.player_id, e.t_start): e for e in events}
                for r in refined:
                    event_map[(r.player_id, r.t_start)] = r
                events = sorted(event_map.values(), key=lambda e: (e.player_id, e.t_start))
                adjudicated = True
            except Exception:
                log.warning("Adjudicator failed, using raw events")

        players = validate(events, expected, time_limit=TIME_LIMIT, lenient=lenient)

        for p in players:
            idx = p.player_id - 1
            if 0 <= idx < len(player_names) and player_names[idx].strip():
                p.name = player_names[idx].strip()

        winner = next((p for p in players if p.passed), None)

        return GameResult(
            expected=expected,
            players=players,
            winner=winner,
            raw_events=events,
            adjudicated=adjudicated,
        )
    finally:
        delete_video(sensor_id)


# -- Routes -------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    result: GameResult | None = None
    sequence = generate_sequence()

    if request.method == "POST":
        seq_json = request.form.get("sequence_json", "")
        if seq_json:
            try:
                sequence = json.loads(seq_json)
            except json.JSONDecodeError:
                pass

        lenient = request.form.get("lenient") == "on"
        names_csv = request.form.get("player_names", "")
        player_names = [n.strip() for n in names_csv.split(",") if n.strip()]

        video = request.files.get("video")
        if not video or video.filename == "":
            flash("Please select a video file.", "error")
            return _render_index(sequence)

        if not allowed_file(video.filename):
            flash(f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED_EXT))}", "error")
            return _render_index(sequence)

        safe_name = secure_filename(video.filename)
        unique_name = f"{uuid.uuid4().hex}_{safe_name}"
        save_path = UPLOAD_DIR / unique_name

        video.save(save_path)
        mp4_path = save_path

        try:
            mp4_path = ensure_mp4(save_path)
            result = run_pipeline(str(mp4_path), sequence, player_names, lenient=lenient)

            db = get_db()
            db.execute(
                """INSERT INTO games
                   (played_at, challenge, winner_name, total_players,
                    results_json, events_json, adjudicated)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(sequence),
                    result.winner.name if result.winner else None,
                    len(result.players),
                    json.dumps([asdict(p) for p in result.players]),
                    json.dumps([asdict(e) for e in result.raw_events]),
                    int(result.adjudicated),
                ),
            )
            db.commit()
        except Exception as exc:
            log.exception("Pipeline failed")
            flash(f"Processing error: {exc}", "error")
        finally:
            save_path.unlink(missing_ok=True)
            if mp4_path != save_path:
                mp4_path.unlink(missing_ok=True)

    return _render_index(sequence, result)


def _render_index(sequence: list[str], result: GameResult | None = None):
    return render_template(
        "index.html",
        sequence=sequence,
        emojis=ACTION_EMOJIS,
        vocab=ACTION_VOCAB,
        result=result,
        sequence_json=json.dumps(sequence),
        time_limit=TIME_LIMIT,
    )


@app.route("/leaderboard")
def leaderboard():
    db = get_db()
    rankings = db.execute("""
        SELECT winner_name AS name, COUNT(*) AS wins
        FROM games WHERE winner_name IS NOT NULL
        GROUP BY winner_name ORDER BY wins DESC LIMIT 20
    """).fetchall()

    recent = db.execute("""
        SELECT played_at, challenge, winner_name, total_players, adjudicated
        FROM games ORDER BY id DESC LIMIT 10
    """).fetchall()

    return render_template("leaderboard.html", rankings=rankings, recent=recent)


@app.route("/api/sequence", methods=["POST"])
def api_new_sequence():
    """Generate a new random sequence (for AJAX refresh)."""
    seq = generate_sequence()
    return json.dumps(seq), 200, {"Content-Type": "application/json"}


@app.errorhandler(413)
def too_large(_):
    flash(f"File too large (max {MAX_MB} MB).", "error")
    return redirect(url_for("index")), 413
