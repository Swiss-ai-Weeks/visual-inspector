import json
import os
import random
import re
import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, render_template, request, flash, redirect, url_for, g
from werkzeug.utils import secure_filename

from process_video import query_vss_agent_video

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
DB_PATH = BASE_DIR / "movematch.db"

ALLOWED_EXT = {"mp4", "mov", "avi", "mkv", "webm"}
MAX_MB = 50
SEQUENCE_LENGTH = 3

ACTIONS = [
    "clap",
    "raise both hands",
    "point down",
    "wave",
    "cross arms",
    "hands on hips",
]

ACTION_EMOJIS = {
    "clap": "👏",
    "raise both hands": "🙆",
    "point down": "👇",
    "wave": "👋",
    "cross arms": "🙅",
    "hands on hips": "💪",
}

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-change-me")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024


# ── Database ──

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
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            played_at   TEXT    NOT NULL,
            challenge   TEXT    NOT NULL,
            winner_name TEXT,
            total_players INTEGER DEFAULT 0,
            results_json TEXT
        )
    """)
    db.commit()
    db.close()


init_db()


# ── Helpers ──

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def generate_sequence() -> list[str]:
    return random.sample(ACTIONS, SEQUENCE_LENGTH)


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
    app.logger.info("ffmpeg OK: %s → %s", src.name, dst.name)
    return dst


def sequences_match(expected: list[str], detected: list[str]) -> bool:
    """Deterministic sequence comparison — normalised to lowercase/stripped."""
    norm = [s.strip().lower() for s in expected]
    got = [s.strip().lower() for s in detected]
    return norm == got


def parse_vss_players(raw: str) -> list[dict]:
    """Extract the structured per-player action list from VSS response text."""
    m = re.search(r"\{[\s\S]*\"players\"[\s\S]*\}", raw)
    if not m:
        return []
    try:
        data = json.loads(m.group())
        out = []
        for p in data.get("players", []):
            out.append({
                "id": int(p.get("id", 0)),
                "actions": list(p.get("actions", [])),
            })
        return out
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return []


def evaluate_game(
    raw_response: str,
    expected: list[str],
    player_names: list[str],
) -> tuple[list[dict], dict | None]:
    """
    Parse VSS output, do deterministic matching, assign names, pick winner.
    Returns (players, winner_or_None).
    """
    players = parse_vss_players(raw_response)

    for p in players:
        idx = p["id"] - 1
        if 0 <= idx < len(player_names) and player_names[idx].strip():
            p["name"] = player_names[idx].strip()
        else:
            p["name"] = f"Player {p['id']}"
        p["match"] = sequences_match(expected, p["actions"])

    winner = next((p for p in players if p["match"]), None)
    return players, winner


# ── Routes ──

@app.route("/", methods=["GET", "POST"])
def index():
    players: list[dict] = []
    winner = None
    raw_response = None
    analyzed = False
    sequence = generate_sequence()

    if request.method == "POST":
        seq_json = request.form.get("sequence_json", "")
        if seq_json:
            try:
                sequence = json.loads(seq_json)
            except json.JSONDecodeError:
                pass

        names_csv = request.form.get("player_names", "")
        player_names = [n.strip() for n in names_csv.split(",") if n.strip()]

        video = request.files.get("video")
        if not video or video.filename == "":
            flash("Please select a video file.", "error")
            return render_template(
                "index.html", sequence=sequence, emojis=ACTION_EMOJIS,
                analyzed=False, players=[], winner=None, raw_response=None,
                sequence_json=json.dumps(sequence),
            )

        if not allowed_file(video.filename):
            flash(
                f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED_EXT))}",
                "error",
            )
            return render_template(
                "index.html", sequence=sequence, emojis=ACTION_EMOJIS,
                analyzed=False, players=[], winner=None, raw_response=None,
                sequence_json=json.dumps(sequence),
            )

        safe_name = secure_filename(video.filename)
        unique_name = f"{uuid.uuid4().hex}_{safe_name}"
        save_path = UPLOAD_DIR / unique_name
        video.save(save_path)

        mp4_path = save_path
        try:
            mp4_path = ensure_mp4(save_path)
            raw_response = query_vss_agent_video(str(mp4_path), sequence)
            players, winner = evaluate_game(raw_response, sequence, player_names)
            analyzed = True

            db = get_db()
            db.execute(
                """INSERT INTO games
                   (played_at, challenge, winner_name, total_players, results_json)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(sequence),
                    winner["name"] if winner else None,
                    len(players),
                    json.dumps(players),
                ),
            )
            db.commit()
        except Exception as exc:
            app.logger.exception("Agent query failed")
            flash(f"Processing error: {exc}", "error")
        finally:
            save_path.unlink(missing_ok=True)
            if mp4_path != save_path:
                mp4_path.unlink(missing_ok=True)

    return render_template(
        "index.html",
        sequence=sequence,
        emojis=ACTION_EMOJIS,
        analyzed=analyzed,
        players=players,
        winner=winner,
        raw_response=raw_response,
        sequence_json=json.dumps(sequence),
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
        SELECT played_at, challenge, winner_name, total_players
        FROM games ORDER BY id DESC LIMIT 10
    """).fetchall()

    return render_template("leaderboard.html", rankings=rankings, recent=recent)


@app.errorhandler(413)
def too_large(_):
    flash(f"File too large (max {MAX_MB} MB).", "error")
    return redirect(url_for("index")), 413
