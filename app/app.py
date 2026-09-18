"""MoveMatch -- Flask orchestrator.

Generates a random challenge, routes the uploaded/recorded clip through the VSS
perception path (VLM captions + text-NIM reasoning, see vss.py), with a
deterministic pose fallback, and stores the winner (plus a boxed snapshot) for
the leaderboard.
"""

import json
import logging
import os
import random
import sqlite3
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, render_template, request, flash, redirect, url_for, g
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

from models import MOVE_POOL, GameResult, move_emoji
from vss import find_winner, fetch_cv_metadata, start_order, winner_snapshot_cv

log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
SNAPSHOT_DIR = BASE_DIR / "static" / "snapshots"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = BASE_DIR / "movematch.db"

ALLOWED_EXT = {"mp4", "mov", "avi", "mkv", "webm"}
MAX_MB = 50
SEQUENCE_LENGTH = 2
CHUNK_DURATION = 2
CHUNK_OVERLAP = 1
RECORD_SECONDS = 8

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-change-me")
app.config["UPLOAD_FOLDER"] = str(UPLOAD_DIR)
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024


def current_url_prefix() -> str:
    prefix = (
        request.headers.get("X-Forwarded-Prefix")
        or request.script_root
        or os.environ.get("SCRIPT_NAME", "")
    ).rstrip("/")

    if not prefix:
        host = request.headers.get("X-Forwarded-Host", request.host)
        port = request.environ.get("SERVER_PORT", "")
        if port and ".apps." in host:
            prefix = f"/coder/proxy/{port}"

    return prefix


@app.context_processor
def inject_helpers():
    return {"url_prefix": current_url_prefix(), "move_emoji": move_emoji}


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
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            played_at     TEXT    NOT NULL,
            challenge     TEXT    NOT NULL,
            winner_name   TEXT,
            winner_number INTEGER DEFAULT 0,
            num_people    INTEGER DEFAULT 0,
            timestamp     REAL,
            method        TEXT,
            winner_image  TEXT
        )
    """)
    # Migrate older databases that predate these columns.
    cols = {r[1] for r in db.execute("PRAGMA table_info(games)")}
    for name, ddl in [
        ("winner_number", "INTEGER DEFAULT 0"),
        ("num_people", "INTEGER DEFAULT 0"),
        ("timestamp", "REAL"),
        ("method", "TEXT"),
        ("winner_image", "TEXT"),
    ]:
        if name not in cols:
            db.execute(f"ALTER TABLE games ADD COLUMN {name} {ddl}")
    db.commit()
    db.close()


init_db()


# -- Helpers ------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def generate_challenge() -> list[str]:
    return random.sample(MOVE_POOL, min(SEQUENCE_LENGTH, len(MOVE_POOL)))


def ensure_mp4(src: Path) -> Path:
    """Re-encode to H.264/AAC MP4 so the VSS uploader accepts it."""
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


def run_pipeline(video_path: str, moves: list[str], player_names: list[str]) -> GameResult:
    """VSS perception -> winner (tracker id), named + boxed from CV metadata.

    The winner comes back as a stable tracker id. We map it to a typed name by
    STARTING position (people ordered left-to-right by their first-seen bbox), and
    box it straight from the tracker's own bbox so the box matches the caption.
    """
    res = find_winner(video_path, moves, chunk_duration=CHUNK_DURATION,
                      chunk_overlap_duration=CHUNK_OVERLAP)

    winner_id = int(res.get("winner", -1))
    timestamp = res.get("timestamp")
    num_people = int(res.get("num_people") or 0)

    # tracker ids are 0-based (0 is a real person), so "nobody" is timestamp None.
    if timestamp is None or winner_id < 0:
        return GameResult(
            moves=moves,
            winner_number=0,
            winner_name=None,
            timestamp=None,
            num_people=num_people,
            method=res.get("method", "vss"),
            winner_image=None,
            timeline=res.get("timeline", []),
        )

    winner_number = 0
    winner_name = None
    winner_image = None
    meta_path = None
    try:
        meta_path = fetch_cv_metadata(res["request_id"])
        ranks, n_meta = start_order(meta_path)
        num_people = n_meta or num_people
        winner_number = ranks.get(winner_id, -1) + 1  # 1-based start rank; 0 if absent

        idx = winner_number - 1
        if winner_number and 0 <= idx < len(player_names) and player_names[idx].strip():
            winner_name = player_names[idx].strip()
        elif winner_number:
            winner_name = f"Person {winner_number}"
        else:
            winner_name = f"Player {winner_id}"

        fname = f"{uuid.uuid4().hex}.jpg"
        winner_snapshot_cv(video_path, timestamp, winner_id, meta_path,
                           SNAPSHOT_DIR / fname)
        winner_image = fname
    except Exception:
        log.warning("CV winner resolution failed", exc_info=True)
        if winner_name is None:
            winner_name = f"Player {winner_id}"
    finally:
        if meta_path:
            Path(meta_path).unlink(missing_ok=True)

    return GameResult(
        moves=moves,
        winner_number=winner_number,
        winner_name=winner_name,
        timestamp=timestamp,
        num_people=num_people,
        method=res.get("method", "vss"),
        winner_image=winner_image,
        timeline=res.get("timeline", []),
    )


def save_game(result: GameResult) -> None:
    db = get_db()
    db.execute(
        """INSERT INTO games
           (played_at, challenge, winner_name, winner_number, num_people,
            timestamp, method, winner_image)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            datetime.now(timezone.utc).isoformat(),
            json.dumps(result.moves),
            result.winner_name,
            result.winner_number,
            result.num_people,
            result.timestamp,
            result.method,
            result.winner_image,
        ),
    )
    db.commit()


def process_upload(moves: list[str], player_names: list[str]):
    """Shared POST handling: validate file, transcode, run pipeline, persist.

    Returns (result, error_message). Exactly one is non-None.
    """
    video = request.files.get("video")
    if not video or video.filename == "":
        return None, "Please select a video file."
    if not allowed_file(video.filename):
        return None, f"Unsupported format. Allowed: {', '.join(sorted(ALLOWED_EXT))}"

    safe_name = secure_filename(video.filename)
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{safe_name}"
    video.save(save_path)
    mp4_path = save_path

    try:
        mp4_path = ensure_mp4(save_path)
        result = run_pipeline(str(mp4_path), moves, player_names)
        save_game(result)
        return result, None
    except Exception as exc:
        log.exception("Pipeline failed")
        return None, f"Processing error: {exc}"
    finally:
        save_path.unlink(missing_ok=True)
        if mp4_path != save_path:
            mp4_path.unlink(missing_ok=True)


def parse_names(csv: str) -> list[str]:
    return [n.strip() for n in csv.split(",") if n.strip()]


# -- Routes -------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    result: GameResult | None = None
    challenge = generate_challenge()

    if request.method == "POST":
        raw = request.form.get("challenge_json", "")
        if raw:
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list) and parsed:
                    challenge = [str(m) for m in parsed]
            except json.JSONDecodeError:
                pass

        player_names = parse_names(request.form.get("player_names", ""))
        result, error = process_upload(challenge, player_names)
        if error:
            flash(error, "error")

    return render_template(
        "index.html",
        challenge=challenge,
        challenge_json=json.dumps(challenge),
        record_seconds=RECORD_SECONDS,
        result=result,
    )


@app.route("/leaderboard")
def leaderboard():
    db = get_db()
    rankings = db.execute("""
        SELECT winner_name AS name, COUNT(*) AS wins
        FROM games WHERE winner_name IS NOT NULL
        GROUP BY winner_name ORDER BY wins DESC LIMIT 20
    """).fetchall()

    winners = db.execute("""
        SELECT played_at, challenge, winner_name, method, winner_image, timestamp
        FROM games
        WHERE winner_image IS NOT NULL
        ORDER BY id DESC LIMIT 24
    """).fetchall()

    def moves_of(row):
        try:
            return json.loads(row["challenge"])
        except (json.JSONDecodeError, TypeError):
            return []

    gallery = [
        {
            "played_at": w["played_at"],
            "moves": moves_of(w),
            "winner_name": w["winner_name"],
            "method": w["method"],
            "winner_image": w["winner_image"],
            "timestamp": w["timestamp"],
        }
        for w in winners
    ]

    return render_template("leaderboard.html", rankings=rankings, gallery=gallery)


@app.route("/test", methods=["GET", "POST"])
def test_page():
    result: GameResult | None = None

    if request.method == "POST":
        try:
            moves = json.loads(request.form.get("moves_json", "[]"))
        except json.JSONDecodeError:
            moves = []
        moves = [str(m).strip() for m in moves if str(m).strip()]

        if not moves:
            flash("Write at least one movement.", "error")
        else:
            player_names = parse_names(request.form.get("player_names", ""))
            result, error = process_upload(moves, player_names)
            if error:
                flash(error, "error")

    return render_template("test.html", result=result)


@app.route("/api/challenge", methods=["POST"])
def api_new_challenge():
    return json.dumps(generate_challenge()), 200, {"Content-Type": "application/json"}


@app.errorhandler(413)
def too_large(_):
    flash(f"File too large (max {MAX_MB} MB).", "error")
    return redirect(url_for("index")), 413
