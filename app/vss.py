"""MoveMatch perception -- ports the methods from test.ipynb.

VSS path: dense VLM captions from the RTVLM (``/generate_vlm_captions`` at :8100)
give a timestamped, per-person timeline; a text NIM (``/v1/chat/completions`` at
:38011) then reasons over that timeline to pick the FIRST person to perform a
move / complete a sequence in order. This is ``caption_video`` + ``first_performer``.

The CV pipeline (GroundingDINO + NvDCF tracker) assigns each person a stable
tracker ID, overlays it on the frames the VLM sees (Set-of-Marks), and writes a
fused CV metadata JSON (id + bbox per frame). Captions refer to people by that
tracker ID, and the winner snapshot is boxed straight from the tracker's bbox --
no separate person detector, so box and caption number always agree.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path

import cv2
import requests

log = logging.getLogger(__name__)

# -- Endpoints (same env contract as the notebook) ----------------------------
RTVLM = os.environ.get("VSS_RTVLM_URL", "http://127.0.0.1:8100").rstrip("/")
LLM = os.environ.get("VSS_LLM_URL", "http://127.0.0.1:38011").rstrip("/")

# via-server container that holds the fused CV metadata JSON. Same container-copy
# mechanism the notebook uses for the overlay video (no API, dir not mounted).
VIA_CONTAINER = os.environ.get(
    "VIA_CONTAINER", "local_deployment_single_gpu-via-server-1"
)

_FUSED_GLOB = "/opt/nvidia/via/*_fused.json"

CAPTION_PROMPT = (
    "Write a dense caption describing every action, pose, "
    "and dance move each person performs, always "
    "referring to a person by the ID drawn on them. "
    "If no one has an ID drawn on them, do not describe the scene. "
    "Do not give any overall picture. Only describe persons, "
    "and the exact movement they are making. "
)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


# =============================================================================
# VSS path -- VLM captions + text-NIM reasoning
# =============================================================================

def _first_model(url: str, path: str) -> str:
    return requests.get(f"{url}{path}", timeout=15).json()["data"][0]["id"]


def _list_fused() -> list[tuple[int, str]]:
    """(mtime_epoch, remote_path) for every fused CV metadata JSON in the
    container, ascending by mtime. The CV pipeline names these with its own
    internal request id (NOT the caption API's response id), so we locate this
    run's file by recency rather than by id."""
    out = subprocess.check_output(
        ["docker", "exec", VIA_CONTAINER, "sh", "-c",
         f'for f in {_FUSED_GLOB}; do [ -e "$f" ] && stat -c "%Y %n" "$f"; done'],
        text=True,
    )
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if line:
            mt, path = line.split(" ", 1)
            rows.append((int(mt), path))
    rows.sort()
    return rows


def caption_video(path, chunk_duration: int = 2, chunk_overlap_duration: int = 1):
    """Upload a clip and densely caption it. Returns ``(captions, cv_marker)``
    where captions is ordered ``[(start, end, text)]``.

    chunk_duration is the game's TEMPORAL RESOLUTION: each chunk yields one
    timestamped caption. chunk_overlap_duration slides the window so a move
    straddling a chunk boundary still lands inside one caption.
    num_frames_per_chunk stays <=5 (cosmos3 caps a prompt at 5 images).

    enable_cv_metadata runs the CV pipeline (GroundingDINO + NvDCF tracker) and
    overlays stable tracking IDs onto the frames the VLM sees (Set-of-Marks), so
    the numbering persists across chunks. cv_marker is the newest fused-file mtime
    BEFORE this run, so fetch_cv_metadata can pick out the file this run created.
    """
    path = Path(path)
    fused_before = _list_fused()
    cv_marker = fused_before[-1][0] if fused_before else 0
    with path.open("rb") as f:
        file_id = requests.post(
            f"{RTVLM}/files",
            files={"file": (path.name, f, "video/mp4")},
            data={"purpose": "vision", "media_type": "video"},
            timeout=(15, 300),
        ).json()["id"]
    try:
        resp = requests.post(
            f"{RTVLM}/generate_vlm_captions",
            json={
                "id": file_id,
                "model": _first_model(RTVLM, "/models"),
                "prompt": CAPTION_PROMPT,
                "chunk_duration": chunk_duration,
                "chunk_overlap_duration": chunk_overlap_duration,
                "num_frames_per_chunk": 5,
                "enable_cv_metadata": True,
                "cv_pipeline_prompt": "person",
            },
            timeout=(15, 900),
        ).json()
    finally:
        requests.delete(f"{RTVLM}/files/{file_id}", timeout=30)

    caps = [
        (float(c["start_time"]), float(c["end_time"]),
         _THINK.sub("", c["content"]).strip())
        for c in resp["chunk_responses"]
    ]
    return sorted(caps, key=lambda c: c[0]), cv_marker


def first_performer(captions, moves):
    """First person to perform a move, or to complete a sequence in order.

    ``moves`` is a move string, or a list of moves to be done in order. One call
    to the text NIM over the whole timestamped timeline (NOT VSS RAG chat, whose
    retrieval is empty here). Returns
    ``{"winner": int, "timestamp": float|None, "num_people": int}`` where
    ``winner`` is a tracker id (0-based) and ``winner == -1`` means nobody did it
    -- test ``timestamp is None`` rather than truthiness, since id 0 is a real
    person.
    """
    if isinstance(moves, str):
        task = f"perform the move: '{moves}'"
    else:
        task = "complete ALL of these moves, in this order: " + \
               "; then ".join(moves)

    timeline = "\n".join(f"[{s} - {e}] {t}" for s, e, t in captions if t)
    prompt = (
        "You are given timestamped video captions. Each person has a fixed ID "
        "number (shown in the captions, e.g. 'Person 0', 'Person 3') that stays "
        "the same across time.\n\n"
        f"TIMELINE:\n{timeline}\n\n"
        f"Find the FIRST person (earliest timestamp) to {task}.\n"
        "Answer with ONLY JSON, no prose:\n"
        '{"winner": <the person\'s id number, or -1 if nobody>, '
        '"timestamp": <seconds when they finish, or null>, '
        '"num_people": <total people visible>}'
    )
    r = requests.post(
        f"{LLM}/v1/chat/completions",
        json={
            "model": _first_model(LLM, "/v1/models"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 200,
        },
        timeout=(15, 300),
    ).json()
    text = _THINK.sub("", r["choices"][0]["message"]["content"])
    parsed = json.loads(re.search(r"\{.*\}", text, re.DOTALL).group())

    # The model sometimes returns numbers as strings -- coerce to real types.
    def _num(v, cast):
        try:
            return cast(v)
        except (TypeError, ValueError):
            return None

    winner = _num(parsed.get("winner"), int)
    return {
        "winner": -1 if winner is None else winner,
        "timestamp": _num(parsed.get("timestamp"), float),
        "num_people": _num(parsed.get("num_people"), int) or 0,
    }


# =============================================================================
# CV metadata -- fetch the fused tracker JSON and box the winner by tracker id
# =============================================================================

def fetch_cv_metadata(cv_marker: int, out_path: str | None = None) -> str:
    """Copy this run's fused CV metadata JSON out of the via-server container.

    The CV pipeline names fused files with its own internal id (not the caption
    API's response id), so we pick the newest fused file created AFTER cv_marker
    (the newest-file mtime captured before the caption call). Falls back to the
    newest file overall if none is strictly newer.
    """
    rows = _list_fused()
    if not rows:
        raise RuntimeError("No fused CV metadata JSON in the container.")
    newer = [r for r in rows if r[0] > cv_marker]
    remote_path = (newer or rows)[-1][1]
    out_path = out_path or f"/tmp/cv_metadata_{os.path.basename(remote_path)}"
    subprocess.check_call(
        ["docker", "cp", f"{VIA_CONTAINER}:{remote_path}", out_path])
    return out_path


def start_order(metadata_path: str):
    """Map each tracker id to a left-to-right rank by STARTING position.

    Iterates frames in time order and records each id's bbox center x the first
    time it appears, then ranks ids by that x. Returns ``({id: rank0}, count)``.
    Starting position is stable even if people move/swap later in the clip.
    """
    meta = json.load(open(metadata_path))
    first_x: dict[int, float] = {}
    for fr in sorted(meta, key=lambda f: f["timestamp"]):
        for o in fr["objects"]:
            if o["id"] not in first_x:
                b = o["bbox"]
                first_x[o["id"]] = (b["lX"] + b["rX"]) / 2
    order = sorted(first_x, key=first_x.get)
    return {tid: i for i, tid in enumerate(order)}, len(order)


def winner_snapshot_cv(video_path, timestamp, winner_id, metadata_path, out_path):
    """Still frame at ``timestamp`` with the winner boxed by tracker id.

    Boxes come from the fused CV metadata (id + bbox), so the box lines up with
    the caption numbering. Picks the metadata frame nearest ``timestamp`` that
    actually contains ``winner_id``. Returns out_path.
    """
    meta = json.load(open(metadata_path))
    t_ns = float(timestamp) * 1e9
    frames = [fr for fr in meta if any(o["id"] == winner_id for o in fr["objects"])]
    if not frames:
        ids = sorted({o["id"] for fr in meta for o in fr["objects"]})
        raise RuntimeError(f"tracker id {winner_id} never appears (ids: {ids}).")
    fr = min(frames, key=lambda f: abs(f["timestamp"] - t_ns))
    obj = next(o for o in fr["objects"] if o["id"] == winner_id)

    cap = cv2.VideoCapture(str(video_path))
    if timestamp is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {video_path}")

    h, w = frame.shape[:2]
    sx, sy = w / fr["frameWidth"], h / fr["frameHeight"]
    b = obj["bbox"]
    x0, y0 = int(b["lX"] * sx), int(b["tY"] * sy)
    x1, y1 = int(b["rX"] * sx), int(b["bY"] * sy)
    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)

    color = (0, 215, 255)  # amber, BGR
    ts = "?" if timestamp is None else f"{timestamp}"
    label = f"WINNER @ {ts}s"
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 4)
    label_y = y0 - 12 if y0 - 12 > 24 else min(h - 8, y1 + 32)
    cv2.putText(frame, label, (x0 + 4, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), frame)
    return out_path


# =============================================================================
# Orchestration
# =============================================================================

def find_winner(video_path, moves, chunk_duration: int = 2,
                chunk_overlap_duration: int = 1) -> dict:
    """Find the first person to perform ``moves`` (a str or ordered list).

    Runs the VSS path: VLM captions (with CV tracking) + text-NIM reasoning over
    the timeline. Returns a dict with keys
    ``winner`` (tracker id, -1 if nobody), ``timestamp``, ``num_people``,
    ``cv_marker``, ``method``, ``timeline``.
    """
    target = moves[0] if isinstance(moves, (list, tuple)) and len(moves) == 1 else moves
    captions, cv_marker = caption_video(
        video_path, chunk_duration=chunk_duration,
        chunk_overlap_duration=chunk_overlap_duration)
    res = first_performer(captions, target)
    res["cv_marker"] = cv_marker
    res["timeline"] = captions
    res["method"] = "vss"
    return res
