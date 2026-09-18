"""MoveMatch perception -- ports the methods from test.ipynb.

VSS path: dense VLM captions from the RTVLM (``/generate_vlm_captions`` at :8100)
give a timestamped, per-person timeline; a text NIM (``/v1/chat/completions`` at
:38011) then reasons over that timeline to pick the FIRST person to perform a
move / complete a sequence in order. This is ``caption_video`` + ``first_performer``.

Winner snapshot: YOLOX person detection boxes the winner at the winning moment.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import requests

log = logging.getLogger(__name__)

# -- Endpoints (same env contract as the notebook) ----------------------------
RTVLM = os.environ.get("VSS_RTVLM_URL", "http://127.0.0.1:8100").rstrip("/")
LLM = os.environ.get("VSS_LLM_URL", "http://127.0.0.1:38011").rstrip("/")

MODEL_DIR = Path(__file__).parent.parent / "models"
YOLOX_ONNX = os.environ.get("YOLOX_ONNX", str(MODEL_DIR / "yolox.onnx"))
_MODEL_URLS = {
    YOLOX_ONNX: ("https://media.githubusercontent.com/media/opencv/opencv_zoo/"
                 "main/models/object_detection_yolox/"
                 "object_detection_yolox_2022nov.onnx"),
}

CAPTION_PROMPT = (
    "Number each person from left to right. Write a dense caption describing "
    "every action, pose, and dance move each person performs, always referring "
    "to a person by their number."
)

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _ensure_model(path: str) -> str:
    if not os.path.exists(path) or os.path.getsize(path) < 1_000_000:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        log.info("Downloading model %s", path)
        urllib.request.urlretrieve(_MODEL_URLS[path], path)
    return path


# =============================================================================
# VSS path -- VLM captions + text-NIM reasoning
# =============================================================================

def _first_model(url: str, path: str) -> str:
    return requests.get(f"{url}{path}", timeout=15).json()["data"][0]["id"]


def caption_video(path, chunk_duration: int = 2, chunk_overlap_duration: int = 1):
    """Upload a clip and densely caption it. Returns ordered [(start, end, text)].

    chunk_duration is the game's TEMPORAL RESOLUTION: each chunk yields one
    timestamped caption. chunk_overlap_duration slides the window so a move
    straddling a chunk boundary still lands inside one caption.
    num_frames_per_chunk stays <=5 (cosmos3 caps a prompt at 5 images).
    """
    path = Path(path)
    with path.open("rb") as f:
        file_id = requests.post(
            f"{RTVLM}/files",
            files={"file": (path.name, f, "video/mp4")},
            data={"purpose": "vision", "media_type": "video"},
            timeout=(15, 300),
        ).json()["id"]
    try:
        chunks = requests.post(
            f"{RTVLM}/generate_vlm_captions",
            json={
                "id": file_id,
                "model": _first_model(RTVLM, "/models"),
                "prompt": CAPTION_PROMPT,
                "chunk_duration": chunk_duration,
                "chunk_overlap_duration": chunk_overlap_duration,
                "num_frames_per_chunk": 5,
            },
            timeout=(15, 900),
        ).json()["chunk_responses"]
    finally:
        requests.delete(f"{RTVLM}/files/{file_id}", timeout=30)

    caps = [
        (float(c["start_time"]), float(c["end_time"]),
         _THINK.sub("", c["content"]).strip())
        for c in chunks
    ]
    return sorted(caps, key=lambda c: c[0])


def first_performer(captions, moves):
    """First person to perform a move, or to complete a sequence in order.

    ``moves`` is a move string, or a list of moves to be done in order. One call
    to the text NIM over the whole timestamped timeline (NOT VSS RAG chat, whose
    retrieval is empty here). Returns
    ``{"winner": int, "timestamp": float|None, "num_people": int}`` where
    ``winner == 0`` means nobody did it.
    """
    if isinstance(moves, str):
        task = f"perform the move: '{moves}'"
    else:
        task = "complete ALL of these moves, in this order: " + \
               "; then ".join(moves)

    timeline = "\n".join(f"[{s} - {e}] {t}" for s, e, t in captions if t)
    prompt = (
        "You are given timestamped video captions. People are numbered "
        "left-to-right and keep the same number across time.\n\n"
        f"TIMELINE:\n{timeline}\n\n"
        f"Find the FIRST person (earliest timestamp) to {task}.\n"
        "Answer with ONLY JSON, no prose:\n"
        '{"winner": <person number, or 0 if nobody>, '
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
    return {
        "winner": _num(parsed.get("winner"), int) or 0,
        "timestamp": _num(parsed.get("timestamp"), float),
        "num_people": _num(parsed.get("num_people"), int) or 0,
    }


# =============================================================================
# YOLOX person detection (for the winner snapshot box)
# =============================================================================
_YOLOX_INP = 640
_YOLOX_STRIDES = (8, 16, 32)
_yolox_net = None


def _yolox():
    global _yolox_net
    if _yolox_net is None:
        _yolox_net = cv2.dnn.readNetFromONNX(_ensure_model(YOLOX_ONNX))
    return _yolox_net


def _yolox_grids():
    grids, strides = [], []
    for s in _YOLOX_STRIDES:
        g = _YOLOX_INP // s
        xv, yv = np.meshgrid(np.arange(g), np.arange(g))
        grid = np.stack((xv, yv), 2).reshape(-1, 2)
        grids.append(grid)
        strides.append(np.full((grid.shape[0], 1), s))
    return np.concatenate(grids, 0), np.concatenate(strides, 0)


_GRID, _EXP = _yolox_grids()


def detect_people(frame, conf_th=0.35, nms_th=0.45):
    """Detect people in a BGR frame. Returns [(x0, y0, x1, y1), ...] in pixels,
    sorted left-to-right so index N-1 is 'person N' (the caption numbering)."""
    h, w = frame.shape[:2]
    r = min(_YOLOX_INP / h, _YOLOX_INP / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    canvas = np.full((_YOLOX_INP, _YOLOX_INP, 3), 114, np.uint8)
    canvas[:nh, :nw] = cv2.resize(frame, (nw, nh))

    net = _yolox()
    net.setInput(cv2.dnn.blobFromImage(canvas, 1.0, (_YOLOX_INP, _YOLOX_INP),
                                       swapRB=False, crop=False))
    out = net.forward()[0]                          # [8400, 85]
    xy = (out[:, :2] + _GRID) * _EXP
    wh = np.exp(out[:, 2:4]) * _EXP
    scores = out[:, 4:5] * out[:, 5:]               # obj_conf * class_conf
    cls = np.argmax(scores, 1)
    conf = scores[np.arange(len(scores)), cls]
    keep = (cls == 0) & (conf > conf_th)            # class 0 == person (COCO)
    xy, wh, conf = xy[keep], wh[keep], conf[keep]
    boxes = np.concatenate([xy - wh / 2, wh], 1) / r  # xywh in original pixels

    idx = cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), conf_th, nms_th)
    kept = [boxes[i] for i in np.array(idx).flatten()] if len(idx) else []
    kept.sort(key=lambda b: b[0])                   # left-to-right
    return [(int(x), int(y), int(x + bw), int(y + bh)) for x, y, bw, bh in kept]


def winner_snapshot(video_path, timestamp, winner_number, num_people, out_path):
    """Still frame at the winning moment with the winner boxed. Returns out_path.

    Boxes the winner with a real person-detection box (YOLOX) for that frame,
    matching person N to the Nth detection from the left. If the detector
    doesn't find at least winner_number people, falls back to an equal-column
    split so an image is still produced.
    """
    cap = cv2.VideoCapture(str(video_path))
    if timestamp is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {video_path}")

    h, w = frame.shape[:2]
    people = detect_people(frame)
    if len(people) >= winner_number:
        x0, y0, x1, y1 = people[winner_number - 1]
    else:
        col = w / max(num_people, 1)
        x0, y0, x1, y1 = (int((winner_number - 1) * col) + 6, 6,
                          int(winner_number * col) - 6, h - 6)

    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    color = (0, 215, 255)  # amber, BGR
    ts = "?" if timestamp is None else f"{timestamp}"
    label = f"WINNER: Person {winner_number} @ {ts}s"
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

    Runs the VSS path: VLM captions + text-NIM reasoning over the timeline.
    Returns a dict with keys ``winner, timestamp, num_people, method, timeline``.
    """
    target = moves[0] if isinstance(moves, (list, tuple)) and len(moves) == 1 else moves
    captions = caption_video(video_path, chunk_duration=chunk_duration,
                             chunk_overlap_duration=chunk_overlap_duration)
    res = first_performer(captions, target)
    res["timeline"] = captions
    res["method"] = "vss"
    return res
