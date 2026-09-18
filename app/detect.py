"""Perception Agent -- deterministic, local pose-based action detection.

Runs YOLOv8n-pose (ONNX, via cv2.dnn on CPU) over the uploaded video,
tracks each player by left-to-right column, and classifies each player's
action per time slice from COCO-17 keypoint geometry. Emits the same
ActionEvent contract the validator consumes -- no VSS, no LLM, no network.
"""

from __future__ import annotations

import logging
import math
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from models import ActionEvent

log = logging.getLogger(__name__)

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "yolov8n-pose.onnx"
MODEL_URL = "https://huggingface.co/Xenova/yolov8n-pose/resolve/main/onnx/model.onnx"

INPUT_SIZE = 640
PERSON_CONF = 0.5
NMS_TH = 0.45
KP_VIS = 0.3

# COCO-17 keypoint indices
NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
LSHO, RSHO, LELB, RELB, LWRI, RWRI = 5, 6, 7, 8, 9, 10
LHIP, RHIP, LKNE, RKNE, LANK, RANK = 11, 12, 13, 14, 15, 16

_net: cv2.dnn.Net | None = None


def _load_net() -> cv2.dnn.Net:
    global _net
    if _net is None:
        if not MODEL_PATH.is_file():
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            log.info("Downloading pose model to %s", MODEL_PATH)
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        _net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
    return _net


def _letterbox(frame: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    h, w = frame.shape[:2]
    r = INPUT_SIZE / max(h, w)
    nw, nh = round(w * r), round(h * r)
    resized = cv2.resize(frame, (nw, nh))
    canvas = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, np.uint8)
    px, py = (INPUT_SIZE - nw) // 2, (INPUT_SIZE - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized
    return canvas, r, px, py


def _pose(frame: np.ndarray) -> list[dict]:
    """Return [{cx, box_conf, kp:(17,3)}] for each detected person in image coords."""
    net = _load_net()
    canvas, r, px, py = _letterbox(frame)
    blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True)
    net.setInput(blob)
    out = net.forward()[0].T  # (8400, 56)

    boxes, scores, kpsets = [], [], []
    for row in out:
        conf = float(row[4])
        if conf < PERSON_CONF:
            continue
        cx, cy, bw, bh = row[:4]
        boxes.append([cx - bw / 2, cy - bh / 2, bw, bh])
        scores.append(conf)
        kp = row[5:].reshape(17, 3).copy()
        kp[:, 0] = (kp[:, 0] - px) / r
        kp[:, 1] = (kp[:, 1] - py) / r
        kpsets_row = kp
        kpsets.append(kpsets_row)

    if not boxes:
        return []

    idxs = cv2.dnn.NMSBoxes(boxes, scores, PERSON_CONF, NMS_TH)
    people = []
    for i in np.array(idxs).flatten():
        bx, by, bw, bh = boxes[i]
        people.append({
            "cx": (bx + bw / 2 - px) / r,
            "box_conf": scores[i],
            "kp": kpsets[i],
            "box": (
                (bx - px) / r,
                (by - py) / r,
                (bx + bw - px) / r,
                (by + bh - py) / r,
            ),
        })
    return people


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return math.hypot(float(a[0] - b[0]), float(a[1] - b[1]))


def _classify(kp: np.ndarray) -> tuple[str, float]:
    """Map COCO-17 keypoints to one SEQUENCE_ACTIONS label (or idle)."""
    vis = kp[:, 2]

    def seen(*idx) -> bool:
        return all(vis[i] > KP_VIS for i in idx)

    sw = _dist(kp[LSHO], kp[RSHO]) or 1.0
    sho_y = (kp[LSHO, 1] + kp[RSHO, 1]) / 2
    hip_y = (kp[LHIP, 1] + kp[RHIP, 1]) / 2

    # turn_around: shoulders visible but face keypoints hidden (back to camera)
    if seen(LSHO, RSHO):
        face_vis = (vis[NOSE] + vis[LEYE] + vis[REYE]) / 3
        if face_vis < 0.35:
            return "turn_around", 0.8

    # hands_on_head: both wrists lifted to head level, near the head horizontally
    if seen(LWRI, RWRI, LSHO, RSHO, NOSE):
        wrists_up = kp[LWRI, 1] < sho_y and kp[RWRI, 1] < sho_y
        at_head = kp[LWRI, 1] > kp[NOSE, 1] - 1.5 * sw and kp[RWRI, 1] > kp[NOSE, 1] - 1.5 * sw
        near_x = abs(kp[LWRI, 0] - kp[NOSE, 0]) < 1.3 * sw and abs(kp[RWRI, 0] - kp[NOSE, 0]) < 1.3 * sw
        if wrists_up and at_head and near_x:
            return "hands_on_head", 0.85

    # clap: both wrists close together, at chest height (between shoulders and hips)
    if seen(LWRI, RWRI, LSHO, RSHO, LHIP, RHIP):
        wy = (kp[LWRI, 1] + kp[RWRI, 1]) / 2
        if abs(kp[LWRI, 0] - kp[RWRI, 0]) < 0.5 * sw and sho_y <= wy <= hip_y + 0.3 * abs(hip_y - sho_y):
            return "clap", 0.75

    # crouch: knees drawn up toward hips (bent legs shorten the thigh span)
    if seen(LSHO, RSHO, LHIP, RHIP, LKNE, RKNE):
        knee_y = (kp[LKNE, 1] + kp[RKNE, 1]) / 2
        torso = abs(hip_y - sho_y) + 1.0
        if (knee_y - hip_y) / torso < 0.5:
            return "crouch", 0.7

    return "idle", 0.5


def extract_events(
    video_path: str,
    dt: float = 0.2,
    min_samples: int = 2,
) -> list[ActionEvent]:
    """Detect per-player action events from a local video file.

    Players are numbered left -> right (1 = leftmost). Consecutive frames of
    the same action collapse into a single event; flickers shorter than
    ``min_samples`` samples are dropped as idle.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps * dt))

    frames: list[tuple[float, list[dict]]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            t = idx / fps
            frames.append((t, _pose(frame)))
        idx += 1
    cap.release()

    counts = [len(p) for _, p in frames if p]
    if not counts:
        return []
    num_players = Counter(counts).most_common(1)[0][0]

    # Column centers from frames that see every player, ordered left -> right.
    cols = [[] for _ in range(num_players)]
    for _, people in frames:
        if len(people) == num_players:
            for rank, p in enumerate(sorted(people, key=lambda d: d["cx"])):
                cols[rank].append(p["cx"])
    if any(not c for c in cols):
        # Fall back to overall left->right ordering if no clean frame exists.
        centers = sorted(p["cx"] for _, people in frames for p in people)
        centers = centers[:: max(1, len(centers) // num_players)][:num_players]
    else:
        centers = [float(np.mean(c)) for c in cols]

    # Per-player timeline: nearest-column assignment, then per-time action.
    timelines: dict[int, list[tuple[float, str, float]]] = {i + 1: [] for i in range(num_players)}
    for t, people in frames:
        claimed: dict[int, dict] = {}
        for p in people:
            pid = min(range(num_players), key=lambda i: abs(p["cx"] - centers[i])) + 1
            if pid not in claimed or p["box_conf"] > claimed[pid]["box_conf"]:
                claimed[pid] = p
        for pid in range(1, num_players + 1):
            if pid in claimed:
                action, conf = _classify(claimed[pid]["kp"])
            else:
                action, conf = "idle", 0.5
            timelines[pid].append((t, action, conf))

    events: list[ActionEvent] = []
    for pid, samples in timelines.items():
        events.extend(_segments(pid, samples, dt, min_samples))
    return events


def _segments(
    pid: int,
    samples: list[tuple[float, str, float]],
    dt: float,
    min_samples: int,
) -> list[ActionEvent]:
    """Collapse a per-player sample stream into deduped action events."""
    events: list[ActionEvent] = []
    i = 0
    n = len(samples)
    while i < n:
        action = samples[i][1]
        j = i
        confs = []
        while j < n and samples[j][1] == action:
            confs.append(samples[j][2])
            j += 1
        run = j - i
        if action != "idle" and run >= min_samples:
            events.append(ActionEvent(
                player_id=pid,
                action=action,
                confidence=float(max(confs)),
                t_start=round(samples[i][0], 2),
                t_end=round(samples[j - 1][0] + dt, 2),
            ))
        i = j
    return events


def winner_snapshot(
    video_path: str,
    player_id: int,
    out_path: str,
    label: str | None = None,
    dt: float = 0.3,
) -> bool:
    """Save a frame with the winning player boxed. Returns True on success.

    Picks the frame where the video sees its usual number of players and the
    target player (by left->right rank) is most confidently detected.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps * dt))

    samples: list[tuple[int, list[dict]]] = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            people = _pose(frame)
            if people:
                samples.append((idx, people))
        idx += 1

    if not samples:
        cap.release()
        return False

    num_players = Counter(len(p) for _, p in samples).most_common(1)[0][0]

    best = None  # (box_conf, frame_idx, box)
    for fidx, people in samples:
        if len(people) < player_id:
            continue
        ordered = sorted(people, key=lambda d: d["cx"])
        target = ordered[player_id - 1]
        if len(people) == num_players and (best is None or target["box_conf"] > best[0]):
            best = (target["box_conf"], fidx, target["box"])
    if best is None:
        cap.release()
        return False

    cap.set(cv2.CAP_PROP_POS_FRAMES, best[1])
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return False

    x1, y1, x2, y2 = (int(round(v)) for v in best[2])
    green = (0, 220, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), green, 3)
    tag = label or f"Player {player_id}"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    ty = max(y1, th + 12)
    cv2.rectangle(frame, (x1, ty - th - 10), (x1 + tw + 12, ty + 4), green, -1)
    cv2.putText(frame, tag, (x1 + 6, ty - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    return bool(cv2.imwrite(out_path, frame))
