"""MoveMatch perception -- the YOLO path.

Deterministic, pose-based sequence detection. Two ONNX models run through
OpenCV's DNN module (no GPU service, no container round-trip):

  * YOLOX (COCO)      -- person detection, used to box the winner left-to-right.
  * YOLOv8n-pose      -- 17-keypoint COCO pose, used to decide each move from
                         geometry and time the FIRST person to complete the
                         ordered move sequence.

Unlike the VSS path (VLM captions + text-NIM reasoning), nothing here is a
language model: a move is a geometric predicate on the 17 keypoints, so
near-simultaneous moves get an exact per-person onset time. People are numbered
left-to-right by STARTING position, matching the VSS path's ``start_order`` so
typed player names line up the same way in both pipelines.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

# Models live at the repo root (../../models relative to this app dir); the same
# files test.ipynb downloads. Overridable via env for other layouts.
_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
YOLOX_ONNX = os.environ.get("YOLOX_ONNX", str(_MODELS_DIR / "yolox.onnx"))
YOLO_POSE_ONNX = os.environ.get("YOLO_POSE_ONNX", str(_MODELS_DIR / "yolov8n-pose.onnx"))

_YOLOX_INP = 640
_YOLOX_STRIDES = (8, 16, 32)
_POSE_INP = 640

_yolox_net = None
_pose_net = None

# COCO keypoints: 0 nose  3/4 ears  5/6 shoulders  7/8 elbows  9/10 wrists
#                 11/12 hips  13/14 knees  15/16 ankles
NOSE, LEAR, REAR = 0, 3, 4
LSH, RSH, LEL, REL, LWR, RWR = 5, 6, 7, 8, 9, 10
LHIP, RHIP, LKNEE, RKNEE = 11, 12, 13, 14


# =============================================================================
# Model loading (YOLOX + YOLOv8-pose via OpenCV DNN)
# =============================================================================

def _yolox():
    global _yolox_net
    if _yolox_net is None:
        _yolox_net = cv2.dnn.readNetFromONNX(YOLOX_ONNX)
    return _yolox_net


def _pose():
    global _pose_net
    if _pose_net is None:
        _pose_net = cv2.dnn.readNetFromONNX(YOLO_POSE_ONNX)
    return _pose_net


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


# =============================================================================
# Detection + pose
# =============================================================================

def detect_people(frame, conf_th=0.35, nms_th=0.45):
    """Detect people in a BGR frame. Returns ``[(x0, y0, x1, y1), ...]`` in
    pixels, sorted left-to-right so index N-1 is 'person N'."""
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


def pose_people(frame, conf_th=0.5, nms_th=0.45):
    """People with 17 COCO keypoints. Returns ``[(center_x, kp[17,3]), ...]``
    sorted left-to-right; each kp row is ``(x_px, y_px, visibility)``."""
    h, w = frame.shape[:2]
    r = min(_POSE_INP / h, _POSE_INP / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    canvas = np.full((_POSE_INP, _POSE_INP, 3), 114, np.uint8)
    canvas[:nh, :nw] = cv2.resize(frame, (nw, nh))
    net = _pose()
    net.setInput(cv2.dnn.blobFromImage(canvas, 1 / 255.0, (_POSE_INP, _POSE_INP),
                                       swapRB=True, crop=False))
    out = net.forward()[0].T                        # [8400, 56]
    conf = out[:, 4]
    keep = conf > conf_th
    out, conf = out[keep], conf[keep]
    if len(out) == 0:
        return []
    boxes = out[:, :4].copy()
    boxes[:, 0] -= boxes[:, 2] / 2
    boxes[:, 1] -= boxes[:, 3] / 2
    idx = cv2.dnn.NMSBoxes((boxes / r).tolist(), conf.tolist(), conf_th, nms_th)
    people = []
    for i in np.array(idx).flatten():
        kp = out[i, 5:].reshape(17, 3).copy()
        kp[:, 0] /= r
        kp[:, 1] /= r
        people.append((float(out[i, 0] / r), kp))
    people.sort(key=lambda p: p[0])
    return people


# =============================================================================
# Move detectors -- single-frame pose geometry, lenient thresholds
# =============================================================================
# Predicates take one person's kp[17,3] and return True when the pose matches.
# Everything is scaled by shoulder width / torso height so it is resolution- and
# distance-independent, and gated on keypoint visibility. Thresholds are tuned
# generous on purpose: a genuine attempt should count in a party game.

_VIS_TH = 0.35


def _vis(kp, *idx):
    return min(kp[i, 2] for i in idx) >= _VIS_TH


def _scale(kp):
    """(shoulder_width, torso_height, mid_shoulder_y). torso is |hip-shoulder|."""
    sh_w = abs(kp[LSH, 0] - kp[RSH, 0])
    sh_y = (kp[LSH, 1] + kp[RSH, 1]) / 2
    hip_y = (kp[LHIP, 1] + kp[RHIP, 1]) / 2
    torso = abs(hip_y - sh_y) + 1e-3
    return sh_w, torso, sh_y


def arms_crossed(kp, ratio_th=0.55):
    """Wrists pulled together over the torso, at/below shoulder height."""
    if not _vis(kp, LSH, RSH, LWR, RWR):
        return False
    sh_w, torso, sh_y = _scale(kp)
    if sh_w < 5:
        return False
    gap = (kp[LWR, 0] - kp[RWR, 0]) / sh_w
    at_chest = (kp[LWR, 1] > sh_y - 0.1 * torso) and (kp[RWR, 1] > sh_y - 0.1 * torso)
    return gap < ratio_th and at_chest


def _one_wrist_up(kp, margin=0.05):
    """A wrist raised above its own shoulder (lenient)."""
    sh_w, torso, _ = _scale(kp)
    left = _vis(kp, LSH, LWR) and kp[LWR, 1] < kp[LSH, 1] - margin * torso
    right = _vis(kp, RSH, RWR) and kp[RWR, 1] < kp[RSH, 1] - margin * torso
    return left or right


def raise_one_arm(kp):
    return _one_wrist_up(kp, margin=0.1)


def wave(kp):
    # Approximation: a raised hand. No temporal oscillation check on purpose.
    return _one_wrist_up(kp, margin=0.05)


def hands_on_head(kp):
    """Both wrists up around head level and horizontally over the head."""
    if not _vis(kp, LSH, RSH, LWR, RWR, NOSE):
        return False
    sh_w, torso, sh_y = _scale(kp)
    if sh_w < 5:
        return False
    both_up = (kp[LWR, 1] < sh_y + 0.15 * torso) and (kp[RWR, 1] < sh_y + 0.15 * torso)
    span = 0.9 * sh_w
    over_head = (abs(kp[LWR, 0] - kp[NOSE, 0]) < span) and \
                (abs(kp[RWR, 0] - kp[NOSE, 0]) < span)
    return both_up and over_head


def _near(kp, wrist, target, frac, dim):
    if not _vis(kp, wrist, target):
        return False
    dx = kp[wrist, 0] - kp[target, 0]
    dy = kp[wrist, 1] - kp[target, 1]
    return (dx * dx + dy * dy) ** 0.5 < frac * dim


def touch_head(kp):
    sh_w, _, _ = _scale(kp)
    reach = 0.7 * sh_w
    return any(_near(kp, wr, hd, 0.7, sh_w)
               for wr in (LWR, RWR) for hd in (NOSE, LEAR, REAR))


def touch_knee(kp):
    _, torso, _ = _scale(kp)
    return any(_near(kp, wr, kn, 0.7, torso)
               for wr in (LWR, RWR) for kn in (LKNEE, RKNEE))


def crouch(kp):
    """Hips lowered toward the knees -- the knee-to-hip vertical gap collapses.

    Standing: knees sit ~a torso-height below the hips. Crouching pulls that gap
    in, so a small (knee_y - hip_y)/torso means bent legs / low stance.
    """
    if not _vis(kp, LHIP, RHIP, LKNEE, RKNEE):
        return False
    _, torso, _ = _scale(kp)
    hip_y = (kp[LHIP, 1] + kp[RHIP, 1]) / 2
    knee_y = (kp[LKNEE, 1] + kp[RKNEE, 1]) / 2
    return (knee_y - hip_y) < 0.55 * torso


def clap(kp):
    """Wrists brought together, centered, at chest height (approx of a clap)."""
    if not _vis(kp, LSH, RSH, LWR, RWR):
        return False
    sh_w, torso, sh_y = _scale(kp)
    if sh_w < 5:
        return False
    gap = abs(kp[LWR, 0] - kp[RWR, 0]) / sh_w
    mid_sh_x = (kp[LSH, 0] + kp[RSH, 0]) / 2
    wrists_x = (kp[LWR, 0] + kp[RWR, 0]) / 2
    centered = abs(wrists_x - mid_sh_x) < 0.6 * sh_w
    at_chest = (kp[LWR, 1] > sh_y) and (kp[RWR, 1] > sh_y)
    return gap < 0.45 and centered and at_chest


def point_at_camera(kp):
    """Approximation: one arm extended roughly horizontally out from the body."""
    sh_w, torso, _ = _scale(kp)
    if sh_w < 5:
        return False
    for sh, wr in ((LSH, LWR), (RSH, RWR)):
        if not _vis(kp, sh, wr):
            continue
        level = abs(kp[wr, 1] - kp[sh, 1]) < 0.35 * torso  # near shoulder height
        reach = abs(kp[wr, 0] - kp[sh, 0]) > 0.4 * sh_w    # extended outward
        if level and reach:
            return True
    return False


# Keyword -> detector, most specific first (all keywords must be present).
_DETECTOR_RULES = [
    (("cross", "arm"), arms_crossed),
    (("touch", "head"), touch_head),
    (("hands", "head"), hands_on_head),
    (("hand", "head"), hands_on_head),
    (("touch", "knee"), touch_knee),
    (("raise", "arm"), raise_one_arm),
    (("crouch",), crouch),
    (("clap",), clap),
    (("wave",), wave),
    (("point",), point_at_camera),
]


def match_move(move: str):
    """Return the pose detector for a (free-text) move, or None if unsupported."""
    low = move.lower()
    for keys, fn in _DETECTOR_RULES:
        if all(k in low for k in keys):
            return fn
    return None


def supported(move: str) -> bool:
    return match_move(move) is not None


# =============================================================================
# Sequence engine + winner snapshot
# =============================================================================

def pose_sequence_winner(video_path, moves, dt=0.1, t_start=0.0, t_end=None):
    """First person to complete ``moves`` in order, timed from pose geometry.

    Samples the clip every ``dt`` seconds, runs pose per frame, and matches each
    detection to a fixed left-to-right column taken from the earliest
    well-populated frame (STARTING position). Each column advances through the
    move list only when the next move's detector fires at a later sample.

    Returns ``{"winner": <1-based column | -1>, "timestamp": float|None,
    "num_people": int}`` (winner -1 / timestamp None == nobody finished).
    """
    if isinstance(moves, str):
        moves = [moves]
    detectors = [match_move(m) for m in moves]
    missing = [m for m, d in zip(moves, detectors) if d is None]
    if missing:
        raise ValueError(f"No YOLO pose detector for: {missing}")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = (n_frames / fps) if n_frames else None
    if t_end is None:
        t_end = duration if duration else 1e9

    samples = []                                    # [(t, [(cx, kp), ...]), ...]
    t = t_start
    while t <= t_end:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        samples.append((round(t, 3), pose_people(frame)))
        t += dt
    cap.release()

    ref = max((p for _, p in samples), key=len, default=[])
    if not ref:
        return {"winner": -1, "timestamp": None, "num_people": 0}
    columns = [cx for cx, _ in ref]                 # left-to-right starting x
    num = len(columns)

    progress = [0] * num
    last_time = [t_start] * num
    done_time: list[float | None] = [None] * num
    for ts, people in samples:
        for cx, kp in people:
            j = min(range(num), key=lambda k: abs(cx - columns[k]))
            if done_time[j] is not None:
                continue
            idx = progress[j]
            if idx < len(detectors) and ts >= last_time[j] and detectors[idx](kp):
                progress[j] = idx + 1
                last_time[j] = ts
                if progress[j] == len(detectors):
                    done_time[j] = ts

    finishers = [(dt_, j) for j, dt_ in enumerate(done_time) if dt_ is not None]
    if not finishers:
        return {"winner": -1, "timestamp": None, "num_people": num}
    ts, j = min(finishers)
    return {"winner": j + 1, "timestamp": ts, "num_people": num}


def winner_snapshot(video_path, timestamp, winner_number, num_people,
                    out_path="winner.jpg"):
    """Still frame at the winning moment with the winner boxed (YOLOX detection,
    Nth person from the left). Falls back to an equal-column split if the
    detector doesn't find enough people."""
    cap = cv2.VideoCapture(str(video_path))
    if timestamp is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {video_path}")

    h, w = frame.shape[:2]
    people = detect_people(frame)
    if len(people) >= winner_number >= 1:
        x0, y0, x1, y1 = people[winner_number - 1]
    else:
        col = w / max(num_people, 1)
        x0, y0, x1, y1 = (int((winner_number - 1) * col) + 6, 6,
                          int(winner_number * col) - 6, h - 6)

    x0, x1 = max(0, x0), min(w, x1)
    y0, y1 = max(0, y0), min(h, y1)
    color = (0, 215, 255)  # amber, BGR
    label = f"WINNER: Person {winner_number} @ {timestamp}s"
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 4)
    label_y = y0 - 12 if y0 - 12 > 24 else min(h - 8, y1 + 32)
    cv2.putText(frame, label, (x0 + 4, label_y), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(out_path), frame)
    return out_path


# =============================================================================
# Orchestration -- mirrors vss.find_winner's output shape
# =============================================================================

def find_winner_yolo(video_path, moves, dt=0.1) -> dict:
    """Find the first person to complete ``moves`` (a str or ordered list) using
    pose-based detection. Returns a dict with ``winner`` (1-based left-to-right
    column, -1 if nobody), ``timestamp``, ``num_people``, ``method``, ``timeline``.
    """
    target = list(moves) if isinstance(moves, (list, tuple)) else [moves]
    res = pose_sequence_winner(video_path, target, dt=dt)
    res["method"] = "yolo"
    res["timeline"] = []
    return res
