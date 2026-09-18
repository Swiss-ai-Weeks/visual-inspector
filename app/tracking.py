"""Per-person tracking from the VSS CV pipeline.

RTVI-CV emits per-frame detections onto Kafka, which land in Elasticsearch as
`mdx-raw-*` documents. Each document is one frame:

    {"sensorId": "<video name>",
     "id": "50",                               # frame number
     "timestamp": "2025-01-01T00:00:02.080Z",  # VSS_UPLOAD_TS + offset
     "objects": [{"id": "1", "type": "Person",
                  "bbox": {"leftX":.., "topY":.., "rightX":.., "bottomY":..}}]}

`objects[].id` is the tracker id, so grouping frames by it gives one position
timeline per person.

We read the store directly rather than going through the agent's
`attribute_search`: that route is a similarity search which deduplicates to a
single best frame per object, which is precisely the timeline we need. The
trade-off is a dependency on VSS's internal index schema.

Two gotchas worth keeping in mind:
  - `sensorId` holds the video NAME (the uploaded filename without extension),
    not the VST sensor UUID.
  - `sensorId` is mapped as text; term queries and sorting need
    `sensorId.keyword`.
"""

import logging
import os
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

ES_URL = os.environ.get("VSS_ES_URL", "http://127.0.0.1:9200").rstrip("/")
# Wildcard: the index is date-suffixed and rolls over.
MDX_INDEX = os.environ.get("MDX_INDEX", "mdx-raw-*")

# The timestamp the app stamps on upload; frame timestamps are offsets from it.
UPLOAD_TIMESTAMP = os.environ.get("VSS_UPLOAD_TS", "2025-01-01T00:00:00")

# Elasticsearch refuses a plain search beyond max_result_window (10k default).
MAX_FRAMES = int(os.environ.get("TRACKING_MAX_FRAMES", "10000"))

# How long to wait for RTVI-CV, which runs slower than real time here: an
# overall budget, plus how long a flat frame count is tolerated before a clip
# that is not yet covered is declared stalled.
CV_TIMEOUT = float(os.environ.get("TRACKING_CV_TIMEOUT", "900"))
CV_STALL_SECONDS = float(os.environ.get("TRACKING_CV_STALL", "90"))
# Fraction of the clip that has to be covered for a flat count to mean "done".
CV_MIN_COVERAGE = float(os.environ.get("TRACKING_CV_MIN_COVERAGE", "0.9"))

# Object embeddings dominate the index (611MB); never pull them.
_SOURCE_FIELDS = (
    "timestamp,id,sensorId,"
    "objects.id,objects.type,objects.bbox,objects.confidence"
)

_PERSON_LABELS = {"person", "people", "human", "pedestrian"}

# Duplicate-track suppression. The CV tracker re-issues an id when it loses and
# re-acquires someone, so one person can wear two or three boxes at once.
#   IOU_MERGE        overlap above which two boxes are the same person
#                    (0 disables suppression entirely)
#   MERGE_DOMINANCE  fraction of its life an id must spend losing to another
#                    before it is folded in — stops two people who genuinely
#                    stand close from being merged
#   MIN_CONFIDENCE   drop tracker-coasted boxes, where the detector did not
#                    fire and the position is predicted (MDX writes these with
#                    a negative confidence)
IOU_MERGE = float(os.environ.get("TRACKING_IOU_MERGE", "0.55"))
MERGE_DOMINANCE = float(os.environ.get("TRACKING_MERGE_DOMINANCE", "0.5"))
MIN_CONFIDENCE = float(os.environ.get("TRACKING_MIN_CONFIDENCE", "0.0"))

# Flicker rejection: a "person" seen this briefly is noise.
MIN_TRACK_POINTS = int(os.environ.get("TRACKING_MIN_POINTS", "5"))
MIN_TRACK_SECONDS = float(os.environ.get("TRACKING_MIN_SECONDS", "0.3"))


class TrackingUnavailable(RuntimeError):
    """The CV metadata could not be fetched or contained no usable tracks."""


def _base_epoch() -> float:
    """Upload timestamp as a UTC epoch.

    Parsed as UTC explicitly: VSS_UPLOAD_TS carries no zone, while frame
    timestamps end in Z, and letting Python apply the local zone would skew
    every offset by the UTC offset of this machine.
    """
    stamp = datetime.fromisoformat(UPLOAD_TIMESTAMP)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp()


def _offset_seconds(timestamp: str, base: float) -> float | None:
    try:
        moment = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp() - base


def _as_bbox(raw) -> tuple[float, float, float, float] | None:
    """MDX corners, with the common alternative spellings tolerated."""
    if not isinstance(raw, dict):
        return None
    for a, b, c, d in (
        ("leftX", "topY", "rightX", "bottomY"),
        ("lx", "ly", "rx", "ry"),
        ("x1", "y1", "x2", "y2"),
    ):
        if all(k in raw for k in (a, b, c, d)):
            try:
                return (float(raw[a]), float(raw[b]), float(raw[c]), float(raw[d]))
            except (TypeError, ValueError):
                return None
    return None


def _fetch_frames(video_name: str) -> list[dict]:
    """Every CV frame recorded for this video, oldest first."""
    body = {
        "query": {"term": {"sensorId.keyword": video_name}},
        "sort": [{"timestamp": {"order": "asc"}}],
    }
    try:
        response = requests.post(
            f"{ES_URL}/{MDX_INDEX}/_search",
            params={"size": MAX_FRAMES, "_source_includes": _SOURCE_FIELDS},
            json=body,
            timeout=(15, 120),
        )
    except requests.RequestException as exc:
        raise TrackingUnavailable(f"Elasticsearch unreachable at {ES_URL}: {exc}") from exc

    if not response.ok:
        raise TrackingUnavailable(
            f"Elasticsearch query failed (HTTP {response.status_code}): "
            f"{response.text[:300]}"
        )

    hits = response.json().get("hits", {})
    total = (hits.get("total") or {}).get("value", 0)
    documents = [hit["_source"] for hit in hits.get("hits", []) if "_source" in hit]

    if total > len(documents):
        logger.warning(
            "Only %d of %d CV frames read for %s; raise TRACKING_MAX_FRAMES.",
            len(documents), total, video_name,
        )
    return documents


def cv_progress(video_name: str) -> tuple[int, float | None]:
    """(frames written so far, latest frame offset in seconds) for this video."""
    body = {
        "query": {"term": {"sensorId.keyword": video_name}},
        "aggs": {"latest": {"max": {"field": "timestamp"}}},
    }
    try:
        response = requests.post(
            f"{ES_URL}/{MDX_INDEX}/_search",
            params={"size": 0}, json=body, timeout=(10, 60),
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return 0, None

    count = ((payload.get("hits") or {}).get("total") or {}).get("value", 0)
    latest = ((payload.get("aggregations") or {}).get("latest") or {}).get("value")
    # A date aggregation returns epoch milliseconds.
    offset = (latest / 1000.0) - _base_epoch() if latest else None
    return count, offset


def wait_for_cv(
    video_name: str,
    duration: float | None = None,
    timeout: float = CV_TIMEOUT,
    poll: float = 3.0,
    stable_polls: int = 3,
    min_coverage: float = CV_MIN_COVERAGE,
    on_poll=None,
) -> int:
    """Block until RTVI-CV has finished writing frames for this video.

    This matters more than it looks. RTVI-CV processes an uploaded file
    asynchronously and, on this hardware, slower than real time. Reading as
    soon as the first frames appear yields a truncated timeline, and deleting
    the sensor at that point kills the stream outright — which is how an
    11-second clip ends up with 0.33 seconds of tracking.

    A flat frame count is only trusted once the clip is actually covered. Below
    `min_coverage` a plateau is treated as RTVI-CV being slow rather than
    finished, and the wait continues for `CV_STALL_SECONDS` before giving up —
    three quiet polls nine seconds apart used to be enough to report 0.2 s of a
    15 s clip as "settled".

    `on_poll(count, latest)` is called after every poll, for callers that would
    otherwise sit silent for minutes.

    Returns the final frame count. Finishing early is not an error: the caller
    works with whatever landed, and a short result is logged as a warning.
    """
    import time

    deadline = time.time() + timeout
    patient_polls = max(stable_polls, int(CV_STALL_SECONDS / max(poll, 0.1)))
    last_count, stable = -1, 0

    while time.time() < deadline:
        count, latest = cv_progress(video_name)
        if on_poll:
            on_poll(count, latest)

        # Covered the whole clip? Done, no need to wait for the count to settle.
        if duration and latest is not None and latest >= duration - 0.5:
            logger.info("CV complete for %s: %d frames, %.2fs", video_name, count, latest)
            return count

        if count and count == last_count:
            stable += 1
            coverage = (latest / duration) if (duration and latest) else 0.0
            # A plateau short of the clip length is a stall, not an ending.
            limit = stable_polls if coverage >= min_coverage else patient_polls
            if stable >= limit:
                report = logger.info if coverage >= min_coverage else logger.warning
                report(
                    "CV stopped for %s: %d frames, latest %.2fs of %.2fs (%.0f%%)",
                    video_name, count, latest or 0.0, duration or 0.0, coverage * 100,
                )
                return count
        else:
            stable, last_count = 0, count

        time.sleep(poll)

    logger.warning(
        "CV still running for %s after %.0fs; using %d frames so far.",
        video_name, timeout, max(last_count, 0),
    )
    return max(last_count, 0)


def _iou(a: tuple, b: tuple) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if overlap <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


def _resolve(parent: dict, node: str) -> str:
    while parent[node] != node:
        parent[node] = parent[parent[node]]
        node = parent[node]
    return node


def _merge_duplicate_ids(detections: list[tuple]) -> dict[str, str]:
    """Map duplicate tracker ids onto the one track that really owns a person.

    The CV tracker hands the same person a fresh id after it loses and
    re-acquires them, while the stale track coasts along on top — so one person
    ends up wearing two or three boxes, more of them the longer they are on
    screen.

    Each frame is treated as a little non-maximum suppression: the
    longest-lived, highest-confidence box wins, and any box overlapping it by
    more than IOU_MERGE is recorded as a duplicate of it. An id that loses this
    way for most of its life is folded into its winner; an id that only
    occasionally overlaps (two people genuinely standing close) is left alone.

    `detections` is a list of (t, track_id, bbox, confidence).
    """
    from collections import Counter, defaultdict

    lifetime: Counter = Counter(track_id for _, track_id, _, _ in detections)
    by_frame: dict[float, list[tuple]] = defaultdict(list)
    for detection in detections:
        by_frame[round(detection[0], 3)].append(detection)

    votes: dict[str, Counter] = defaultdict(Counter)
    appearances: Counter = Counter()

    for frame_detections in by_frame.values():
        # Prefer the established track, then the confident one.
        frame_detections.sort(key=lambda d: (-lifetime[d[1]], -d[3]))
        kept: list[tuple] = []
        for detection in frame_detections:
            appearances[detection[1]] += 1
            winner = next(
                (k for k in kept if _iou(k[2], detection[2]) >= IOU_MERGE), None
            )
            if winner is not None:
                votes[detection[1]][winner[1]] += 1
            else:
                kept.append(detection)

    parent = {track_id: track_id for track_id in lifetime}
    for track_id, counter in votes.items():
        winner, count = counter.most_common(1)[0]
        seen = appearances[track_id]
        if seen and count / seen >= MERGE_DOMINANCE:
            a, b = _resolve(parent, track_id), _resolve(parent, winner)
            if a != b:
                # Keep the longer-lived id as the survivor.
                if lifetime[a] >= lifetime[b]:
                    parent[b] = a
                else:
                    parent[a] = b

    return {track_id: _resolve(parent, track_id) for track_id in parent}


def fetch_tracks(
    video_name: str,
    frame_size: tuple[int, int] | None = None,
) -> list[dict]:
    """Return one track per tracked person, with normalised (0-1) bboxes.

    Each track is `{"track_id": str, "points": [{"t": float, "bbox": (...)}]}`,
    `t` in seconds from the start of the video, points sorted by time.

    `video_name` is the uploaded filename without its extension — what VSS
    stores as `sensorId`. `frame_size` is (width, height) of the source video,
    needed because MDX reports bboxes in pixels.
    """
    frames = _fetch_frames(video_name)
    if not frames:
        raise TrackingUnavailable(
            f"No CV frames in {MDX_INDEX} for '{video_name}'. The clip was "
            "either never processed by RTVI-CV (check that /complete "
            "succeeded) or is registered under a different name."
        )

    base = _base_epoch()
    detections: list[tuple] = []
    coasted = 0

    for frame in frames:
        offset = _offset_seconds(frame.get("timestamp", ""), base)
        if offset is None:
            continue

        for obj in frame.get("objects") or []:
            if str(obj.get("type", "")).strip().lower() not in _PERSON_LABELS:
                continue
            bbox = _as_bbox(obj.get("bbox"))
            if bbox is None:
                continue

            # A non-positive confidence means the detector did not fire on this
            # frame and the tracker predicted the box. Those coasted boxes are
            # what drift off a person and show up as a second box on someone
            # else, so they are dropped by default.
            confidence = float(obj.get("confidence") or 0.0)
            if confidence <= MIN_CONFIDENCE:
                coasted += 1
                continue

            detections.append((offset, str(obj.get("id")), bbox, confidence))

    if not detections:
        kinds = {str(o.get("type")) for f in frames for o in (f.get("objects") or [])}
        hint = (
            f" Every one was dropped by the confidence filter — rerun with "
            f"TRACKING_MIN_CONFIDENCE=-99 to keep them."
            if coasted else
            f" Detected types: {sorted(kinds) or 'none'}."
        )
        raise TrackingUnavailable(
            f"{len(frames)} CV frames found for '{video_name}' but no usable "
            f"Person detections ({coasted} dropped as low-confidence).{hint}"
        )

    raw_ids = len({track_id for _, track_id, _, _ in detections})

    # Fold duplicate ids onto the track that owns the person.
    if IOU_MERGE > 0:
        canonical = _merge_duplicate_ids(detections)
    else:
        canonical = {track_id: track_id for _, track_id, _, _ in detections}

    # One box per person per frame: the most confident survivor.
    best: dict[tuple[str, float], tuple] = {}
    for offset, track_id, bbox, confidence in detections:
        key = (canonical[track_id], round(offset, 3))
        if key not in best or confidence > best[key][1]:
            best[key] = (bbox, confidence)

    grouped: dict[str, list[dict]] = {}
    for (track_id, offset), (bbox, _) in best.items():
        grouped.setdefault(track_id, []).append({"t": offset, "bbox": bbox})

    # Drop flickers: a track seen for a fraction of a second is noise, not a
    # person, and would otherwise add a box that blinks on and vanishes.
    grouped = {
        track_id: points for track_id, points in grouped.items()
        if len(points) >= MIN_TRACK_POINTS
        and (max(p["t"] for p in points) - min(p["t"] for p in points)) >= MIN_TRACK_SECONDS
    } or grouped  # never filter everything away

    people = sum(len(points) for points in grouped.values())
    logger.info(
        "tracking dedup: %d raw id(s) -> %d track(s); %d coasted box(es) dropped",
        raw_ids, len(grouped), coasted,
    )

    # More documents than distinct instants means the clip was ingested more
    # than once under this name, and every person is duplicated at every
    # timestamp. No amount of IoU tuning fixes that — the stale sensor has to go.
    distinct_instants = len({round(d[0], 3) for d in detections})
    if distinct_instants and len(frames) > distinct_instants * 1.5:
        logger.warning(
            "%d CV documents but only %d distinct timestamps for '%s' — the "
            "clip looks ingested more than once. Delete the sensor and re-upload.",
            len(frames), distinct_instants, video_name,
        )

    # MDX reports pixels; the overlay wants resolution-independent boxes.
    if frame_size:
        width, height = frame_size
    else:
        raise TrackingUnavailable(
            "MDX bboxes are in pixels but the video frame size is unknown."
        )

    tracks = [
        {
            "track_id": track_id,
            "points": sorted(
                (
                    {
                        "t": point["t"],
                        "bbox": (
                            point["bbox"][0] / width, point["bbox"][1] / height,
                            point["bbox"][2] / width, point["bbox"][3] / height,
                        ),
                    }
                    for point in points
                ),
                key=lambda p: p["t"],
            ),
        }
        for track_id, points in grouped.items()
    ]
    tracks.sort(key=lambda track: track["points"][0]["t"])

    logger.info(
        "tracking: %d frame(s), %d person detection(s), %d track(s) for %s",
        len(frames), people, len(tracks), video_name,
    )
    return tracks


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python3 tracking.py <video_name> [width height]")
        raise SystemExit(1)

    size = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else None
    for track in fetch_tracks(sys.argv[1], size):
        points = track["points"]
        print(
            f"track {track['track_id']}: {len(points)} points, "
            f"t={points[0]['t']:.2f}..{points[-1]['t']:.2f}"
        )
