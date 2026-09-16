"""Render CV tracking results back onto the uploaded video.

Draws one coloured box per tracked person, labelled with a stable player
number, so the per-timestep positions coming out of RTVI-CV can be checked by
eye.

Rendering goes through ffmpeg's drawbox/drawtext filters rather than a Python
imaging library: ffmpeg is already a hard dependency (see `ensure_mp4`), so the
overlay adds nothing to install. The cost is that the filter graph grows with
the number of metadata samples, which is why samples are decimated and merged
before the script is built.
"""

import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# A box stays on screen this long after its last metadata sample. CV metadata
# is usually sparser than the video framerate; without the hold the overlay
# flickers badly.
HOLD_SECONDS = 0.5

# Filter-graph budget. ffmpeg parses a few thousand chained filters happily;
# tens of thousands gets slow, so samples are decimated until we fit.
MAX_FILTER_ENTRIES = 4000
INITIAL_STEP = 0.1   # seconds between kept samples (10 Hz)
INITIAL_TOL = 0.015  # merge boxes that move less than this (fraction of frame)

# RGB hex, picked to stay distinguishable against skin tones and indoor walls.
PALETTE = [
    "0x00A0FF",  # blue
    "0xFFC400",  # amber
    "0x35D07F",  # green
    "0xE45CFF",  # magenta
    "0xFF4D4D",  # red
    "0x00E5E5",  # cyan
]

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
)


def ensure_mp4(src: Path) -> Path:
    """Re-encode to H.264/AAC MP4 for VST compatibility and browser playback."""
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
    logger.info("ffmpeg OK: %s -> %s", src.name, dst.name)
    return dst


def probe_duration(src: Path) -> float | None:
    """Clip length in seconds, or None when the container doesn't say."""
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "csv=p=0", str(src),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        return float(probe.stdout.strip())
    except ValueError:
        return None


def probe_video(src: Path) -> tuple[float, int, int]:
    """Return (fps, width, height) of a video file, via ffprobe."""
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate",
            "-of", "json", str(src),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    streams = json.loads(probe.stdout).get("streams") or []
    if not streams:
        raise RuntimeError(f"No video stream found in {src}")
    stream = streams[0]

    fps = 25.0
    rate = stream.get("r_frame_rate", "")
    if "/" in rate:
        num, _, den = rate.partition("/")
        try:
            if float(den):
                fps = float(num) / float(den)
        except ValueError:
            pass

    return fps, int(stream["width"]), int(stream["height"])


# --------------------------------------------------------------------------
# Turning tracks into time intervals
# --------------------------------------------------------------------------

def _decimate(points: list[dict], step: float) -> list[dict]:
    """Keep at most one sample per `step` window."""
    kept: list[dict] = []
    next_allowed = float("-inf")
    for point in points:
        if point["t"] >= next_allowed:
            kept.append(point)
            next_allowed = point["t"] + step
    return kept


def _is_close(a, b, tol: float) -> bool:
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def _intervals(points: list[dict], tol: float) -> list[tuple[float, float, tuple]]:
    """Collapse samples into (start, end, bbox) spans, merging static boxes."""
    spans: list[tuple[float, float, tuple]] = []

    for index, point in enumerate(points):
        start = point["t"]
        following = points[index + 1]["t"] if index + 1 < len(points) else None
        end = min(following, start + HOLD_SECONDS) if following else start + HOLD_SECONDS
        if end - start < 1e-3:
            continue

        if spans:
            prev_start, prev_end, prev_bbox = spans[-1]
            # Contiguous and barely moved? Extend rather than emit a new pair.
            if abs(prev_end - start) < 1e-3 and _is_close(prev_bbox, point["bbox"], tol):
                spans[-1] = (prev_start, end, prev_bbox)
                continue

        spans.append((start, end, point["bbox"]))

    return spans


def _fit_budget(tracks: list[dict]) -> list[list[tuple[float, float, tuple]]]:
    """Decimate and merge until the filter graph fits MAX_FILTER_ENTRIES."""
    step, tol = INITIAL_STEP, INITIAL_TOL

    while True:
        per_track = [_intervals(_decimate(t["points"], step), tol) for t in tracks]
        total = sum(len(spans) for spans in per_track)
        if total <= MAX_FILTER_ENTRIES or step >= 1.0:
            logger.info(
                "overlay filter graph: %d spans (step=%.2fs, tol=%.3f)", total, step, tol
            )
            return per_track
        step *= 2
        tol *= 1.5


# --------------------------------------------------------------------------
# Filter-script generation
# --------------------------------------------------------------------------

def _find_font() -> str | None:
    for candidate in _FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    logger.warning("No usable font found; overlay boxes will be unlabelled.")
    return None


def _build_filters(
    per_track: list[list[tuple[float, float, tuple]]],
    width: int,
    height: int,
) -> list[str]:
    """One drawbox (plus drawtext label) per span."""
    font = _find_font()
    thickness = max(2, round(width / 480))
    fontsize = max(14, round(width / 40))

    filters: list[str] = []
    for track_index, spans in enumerate(per_track):
        colour = PALETTE[track_index % len(PALETTE)]
        # Tracker ids can be long and opaque; players are numbered by the order
        # they first appear, which is also what the game prompt asks for.
        label = f"P{track_index + 1}"

        for start, end, bbox in spans:
            x1 = max(0, min(width - 1, int(bbox[0] * width)))
            y1 = max(0, min(height - 1, int(bbox[1] * height)))
            x2 = max(0, min(width, int(bbox[2] * width)))
            y2 = max(0, min(height, int(bbox[3] * height)))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue

            window = f"enable='between(t,{start:.3f},{end:.3f})'"
            filters.append(
                f"drawbox=x={x1}:y={y1}:w={x2 - x1}:h={y2 - y1}"
                f":color={colour}@1:t={thickness}:{window}"
            )

            if font:
                # Sit the label above the box, or inside it when at the top edge.
                label_y = y1 - fontsize - 8
                if label_y < 0:
                    label_y = y1 + 4
                filters.append(
                    f"drawtext=fontfile='{font}':text='{label}'"
                    f":x={x1 + 2}:y={label_y}:fontsize={fontsize}"
                    f":fontcolor=white:box=1:boxcolor={colour}@1:boxborderw=4"
                    f":{window}"
                )

    return filters


def render_overlay(src: Path, tracks: list[dict], dst: Path) -> Path:
    """Draw `tracks` over `src` and write a browser-playable MP4 to `dst`.

    `tracks` is the structure returned by `tracking.fetch_tracks` — bboxes
    normalised to 0-1, timestamps in seconds from the start of the clip.
    """
    _, width, height = probe_video(src)

    filters = _build_filters(_fit_budget(tracks), width, height)
    if not filters:
        raise RuntimeError("Tracks contained no drawable boxes.")

    # The graph is far too long for a command-line argument.
    script_path = dst.with_name(dst.stem + "_filters.txt")
    script_path.write_text(",\n".join(filters), encoding="utf-8")

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(src),
                "-filter_script:v", str(script_path),
                "-map", "0:v:0", "-map", "0:a?",
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
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", "replace")[-1500:]
        raise RuntimeError(f"ffmpeg overlay render failed:\n{stderr}") from exc
    finally:
        script_path.unlink(missing_ok=True)

    logger.info("overlay written: %s (%d filters)", dst.name, len(filters))
    return dst
