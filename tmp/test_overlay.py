#!/usr/bin/env python3
"""Render a tracking overlay from synthetic tracks — no VSS involved.

Proves the ffmpeg drawbox/drawtext path works before the real CV metadata
endpoint is known. Two fake people walk across the frame in opposite
directions; if the output shows two labelled boxes crossing smoothly, the
renderer is good and only the metadata source is left to wire up.

Usage:
    python3 tmp/test_overlay.py <video.mp4> [out.mp4]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

from overlay import probe_video, render_overlay  # noqa: E402

DURATION = 10.0   # seconds of synthetic track to generate
RATE = 10.0       # samples per second, like a typical CV pipeline


def synthetic_tracks() -> list[dict]:
    """Two people crossing, as normalised 0-1 bboxes over time."""
    steps = int(DURATION * RATE)
    left_to_right, right_to_left = [], []

    for i in range(steps):
        t = i / RATE
        progress = i / max(1, steps - 1)

        x = 0.05 + progress * 0.65
        left_to_right.append({"t": t, "bbox": (x, 0.25, x + 0.18, 0.92)})

        x = 0.77 - progress * 0.65
        right_to_left.append({"t": t, "bbox": (x, 0.30, x + 0.16, 0.88)})

    return [
        {"track_id": "synthetic-a", "points": left_to_right},
        {"track_id": "synthetic-b", "points": right_to_left},
    ]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    src = Path(sys.argv[1]).resolve()
    if not src.is_file():
        raise SystemExit(f"No such video: {src}")

    dst = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else \
        src.with_name(src.stem + "_overlay_test.mp4")

    fps, width, height = probe_video(src)
    print(f"source : {src.name}  {width}x{height} @ {fps:.2f}fps")

    render_overlay(src, synthetic_tracks(), dst)
    print(f"written: {dst}")
    print("\nExpect two labelled boxes (P1 blue, P2 amber) crossing the frame.")


if __name__ == "__main__":
    main()
