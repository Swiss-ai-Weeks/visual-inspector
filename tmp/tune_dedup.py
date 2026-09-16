#!/usr/bin/env python3
"""Sweep the duplicate-suppression settings to match the real headcount.

The CV tracker gives one person several ids when it loses and re-acquires them,
so the raw track count overshoots. This tries the knobs and prints how many
tracks each setting yields — pick the one matching the people in the clip.

Usage:
    python3 tmp/tune_dedup.py <video_name> <local_video.mp4> [expected_people]

e.g.
    python3 tmp/tune_dedup.py dance_vst /home/nvidia/Documents/test_repo/dance.mp4 7
"""

import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP))

import tracking                        # noqa: E402
from overlay import probe_video        # noqa: E402


def run(frame_size, *, iou, confidence, dominance):
    """fetch_tracks with the module constants overridden for this trial."""
    tracking.IOU_MERGE = iou
    tracking.MIN_CONFIDENCE = confidence
    tracking.MERGE_DOMINANCE = dominance
    try:
        return tracking.fetch_tracks(sys.argv[1], frame_size)
    except Exception as exc:
        print(f"    !! {exc}")
        return []


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)

    src = Path(sys.argv[2]).expanduser().resolve()
    expected = int(sys.argv[3]) if len(sys.argv) > 3 else None

    _, width, height = probe_video(src)
    size = (width, height)
    print(f"video {src.name}  {width}x{height}"
          + (f"   expecting {expected} people" if expected else ""))

    print(f"\n{'coasted':>9} {'iou':>6} {'domin':>6} {'tracks':>7}   "
          f"track lengths (seconds)")
    print("-" * 78)

    for confidence, coasted_label in ((-99.0, "kept"), (0.0, "dropped")):
        for iou in (0.0, 0.4, 0.55, 0.7, 0.85):
            tracks = run(size, iou=iou, confidence=confidence, dominance=0.5)
            spans = sorted(
                (t["points"][-1]["t"] - t["points"][0]["t"] for t in tracks),
                reverse=True,
            )
            flag = ""
            if expected and len(tracks) == expected:
                flag = "  <== matches"
            print(f"{coasted_label:>9} {iou:6.2f} {0.5:6.2f} {len(tracks):7}   "
                  + " ".join(f"{s:.1f}" for s in spans[:12]) + flag)

    print("\nPick the row matching the real headcount, then set it in app/run.sh:")
    print("    export TRACKING_IOU_MERGE=<iou>")
    print("    export TRACKING_MIN_CONFIDENCE=0.0     # drop coasted boxes")
    print("\nThen re-render:")
    print(f"    python3 tmp/render_real.py {sys.argv[1]} {src}")


if __name__ == "__main__":
    main()
