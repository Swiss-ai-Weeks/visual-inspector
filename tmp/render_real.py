#!/usr/bin/env python3
"""End-to-end overlay from REAL CV tracks — everything except the upload.

Pulls the per-frame tracks an already-ingested video has in Elasticsearch and
draws them over your local copy of that video.

    python3 tmp/render_real.py <video_name> <local_video.mp4> [out.mp4]

e.g.
    python3 tmp/render_real.py \\
        8934ec6e14e340d9bc1f6150aff087c6_cringe_video \\
        ~/Documents/cringe_video.mp4

The local file must be the same clip that was ingested — the boxes are
normalised against its width/height, so a different crop or resolution will
misalign them.
"""

import sys
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP))

from overlay import render_overlay, probe_video  # noqa: E402
from tracking import fetch_tracks                # noqa: E402


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        raise SystemExit(1)

    video_name = sys.argv[1]
    src = Path(sys.argv[2]).expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f"No such video: {src}")

    dst = (Path(sys.argv[3]).expanduser().resolve() if len(sys.argv) > 3
           else src.with_name(src.stem + "_tracked.mp4"))

    fps, width, height = probe_video(src)
    print(f"local video : {src.name}  {width}x{height} @ {fps:.2f}fps")

    tracks = fetch_tracks(video_name, (width, height))
    print(f"tracks      : {len(tracks)}")
    for track in tracks:
        points = track["points"]
        first, last = points[0], points[-1]
        fx = (first["bbox"][0] + first["bbox"][2]) / 2
        lx = (last["bbox"][0] + last["bbox"][2]) / 2
        print(f"  id {track['track_id']:>3}  {len(points):4} points  "
              f"t={first['t']:5.2f}..{last['t']:5.2f}s  "
              f"centre x {fx:.2f} -> {lx:.2f}")

    render_overlay(src, tracks, dst)
    print(f"\nwritten     : {dst}")
    print("Boxes should sit on the people and keep their id for the whole clip.")


if __name__ == "__main__":
    main()
