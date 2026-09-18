#!/usr/bin/env python3
"""Go/no-go: does the CV pipeline give us per-person tracking for a video?

This is the one question everything else depends on. It runs the real code
path — upload -> /complete -> fetch metadata — and prints a verdict.

It DOES upload to VSS and register a sensor, then deletes it again.

Usage:
    python3 tmp/check_tracking.py <video.mp4>
"""

import sys
import time
from pathlib import Path

APP = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP))

from overlay import ensure_mp4, probe_duration, probe_video  # noqa: E402
from process_video import delete_video, upload_video         # noqa: E402
from tracking import cv_progress, fetch_tracks, wait_for_cv  # noqa: E402


def rule(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        raise SystemExit(1)

    src = Path(sys.argv[1]).resolve()
    if not src.is_file():
        raise SystemExit(f"No such video: {src}")

    rule("1. re-encode for VST")
    mp4 = ensure_mp4(src)
    fps, width, height = probe_video(mp4)
    print(f"   {mp4.name}  {width}x{height} @ {fps:.2f}fps")

    sensor_id, video_name = None, mp4.stem
    try:
        rule("2. upload + /complete  (this is what runs RTVI-CV)")
        # The clip is registered under a name of its own: reusing the filename
        # inherits the stream id of every earlier run of this video, and with it
        # whatever that run orphaned in the VSS asset store.
        sensor_id, video_name, cv_ready = upload_video(str(mp4))
        print(f"   sensor_id : {sensor_id}")
        print(f"   video_name: {video_name}   <- what Elasticsearch files it under")
        print(f"   cv_ready  : {cv_ready}")
        if not cv_ready:
            print("\n   >> /complete FAILED — RTVI-CV did not process this clip.")
            print("      The warning above says why. Tracking cannot work until")
            print("      that is fixed; everything below will fail as a result.")

        rule("3. wait for RTVI-CV to finish")
        # Do NOT read (or delete the sensor) before this returns: RTVI-CV runs
        # slower than real time here, and deleting mid-run truncates the clip.
        duration = probe_duration(mp4)
        print(f"   clip duration : {duration:.2f}s" if duration else "   duration: ?")
        def show(count, latest):
            # RTVI-CV can take minutes; without this the script looks hung.
            reached = f"{latest:6.2f}s" if latest is not None else "     ?"
            print(f"   ... frames={count:<6} reached={reached}", flush=True)

        frames = wait_for_cv(video_name, duration, poll=3.0, on_poll=show)
        expected = int((duration or 0) * fps)
        print(f"   frames written: {frames}"
              f"{f'  (clip has ~{expected} frames)' if expected else ''}")

        _, latest = cv_progress(video_name)
        if duration and latest is not None:
            print(f"   CV reached    : {latest:.2f}s of {duration:.2f}s "
                  f"({latest / duration:.0%})")
            if latest < duration - 0.5:
                print("   >> RTVI-CV stopped short. It is still writing, it hit an")
                print("      error, or something deleted the stream underneath it.")

        rule("4. fetch tracking metadata")
        tracks = None
        try:
            tracks = fetch_tracks(video_name, (width, height))
        except Exception as exc:
            print(f"   {exc}")

        if tracks is None:
            rule("VERDICT: NO TRACKING")
            print(f"   Nothing in Elasticsearch for '{video_name}' after 60s.")
            print("   Check whether the frames landed under a different name:")
            print("     curl -s localhost:9200/mdx-raw-*/_search -H 'Content-Type: application/json' \\")
            print("       -d '{\"size\":0,\"aggs\":{\"s\":{\"terms\":"
                  "{\"field\":\"sensorId.keyword\",\"size\":50}}}}' | head -40")
            return

        rule("5. positions per timestep")
        for track in tracks:
            points = track["points"]
            span = points[-1]["t"] - points[0]["t"]
            print(f"\n   track {track['track_id']}  ({len(points)} samples, "
                  f"{points[0]['t']:.2f}..{points[-1]['t']:.2f}s = {span:.2f}s)")
            for point in points[:8]:
                x1, y1, x2, y2 = point["bbox"]
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                print(f"     t={point['t']:6.2f}s  centre=({cx:.3f}, {cy:.3f})  "
                      f"bbox=({x1:.3f}, {y1:.3f}, {x2:.3f}, {y2:.3f})")
            if len(points) > 8:
                print(f"     ... {len(points) - 8} more")

        covered = max(t["points"][-1]["t"] for t in tracks)
        ratio = covered / duration if duration else 0
        rule(f"VERDICT: {len(tracks)} person track(s), "
             f"{covered:.1f}s of {duration:.1f}s covered ({ratio:.0%})"
             if duration else f"VERDICT: {len(tracks)} person track(s)")
        if ratio < 0.8:
            print("   >> TRUNCATED. RTVI-CV stopped early — the sensor was torn")
            print("      down while it was writing, or it timed out. It is not")
            print("      the query: stage 3 shows how far CV actually got.")
        else:
            print("   Full-length tracking. Track count should match the")
            print("   number of people in the clip.")

    finally:
        # Deleting the sensor also wipes its frames from Elasticsearch, so the
        # tracking data is kept by default and removed only on request.
        if sensor_id and "--delete" in sys.argv:
            print(f"\n[cleanup] deleting sensor {sensor_id}: "
                  f"{'ok' if delete_video(sensor_id) else 'failed'}")
        elif sensor_id:
            print(f"\n[kept] sensor {sensor_id} as '{video_name}' — render it with:")
            print(f"   python3 tmp/render_real.py {video_name} {src} "
                  f"{src.with_suffix('')}.tracked.mp4")
        mp4.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
