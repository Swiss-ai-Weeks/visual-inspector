#!/usr/bin/env python3
"""Can we get a position-per-timestep out of attribute_search? Read-only.

Established so far:
  - video_sources wants the video NAME, not the sensor id
  - rows carry object_id / object_type / bbox (pixel leftX,topY,rightX,bottomY)
  - but frame_timestamp is the "best frame", deduplicated per object, so one
    query returns one row per object, not a timeline

Two ways that might yield a timeline:
  A. loosen the knobs (min_similarity, fuse_multi_attribute, top_k)
  B. slide timestamp_start/timestamp_end across the clip, one query per window

Usage:
    python3 tmp/probe_timeline.py [video_name]
"""

import sys
from datetime import datetime, timedelta, timezone

import requests

AGENT = "http://127.0.0.1:8000"
VST = "http://127.0.0.1:30000"

# The app uploads with this as the stream start, so offsets from it are
# offsets into the video itself.
BASE = datetime(2025, 1, 1, tzinfo=timezone.utc)

DURATION = 20.0   # seconds of clip to sweep
WINDOW = 1.0      # seconds per window in the sliding sweep


def search(**body):
    body.setdefault("query", "person")
    try:
        response = requests.post(f"{AGENT}/api/v1/attribute_search",
                                 json=body, timeout=(15, 120))
        response.raise_for_status()
        return response.json().get("value") or []
    except Exception as exc:
        print(f"    !! {exc}")
        return []


def describe(label, rows):
    ids, stamps = set(), set()
    for row in rows:
        meta = row.get("metadata") or {}
        ids.add(meta.get("object_id"))
        stamps.add(meta.get("frame_timestamp"))
    print(f"  {label:52} rows={len(rows):4}  objects={len(ids):3}  frames={len(stamps):3}")
    return ids, stamps


def iso(offset):
    return (BASE + timedelta(seconds=offset)).isoformat().replace("+00:00", "Z")


def pick_name():
    sensors = requests.get(f"{VST}/vst/api/v1/sensor/list", timeout=10).json()
    for sensor in sensors:
        if "cringe" in (sensor.get("name") or "").lower():
            return sensor.get("name")
    return sensors[0].get("name")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else pick_name()
    print(f">>> video_name: {name}\n")

    print("A. knob sweep — can one query return a timeline?")
    base = {"video_sources": [name]}
    describe("top_k=500", search(**base, top_k=500))
    describe("top_k=500 min_similarity=0.0", search(**base, top_k=500, min_similarity=0.0))
    describe("top_k=500 fuse_multi_attribute=False",
             search(**base, top_k=500, fuse_multi_attribute=False))
    describe("top_k=500 min_sim=0.0 fuse=False",
             search(**base, top_k=500, min_similarity=0.0, fuse_multi_attribute=False))
    describe("query='man' top_k=500 min_sim=0.0",
             search(**base, query="man", top_k=500, min_similarity=0.0))

    print(f"\nB. sliding window, fusion OFF — {WINDOW}s windows across {DURATION}s")
    print("   (the previous run left fusion ON, which collapsed every window to 1 row)")
    all_ids, all_frames, total = set(), set(), 0
    for i in range(int(DURATION / WINDOW)):
        start, end = i * WINDOW, (i + 1) * WINDOW
        rows = search(**base, top_k=50, min_similarity=0.0,
                      fuse_multi_attribute=False,
                      timestamp_start=iso(start), timestamp_end=iso(end))
        ids, stamps = describe(f"  [{start:5.1f}s - {end:5.1f}s]", rows)
        # The decisive detail: does the best frame land INSIDE the window?
        for stamp in sorted(s for s in stamps if s):
            inside = "in " if iso(start) <= stamp <= iso(end) else "OUT"
            print(f"       {inside} {stamp}")
        all_ids |= ids
        all_frames |= stamps
        total += len(rows)

    print(f"\n  sweep total: rows={total}  distinct objects={len(all_ids)}  "
          f"distinct frames={len(all_frames)}")
    print(f"  object ids: {sorted(i for i in all_ids if i)}")

    if len(all_frames) > 5:
        print("\n  >> VERDICT: windowing produces a usable timeline.")
    else:
        print("\n  >> VERDICT: windowing does NOT produce enough distinct frames.")
        print("     attribute_search cannot give per-timestep positions here.")


if __name__ == "__main__":
    main()
