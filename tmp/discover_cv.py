#!/usr/bin/env python3
"""Step 0: find the endpoint that serves RTVI-CV per-frame tracking metadata.

Read-only — GETs plus the agent's own search POST. Nothing is uploaded or
mutated. Throwaway diagnostic; this whole tmp/ folder can be deleted.

Usage:
    python3 tmp/discover_cv.py              # sensors + route schemas
    python3 tmp/discover_cv.py <sensor_id>  # also pull metadata for one sensor
"""

import json
import sys

import requests

AGENT = "http://127.0.0.1:8000"
VST_CANDIDATES = ["http://127.0.0.1:30000", "http://127.0.0.1:30888",
                  "http://172.16.0.189:7777"]


def show(label, fn):
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}")
    try:
        response = fn()
    except Exception as exc:
        print(f"  !! {type(exc).__name__}: {exc}")
        return None
    print(f"  HTTP {response.status_code}")
    body = response.text
    print("  " + (body[:1500].replace("\n", "\n  ") or "<empty>"))
    try:
        return response.json()
    except ValueError:
        return None


def preflight():
    """Check the only two things the overlay renderer needs."""
    import shutil
    import subprocess
    from pathlib import Path

    print(f"\n{'=' * 70}\npreflight: ffmpeg + font\n{'=' * 70}")
    for tool in ("ffmpeg", "ffprobe"):
        path = shutil.which(tool)
        print(f"  {tool:8} {path or '!! NOT FOUND — overlay cannot render'}")

    if shutil.which("ffmpeg"):
        out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                             capture_output=True, text=True).stdout
        for name in ("drawbox", "drawtext"):
            ok = any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())
            print(f"  filter {name:9} {'present' if ok else '!! MISSING'}")

    fonts = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    found = [f for f in fonts if Path(f).is_file()]
    print(f"  font     {found[0] if found else 'none of the candidates — boxes will be unlabelled'}")


def main():
    sensor_id = sys.argv[1] if len(sys.argv) > 1 else None

    preflight()

    spec = show("agent /openapi.json",
                lambda: requests.get(f"{AGENT}/openapi.json", timeout=15))
    if spec:
        paths = spec.get("paths", {})
        print("\n  -- routes mentioning search/metadata/cv/video --")
        for path in sorted(paths):
            if any(k in path for k in ("search", "metadata", "cv", "video")):
                print(f"     {path}  {list(paths[path])}")

        print("\n  -- request schema for attribute_search/full --")
        node = paths.get("/api/v1/attribute_search/full", {})
        print("  " + json.dumps(node, indent=2)[:2000].replace("\n", "\n  "))

        print("\n  -- component schema names --")
        print("  " + ", ".join(sorted(spec.get("components", {}).get("schemas", {}))))

    show("agent GET /api/v1/videos  (registered sensors)",
         lambda: requests.get(f"{AGENT}/api/v1/videos", timeout=15))

    for base in VST_CANDIDATES:
        for path in ("/vst/api/v1/sensor/list", "/api/v1/sensor/list"):
            show(f"VST {base}{path}",
                 lambda b=base, p=path: requests.get(f"{b}{p}", timeout=10))

    if not sensor_id:
        print("\n\nRe-run with a sensor_id to probe the metadata routes:"
              "\n    python3 tmp/discover_cv.py <sensor_id>")
        return

    show(f"agent POST /api/v1/attribute_search/full  (sensor {sensor_id})",
         lambda: requests.post(
             f"{AGENT}/api/v1/attribute_search/full",
             json={"sensor_id": sensor_id,
                   "input_message": "every person detection with bounding box and track id"},
             timeout=(15, 300)))

    for base in VST_CANDIDATES:
        for path in (f"/vst/api/v1/metadata/{sensor_id}",
                     f"/vst/api/v1/sensor/{sensor_id}/metadata",
                     f"/api/v1/metadata/{sensor_id}"):
            show(f"VST {base}{path}",
                 lambda b=base, p=path: requests.get(f"{b}{p}", timeout=30))


if __name__ == "__main__":
    main()
