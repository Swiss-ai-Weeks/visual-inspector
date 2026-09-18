#!/usr/bin/env python3
"""Find what RTVI-CV is holding when /complete fails with 'Duplicate Camera id'.

That error comes from RTVI-CV, not from VST or the agent, and it survives a
fresh upload name and a fresh stream id — so the duplicate is a camera left
behind by a run that died before anything removed it. Nothing in this repo can
reach that registry, so first we have to find it.

Read-only by default: GETs against the agent, VST, and every port listening on
this box, plus a couple of probe paths each. Nothing is uploaded or deleted.

    python3 tmp/probe_cv_service.py                 # what is registered, where
    python3 tmp/probe_cv_service.py --delete <id>…  # drop these sensors (agent)

`--delete` takes explicit sensor ids only. It never deletes by pattern: this box
is shared, and most of what VST lists belongs to other people's uploads.
Deleting a sensor also drops its CV frames from Elasticsearch.
"""

import subprocess
import sys

import requests

AGENT = "http://127.0.0.1:8000"
VST = "http://127.0.0.1:30000"

# Ports we already know, so the sweep can say what it is looking at.
KNOWN = {
    8000: "VSS agent", 30000: "VST", 9200: "Elasticsearch", 9092: "Kafka",
    5601: "Kibana", 6379: "Redis", 5901: "VNC", 8017: "RT-VLM", 8018: "RT-VLM",
}

# Routes a DeepStream/RTVI-style stream manager tends to expose.
PROBE_PATHS = (
    "/api/v1/stream", "/api/v1/streams", "/api/v1/stream/list",
    "/v1/stream", "/stream/list", "/api/v1/sensor/list",
    "/openapi.json", "/docs", "/health", "/",
)

CV_HINTS = ("camera", "stream", "rtvi", "deepstream", "sensor", "pipeline")


def rule(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def listening_ports():
    out = subprocess.run(
        ["bash", "-lc", "ss -ltn | awk 'NR>1{print $4}'"],
        capture_output=True, text=True,
    ).stdout
    ports = set()
    for line in out.split():
        _, _, port = line.rpartition(":")
        if port.isdigit():
            ports.add(int(port))
    return sorted(ports)


def show_sensors():
    rule("what VST has registered")
    try:
        sensors = requests.get(f"{VST}/vst/api/v1/sensor/list", timeout=15).json()
    except Exception as exc:
        print(f"  !! {type(exc).__name__}: {exc}")
        return
    print(f"  {len(sensors)} sensor(s)\n")
    for s in sorted(sensors, key=lambda s: s.get("name") or ""):
        print(f"  {s.get('sensorId','?'):<40} {s.get('name','?')}")
    print("\n  Every one of these may still own a camera in RTVI-CV. The ones")
    print("  from your own aborted runs are safe to delete; the rest are not")
    print("  yours. Delete explicitly:")
    print("    python3 tmp/probe_cv_service.py --delete <sensorId> [<sensorId>…]")


def show_agent_routes():
    rule("agent routes that mention cv / stream / video")
    try:
        paths = requests.get(f"{AGENT}/openapi.json", timeout=15).json()["paths"]
    except Exception as exc:
        print(f"  !! {type(exc).__name__}: {exc}")
        return
    for path in sorted(paths):
        if any(k in path.lower() for k in ("cv", "stream", "video", "file", "asset")):
            print(f"  {path:<45} {sorted(paths[path])}")


def sweep():
    rule("hunting for the RTVI-CV stream manager")
    print("  A port that answers one of the stream routes below is the registry")
    print("  holding the duplicate camera.\n")
    for port in listening_ports():
        if port in (8000, 30000, 9200, 5601, 9092):
            continue
        label = KNOWN.get(port, "")
        for path in PROBE_PATHS:
            url = f"http://127.0.0.1:{port}{path}"
            try:
                r = requests.get(url, timeout=1.5)
            except requests.RequestException:
                break                      # nothing speaks HTTP here
            if r.status_code >= 400:
                continue
            body = (r.text or "")[:160].replace("\n", " ")
            hit = any(h in body.lower() for h in CV_HINTS)
            print(f"  {port:<6}{label:<12} {path:<22} {r.status_code} "
                  f"{'<-- look here  ' if hit else ''}{body[:80]}")


def delete(ids):
    rule("deleting sensors through the agent")
    for sensor_id in ids:
        try:
            r = requests.delete(f"{AGENT}/api/v1/videos/{sensor_id}", timeout=(10, 120))
            print(f"  DELETE {sensor_id} -> {r.status_code} {r.text[:200]}")
        except requests.RequestException as exc:
            print(f"  DELETE {sensor_id} -> FAILED {type(exc).__name__}: {exc}")
    print("\n  Re-run without --delete to confirm they are gone, then retry the")
    print("  upload. If 'Duplicate Camera id' survives an empty sensor list,")
    print("  the camera is orphaned inside RTVI-CV and only restarting that")
    print("  service will clear it.")


def main():
    if "--delete" in sys.argv:
        ids = sys.argv[sys.argv.index("--delete") + 1:]
        if not ids:
            print(__doc__)
            raise SystemExit(1)
        delete(ids)
        return

    show_sensors()
    show_agent_routes()
    sweep()


if __name__ == "__main__":
    main()
