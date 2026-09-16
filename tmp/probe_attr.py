#!/usr/bin/env python3
"""Nail down how to call attribute_search. Read-only.

AttributeSearchMetadata is the tracking record we want:
    object_id / object_type / frame_timestamp / bbox

Open questions this answers:
  - what values does `source_type` take?
  - does `video_sources` want a sensor_id or a video name?
  - does /attribute_search or /attribute_search/full give cleaner rows?

Usage:
    python3 tmp/probe_attr.py              # picks a 'cringe_video' sensor
    python3 tmp/probe_attr.py <sensor_id>
"""

import json
import sys

import requests

AGENT = "http://127.0.0.1:8000"
VST = "http://127.0.0.1:30000"
LIMIT = 2500


def rule(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def post(path, body):
    rule(f"POST {path}\n  body: {json.dumps(body)}")
    try:
        response = requests.post(f"{AGENT}{path}", json=body, timeout=(15, 300))
    except Exception as exc:
        print(f"  !! {type(exc).__name__}: {exc}")
        return
    print(f"  HTTP {response.status_code}")
    text = response.text
    print("  " + (text[:LIMIT].replace("\n", "\n  ") or "<empty>"))
    if len(text) > LIMIT:
        print(f"  ... [{len(text) - LIMIT} more chars]")
    # Surface the fields we actually care about, wherever they are nested.
    hits = [k for k in ("object_id", "bbox", "frame_timestamp", "object_type")
            if k in text]
    print(f"\n  >> tracking fields present in response: {hits or 'NONE'}")


def raw_input_schema():
    """Defaults and enums the compact dump omitted."""
    rule("AttributeSearchInput — full schema (defaults, enums)")
    spec = requests.get(f"{AGENT}/openapi.json", timeout=15).json()
    schemas = spec.get("components", {}).get("schemas", {})
    for name in ("AttributeSearchInput", "AttributeSearchMetadata"):
        print(f"\n  -- {name} --")
        print("  " + json.dumps(schemas.get(name, {}), indent=2).replace("\n", "\n  "))


def pick_sensor():
    sensors = requests.get(f"{VST}/vst/api/v1/sensor/list", timeout=10).json()
    for sensor in sensors:
        if "cringe" in (sensor.get("name") or "").lower():
            return sensor.get("sensorId"), sensor.get("name")
    first = sensors[0]
    return first.get("sensorId"), first.get("name")


def main():
    raw_input_schema()

    if len(sys.argv) > 1:
        sensor_id, name = sys.argv[1], None
    else:
        sensor_id, name = pick_sensor()
    print(f"\n\n>>> testing against sensor {sensor_id}  (name: {name})")

    bodies = [
        {"query": "person", "top_k": 100},
        {"query": "person", "top_k": 100, "video_sources": [sensor_id]},
    ]
    if name:
        bodies.append({"query": "person", "top_k": 100, "video_sources": [name]})

    for body in bodies:
        post("/api/v1/attribute_search", body)

    # /full streams raw intermediate steps — may expose the tool's own rows.
    post("/api/v1/attribute_search/full",
         {"query": "person", "top_k": 100, "video_sources": [sensor_id]})


if __name__ == "__main__":
    main()
