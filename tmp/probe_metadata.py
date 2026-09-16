#!/usr/bin/env python3
"""Find where per-person tracking metadata lives. Read-only.

Follows up the first discovery run, which established:
  - /api/v1/attribute_search/full wants a `query` field (422 told us so)
  - /vst/api/v1/sensor/{id}/metadata is a real route (structured
    CameraNotFoundError, not an nginx 404) — it just needs a LIVE sensor id
  - GET /api/v1/videos is 405; the registry must be read from VST

Usage:
    python3 tmp/probe_metadata.py             # uses a live sensor from VST
    python3 tmp/probe_metadata.py <sensor_id> # pin one
"""

import json
import sys

import requests

AGENT = "http://127.0.0.1:8000"
VST = "http://127.0.0.1:30000"
LIMIT = 900


def rule(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def show(label, fn):
    rule(label)
    try:
        response = fn()
    except Exception as exc:
        print(f"  !! {type(exc).__name__}: {exc}")
        return None
    print(f"  HTTP {response.status_code}")
    print("  " + (response.text[:LIMIT].replace("\n", "\n  ") or "<empty>"))
    try:
        return response.json()
    except ValueError:
        return None


# --------------------------------------------------------------------------

def dump_schemas():
    """The agent's own schemas say whether bbox/track fields exist at all."""
    rule("agent schemas: does the search result carry boxes and ids?")
    try:
        schemas = requests.get(f"{AGENT}/openapi.json", timeout=15).json()
        schemas = schemas.get("components", {}).get("schemas", {})
    except Exception as exc:
        print(f"  !! {exc}")
        return

    for name in ("AttributeSearchInput", "AttributeSearchResult",
                 "AttributeSearchMetadata", "SearchInput", "SearchResult",
                 "EmbedSearchResultItem", "VideoInfo", "VideoResult"):
        node = schemas.get(name)
        if not node:
            continue
        print(f"\n  -- {name} --")
        print(f"     required: {node.get('required', [])}")
        for field, spec in (node.get("properties") or {}).items():
            kind = spec.get("type") or spec.get("anyOf") or spec.get("$ref") or "?"
            if isinstance(kind, list):
                kind = "/".join(str(k.get("type", k)) for k in kind)
            print(f"     {field:24} {str(kind)[:60]}")


def vst_spec():
    """VST's own route list would settle this outright."""
    for path in ("/vst/api/v1/openapi.json", "/vst/openapi.json",
                 "/vst/api/v1/swagger.json", "/vst/api/swagger.json"):
        data = show(f"VST spec {path}", lambda p=path: requests.get(f"{VST}{p}", timeout=10))
        if isinstance(data, dict) and data.get("paths"):
            rule("VST routes mentioning metadata/object/event/track/analytic")
            for route in sorted(data["paths"]):
                if any(k in route.lower() for k in
                       ("metadata", "object", "event", "track", "analytic", "alert")):
                    print(f"     {route}  {list(data['paths'][route])}")
            return True
    return False


def live_sensor():
    try:
        sensors = requests.get(f"{VST}/vst/api/v1/sensor/list", timeout=10).json()
    except Exception as exc:
        print(f"  !! could not list sensors: {exc}")
        return None
    rule("live sensors in VST")
    for sensor in sensors:
        print(f"  {sensor.get('sensorId')}  {sensor.get('name')}  "
              f"state={sensor.get('state')}")
    return sensors[0].get("sensorId") if sensors else None


def probe(sensor_id):
    rule(f"probing metadata routes for LIVE sensor {sensor_id}")

    for path in (
        f"/vst/api/v1/sensor/{sensor_id}/metadata",
        f"/vst/api/v1/sensor/{sensor_id}",
        f"/vst/api/v1/metadata/timeline?sensorId={sensor_id}",
        f"/vst/api/v1/recording/list?sensorId={sensor_id}",
        f"/vst/api/v1/timeline/{sensor_id}",
        f"/vst/api/v1/events?sensorId={sensor_id}",
        f"/vst/api/v1/alerts?sensorId={sensor_id}",
    ):
        show(f"VST {path}", lambda p=path: requests.get(f"{VST}{p}", timeout=30))

    # The corrected agent payload — `query` is the required field.
    for body in (
        {"query": "every person with bounding box and track id at every timestamp",
         "sensor_id": sensor_id},
        {"query": "list all detected persons with their bounding boxes and track ids"},
    ):
        show(f"agent POST /api/v1/attribute_search/full  body={json.dumps(body)[:90]}",
             lambda b=body: requests.post(
                 f"{AGENT}/api/v1/attribute_search/full", json=b, timeout=(15, 300)))


def main():
    dump_schemas()
    vst_spec()
    sensor_id = sys.argv[1] if len(sys.argv) > 1 else live_sensor()
    if not sensor_id:
        print("\nNo live sensor. Upload one first, and do NOT delete it:")
        print("  python3 tmp/check_tracking.py <video.mp4>   # deletes on exit")
        return
    probe(sensor_id)


if __name__ == "__main__":
    main()
