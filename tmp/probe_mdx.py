#!/usr/bin/env python3
"""Pin down the mdx-raw document shape so tracking.py can query it. Read-only.

mdx-raw-2025-01-01 holds the per-frame CV output: objects[].id (tracker id),
objects[].type, objects[].bbox{leftX,topY,rightX,bottomY}. Still needed:
  - the top-level field names for sensor and timestamp
  - which value identifies our video (VST sensorId, or the name?)
  - whether Person rows exist, and at what frame rate

Embeddings are excluded from every query — they dominate the 611MB index.

Usage:
    python3 tmp/probe_mdx.py [name_fragment]      # default: cringe
"""

import json
import sys

import requests

ES = "http://127.0.0.1:9200"
INDEX = "mdx-raw-2025-01-01"
NO_VECTORS = "objects.embedding,objects.bbox.embeddings,embeddings"


def rule(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def search(body, size=2):
    response = requests.post(
        f"{ES}/{INDEX}/_search",
        params={"size": size, "_source_excludes": NO_VECTORS},
        json=body, timeout=60,
    )
    if not response.ok:
        print(f"  !! HTTP {response.status_code}: {response.text[:400]}")
        return {}
    return response.json()


def top_level_fields():
    rule("top-level fields (one document, vectors stripped)")
    hits = search({"query": {"match_all": {}}}).get("hits", {}).get("hits", [])
    for hit in hits[:1]:
        print("  " + json.dumps(hit.get("_source", {}), indent=2)[:2000]
              .replace("\n", "\n  "))


def sensor_values():
    """Which field identifies the video, and what values does it hold?"""
    rule("distinct sensors in the index")
    for field in ("sensorId", "sensorId.keyword", "sensor.id", "sensor.id.keyword",
                  "place.name.keyword", "id.keyword"):
        result = search({"aggs": {"s": {"terms": {"field": field, "size": 40}}}}, size=0)
        buckets = (result.get("aggregations") or {}).get("s", {}).get("buckets")
        if buckets:
            print(f"\n  field `{field}` works:")
            for bucket in buckets:
                print(f"     {bucket['doc_count']:7}  {bucket['key']}")
            return field
    print("  !! none of the candidate sensor fields aggregated")
    return None


def person_rows(fragment):
    rule(f"Person detections for a video matching '*{fragment}*'")
    body = {
        "query": {"bool": {"must": [
            {"query_string": {"query": f"*{fragment}*"}},
            {"nested": {"path": "objects",
                        "query": {"match": {"objects.type": "Person"}}}},
        ]}},
        "sort": [{"@timestamp": {"order": "asc", "unmapped_type": "date"}}],
    }
    result = search(body, size=3)
    total = (result.get("hits", {}).get("total") or {}).get("value")
    print(f"  matching frames: {total}")
    for hit in result.get("hits", {}).get("hits", []):
        print("  " + json.dumps(hit.get("_source", {}), indent=2)[:1400]
              .replace("\n", "\n  "))
        print()

    if not total:
        # The nested query may not apply if objects isn't mapped as nested.
        print("  retrying without the nested wrapper...")
        result = search({"query": {"query_string": {"query": f"*{fragment}* AND Person"}}}, size=2)
        for hit in result.get("hits", {}).get("hits", []):
            print("  " + json.dumps(hit.get("_source", {}), indent=2)[:1400]
                  .replace("\n", "\n  "))


def main():
    fragment = sys.argv[1] if len(sys.argv) > 1 else "cringe"
    top_level_fields()
    sensor_values()
    person_rows(fragment)


if __name__ == "__main__":
    main()
