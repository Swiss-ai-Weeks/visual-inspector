#!/usr/bin/env python3
"""Is the per-frame CV metadata in Elasticsearch? Read-only.

The port scan found Elasticsearch on 9200 and Kafka on 9092 — the usual VSS
path is RTVI-CV -> Kafka -> Elasticsearch, with attribute_search reading from
the store and deduplicating. The raw per-frame rows should be in the index.

Usage:
    python3 tmp/probe_es.py [name_fragment]     # default: cringe
"""

import json
import sys

import requests

ES = "http://127.0.0.1:9200"


def rule(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def get(path, **kwargs):
    try:
        response = requests.get(f"{ES}{path}", timeout=30, **kwargs)
        return response
    except Exception as exc:
        print(f"  !! {type(exc).__name__}: {exc}")
        return None


def indices():
    rule("indices (docs.count is the tell — per-frame data is BIG)")
    response = get("/_cat/indices?v&s=docs.count:desc&h=index,docs.count,store.size")
    if not response:
        return []
    print("  " + response.text.replace("\n", "\n  "))
    names = []
    for line in response.text.splitlines()[1:]:
        parts = line.split()
        if parts:
            names.append(parts[0])
    return names


def sample(index, fragment):
    rule(f"index: {index}")

    mapping = get(f"/{index}/_mapping")
    if mapping is not None and mapping.ok:
        fields = json.dumps(mapping.json())
        interesting = [k for k in ("object", "bbox", "coordinates", "id",
                                   "timestamp", "sensor", "track", "frame")
                       if f'"{k}' in fields]
        print(f"  mapping mentions: {interesting}")

    # A couple of raw documents say more than any schema.
    response = get(f"/{index}/_search?size=2")
    if response is None or not response.ok:
        print(f"  !! search failed: {response.status_code if response else '?'}")
        return
    hits = response.json().get("hits", {}).get("hits", [])
    if not hits:
        print("  (empty index)")
        return
    for hit in hits:
        print("  " + json.dumps(hit.get("_source", {}), indent=2)[:1200]
              .replace("\n", "\n  "))
        print()

    # How many documents mention our test video? Many == a real timeline.
    query = {"query": {"query_string": {"query": f"*{fragment}*"}}}
    response = requests.post(f"{ES}/{index}/_count", json=query, timeout=30)
    if response.ok:
        print(f"  >> docs matching '*{fragment}*': {response.json().get('count')}")


def main():
    fragment = sys.argv[1] if len(sys.argv) > 1 else "cringe"

    names = indices()
    if not names:
        print("\nNo indices readable. Elasticsearch may need auth.")
        return

    # Look at the biggest non-system indices; per-frame CV data will dominate.
    candidates = [n for n in names if not n.startswith(".")][:4]
    print(f"\n\ninspecting: {candidates}")
    for index in candidates:
        sample(index, fragment)


if __name__ == "__main__":
    main()
