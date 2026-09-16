# Per-person tracking

How MoveMatch gets **each person's position at each timestep** out of VSS, and
renders it back over the uploaded clip as a bounding-box overlay.

New files: `app/tracking.py` (fetch + clean the tracks), `app/overlay.py`
(draw them). `app/app.py`, `app/process_video.py`, the template and the CSS
were modified to wire it in. Throwaway probes live in `tmp/`.

---

## The pipeline

```
upload  →  VST  →  /complete  →  RTVI-CV  →  Kafka  →  Elasticsearch (mdx-raw-*)
                        │                                      │
                        └──→ agent /chat (video_understanding)  └──→ tracking.py → overlay.py
                                    │                                        │
                                 winner number                        overlay.mp4
```

One upload feeds **two independent consumers**: the VLM question that picks the
winner, and the CV metadata that produces the overlay. They fail separately —
a tracking failure flashes a message and still renders the game result.

## How the VSS agent is called

Unchanged from before, just split apart. `process_video.py` does:

1. `POST /api/v1/videos` — get a chunked-upload URL.
2. Chunked `PUT`/`POST` into **VST** with the nvstreamer headers → `sensorId`.
3. `POST /api/v1/videos/{sensor_id}/complete` — **this is the step that runs
   RTVI-CV** (detection + tracking) and RTVI-Embed over the clip. It used to be
   swallowed as a harmless warning; without it there is no tracking data at
   all, so `upload_video()` now returns `(sensor_id, cv_ready)`.
4. `POST /chat` (endpoint auto-discovered) with a `video_understanding` prompt.

The upload/ask split matters: `query_vss_agent_video()` used to upload *and*
ask in one call, so adding tracking would have meant uploading twice. It is now
`upload_video()` + `ask_agent(sensor_id, prompt)`, with the old function kept
as a thin wrapper.

`purge_videos()` deletes any sensor already registered under the same filename
before uploading. RTVI-CV rejects a duplicate camera id
(`STREAM_ADD_FAIL, Duplicate Camera id`) and that silently costs the whole CV
run. Sensor deletion after the run is **opt-in** (`DELETE_SENSORS=1`): deleting
a sensor also drops its CV frames from Elasticsearch.

One upload leaves traces in **three** registries — the VST sensor list and the
VSS asset store, both keyed by the stream UUID, and `mdx-raw-*`, keyed by the
video *name* — and they are cleaned up independently. Deleting the sensor does
not always drop the asset, and VST hands the same stream id back for a file of
the same name, so the orphan comes back and `/complete` fails the fan-out with

```
502  Embedding generation failed ... {"code":"AssetAlreadyExists",
     "message":"Asset with id <uuid> already exists."}
```

which looks impossible, because that UUID was issued seconds earlier. Hence:
`purge_videos()` re-reads the registry and warns about survivors rather than
counting successful `DELETE`s, and `upload_video()` clears the id with
`drop_asset()` and retries `/complete` once before giving up. The Flask app
rarely hits this — it saves every upload as `<uuid4>_<name>.mp4`, so each run has
its own filename. `tracking_walkthrough.ipynb`, which keeps the readable name
`dance_vst` on purpose, collides with itself on every re-run; step 5 there also
falls back to a stamped filename, and `UNIQUE_UPLOAD_NAME = True` in step 0
sidesteps the collision from the start.

## Where the tracking data actually lives

**Elasticsearch, index `mdx-raw-*`, read directly.** One document per frame:

```json
{"sensorId": "<video name>",                  // NOT the VST sensor UUID
 "id": "50",                                   // frame number
 "timestamp": "2025-01-01T00:00:02.080Z",      // VSS_UPLOAD_TS + offset into the clip
 "objects": [{"id": "1", "type": "Person",     // objects[].id is the tracker id
              "bbox": {"leftX": 736.5, "topY": 156.4,
                       "rightX": 1845.9, "bottomY": 1434.0},
              "confidence": 0.84}]}
```

This is MDX (Metropolis Data eXchange) — the per-frame CV stream RTVI-CV emits
onto Kafka. Grouping by `objects[].id` gives one position timeline per person,
at full framerate (~24–30 fps).

Two gotchas: `sensorId` holds the **video name** (uploaded filename minus
extension), not the VST UUID; and it is mapped as `text`, so term queries and
sorts need `sensorId.keyword`. Object embeddings dominate the index size
(611 MB), so every query passes `_source_includes` and never pulls them.

## Which VSS tool is linked to it — and why we don't use it

**`attribute_search`** (`POST /api/v1/attribute_search[/full]` on the agent) is
the VSS-supported consumer of this same store. Its result rows
(`AttributeSearchMetadata`) carry exactly the fields we want:

```
sensor_id, object_id, object_type, frame_timestamp, bbox, behavior_score
```

It is nonetheless the wrong door. `attribute_search` is a **similarity search**:
its own schema describes `frame_timestamp` as *"Best frame timestamp"*, with
`start_time`/`end_time` being *"earliest/latest from duplicates"*. It
deduplicates to one best-frame row per object — which is precisely the timeline
we need, aggregated away by design. Scoping also goes through `video_sources`
(by video **name**), not a `sensor_id` field.

So: `attribute_search` for "find me the person in the red shirt";
`mdx-raw-*` for "where was everyone, frame by frame".

The trade-off of reading the store directly is a dependency on VSS's internal
index schema rather than a public API.

## How we found it

The plan assumed the agent would serve the metadata. It didn't, and the route
there was mostly elimination (probes preserved in `tmp/`):

| Probe | Result |
| --- | --- |
| `GET /api/v1/videos` | 405 — the sensor registry has to be read from VST's `/vst/api/v1/sensor/list` |
| `POST /api/v1/attribute_search/full` | 422 `{"loc":["body","query"]}` — route exists, wrong payload shape. The 422 *was* the schema |
| `/vst/api/v1/sensor/{id}/metadata` | structured `CameraNotFoundError`, not an nginx 404 — real route, but 405 on GET for the metadata variants. Dead end |
| `attribute_search` with the right shape | real rows, but one deduplicated row per object — no timeline |
| `ss -ltnp` port scan | **the turning point:** 9200 Elasticsearch, 5601 Kibana, 9092 Kafka, 6379 Redis, 8017/8018 RT-VLM |
| `GET :9200/_cat/indices` | `mdx-raw-2025-01-01`, 74,175 docs / 611 MB — the per-frame stream |

The port scan is what reframed the problem: seeing Kafka and Elasticsearch
side by side identified the standard VSS CV path and made it obvious that
`attribute_search` is a *consumer* of the store, not the store itself.

Worth recording as a correction: early on, the `Duplicate Camera id` error was
read as proof that the CV pipeline was deployed and reachable through the
agent. It only proved a tracker **ran** — not that the agent exposes its
output. Those are different claims and conflating them cost several probes.

## What changed versus the approved plan

| Planned | Built | Why |
| --- | --- | --- |
| Agent `attribute_search` as the source | Elasticsearch `mdx-raw-*` direct | search API dedupes the timeline away |
| OpenCV (`opencv-python-headless`) for drawing | **ffmpeg `drawbox`/`drawtext`** | ffmpeg is already a hard dependency for VST re-encoding; zero new installs, and no aarch64 wheel to source |
| Read tracks right after upload | `wait_for_cv()` polls first | RTVI-CV runs async and slower than real time; reading early truncated an 11 s clip to 0.33 s of tracking |
| Raw tracker ids trusted | IoU merge + confidence filter | the tracker re-issues ids, so one person wore 2–3 boxes at once |

No new Python dependency was added. `pyproject.toml` only gained a comment
recording the ffmpeg/ffprobe requirement.

## Cleaning the tracks (`tracking.py`)

Raw tracker output was not directly usable — one person routinely carried two
or three concurrent boxes. Three filters, all env-tunable:

- **Coasted-box rejection** (`TRACKING_MIN_CONFIDENCE`, default `0.0`) — MDX
  writes a non-positive confidence when the detector did not fire and the
  position is *predicted*. Those are the boxes that drift off a person and land
  on someone else.
- **IoU duplicate merge** (`TRACKING_IOU_MERGE`, default `0.55`) — each frame is
  treated as a small non-maximum suppression: longest-lived, most-confident box
  wins, overlapping boxes are recorded as duplicates *of it*, and union-find
  folds an id into its winner. Merging ids rather than deleting boxes is what
  keeps one stable label for the whole clip.
  `TRACKING_MERGE_DOMINANCE` (`0.5`) guards against over-merging: an id must
  spend most of its life losing to the *same* winner, so two people genuinely
  standing close stay separate.
- **Flicker rejection** (`TRACKING_MIN_POINTS` 5, `TRACKING_MIN_SECONDS` 0.3),
  with a fallback that never filters every track away.

There is also a **double-ingest warning**: more documents than distinct
timestamps means the clip was ingested twice under one name and every person is
duplicated at every instant. No IoU threshold fixes that — the stale sensor has
to be deleted and the clip re-uploaded. The two causes produce identical
symptoms and need opposite fixes, so the check runs before any tuning.

Output shape:

```python
[{"track_id": "1", "points": [{"t": 2.08, "bbox": (x1, y1, x2, y2)}, ...]}]
```

`t` in seconds from the start of the clip, bboxes normalised 0–1 (MDX reports
pixels, so `fetch_tracks` needs the frame size).

## Rendering (`overlay.py`)

ffmpeg `drawbox` + `drawtext`, one pair per span, written to a
`-filter_script:v` file because the graph is far too long for a command line.

- Samples are **decimated and merged** until the graph fits
  `MAX_FILTER_ENTRIES` (4000): keep one sample per 0.1 s, then collapse
  consecutive near-identical boxes into one span. It doubles the step until it
  fits.
- Each box is **held 0.5 s** past its last sample — CV metadata is sparser than
  the video framerate and drawing only on exact matches flickers badly.
- Labels are `P1, P2, …` by order of first appearance, not raw tracker ids,
  with a colour per track from a fixed palette.
- Output is H.264 baseline + `+faststart` so the browser can play it inline.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `VSS_ES_URL` | `http://127.0.0.1:9200` | Elasticsearch |
| `MDX_INDEX` | `mdx-raw-*` | date-suffixed, rolls over |
| `VSS_UPLOAD_TS` | `2025-01-01T00:00:00` | frame timestamps are offsets from this |
| `VSS_VST_URL` | `http://127.0.0.1:30000` | sensor registry |
| `TRACKING_MAX_FRAMES` | `10000` | ES `max_result_window` |
| `TRACKING_IOU_MERGE` / `_MERGE_DOMINANCE` / `_MIN_CONFIDENCE` | `0.55` / `0.5` / `0.0` | duplicate suppression |
| `TRACKING_MIN_POINTS` / `_MIN_SECONDS` | `5` / `0.3` | flicker rejection |
| `KEEP_UPLOADS` | set in `run.sh` | now actually honoured by `app.py` |
| `DELETE_SENSORS` | unset | opt-in; deleting a sensor drops its CV frames |

## Checking it without the UI

```bash
# tracks straight out of Elasticsearch, no upload
python3 app/tracking.py <video_name> <width> <height>

# what CV has written for a clip
curl -s 'localhost:9200/mdx-raw-*/_search?size=0' -H 'Content-Type: application/json' \
  -d '{"query":{"term":{"sensorId.keyword":"<video_name>"}},
       "aggs":{"instants":{"cardinality":{"field":"timestamp"}}}}'
```

`hits.total` ≫ `instants` ⇒ the clip is in there more than once.

## Open items

- Duplicate-track tuning was still in progress: `tmp/tune_dedup.py` sweeps
  confidence × IoU against an expected person count. The last run on
  `dance_vst` was blocked by a bug (`objects.confidence` was missing from the
  fetched `_source`, so every detection failed the confidence filter) — that is
  fixed, but the sweep has not been re-run since.
- The `dance_vst` double-ingest question (657 docs for an 11 s clip at ~30 fps,
  roughly double expected) was never resolved. Settle it with the cardinality
  query above **before** tuning thresholds.
- The full upload → overlay path through the Flask UI has been exercised less
  than the read-from-ES path.
