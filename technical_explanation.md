# MoveMatch — Technical Explanation

How the notebook decides **which person, standing in a line, is the first to
perform a given move** (or to complete an ordered sequence of moves) — and then
produces a still image with that "winner" boxed.

People are always numbered **left → right**: person 1 is leftmost. Every stage
keeps this numbering, so the same "person N" means the same body throughout.

We found **two independent ways** to find the winner. They answer the
same question with different trade-offs:

| | Our method — VLM captions + LLM | Other option — pose geometry |
|---|---|---|
| Handles | *any* move, described in plain English | one hard-coded move: arms crossed |
| Runs on | remote services over HTTP (a vision-language model + a text model) | fully local, on CPU |
| Timing resolution | ~1–2 s (the caption chunk size) | ~0.1 s per frame |
| Deterministic | no (model outputs vary) | yes (same video → same answer) |

Use Method A for flexibility; use Method B when you need exact, repeatable
sub-second ordering of a near-simultaneous move.

Both methods return the same little result dictionary:

```python
{"winner": int,      # person number, 0 == nobody did it
 "timestamp": float, # seconds when they finish (None if nobody)
 "num_people": int}  # total people seen
```

---

## Method: VLM captions + LLM reasoning

This path never looks at pixels itself. It asks a vision-language model to
*describe* the video in words, then asks a text model to *reason* over those
words. Two functions, run in sequence.

### 1. `caption_video` — turn the clip into a timestamped timeline

1. Upload the clip to the vision service (`POST /files`).
2. Ask it to densely caption the clip (`POST /generate_vlm_captions`),
   split into fixed windows of `chunk_duration` seconds. Each window is
   summarized from at most **5 sampled frames** (the model caps a prompt at 5
   images), and each yields one timestamped caption.
3. Delete the uploaded file, strip any `<think>…</think>` reasoning the model
   leaked, and return the captions sorted by start time as
   `[(start, end, text), …]`.

The caption prompt is deliberately **generic** — "number each person left to
right and describe every action, pose, and move" — so the same captions can
later answer questions about *any* move, not a fixed list.

**Why `chunk_duration` matters.** It is the timeline's temporal resolution.
Make it too large and a brief action near the end of the clip gets
averaged into a static "everyone standing" caption and disappears.

### 2. `first_performer` — reason over the timeline

1. Flatten the captions into one text block: `[start - end] caption` per line.
2. Send **one** prompt to the text model (`POST /v1/chat/completions`,
   `temperature = 0` for repeatability). The prompt states that people are
   numbered left-to-right and keep their number, pastes the timeline, and asks
   for the **earliest** person to either perform a single move or complete a
   list of moves *in order*.
3. The model must answer with JSON only; the function extracts the JSON and
   returns it.

`moves` can be a single string (`"cross their arms"`) or an ordered list
(`["cross arms", "turn around", "touch your head"]`). Ordering is enforced by
the prompt wording, not by code — the model decides whether the moves appear in
sequence.

> **Known limitation.** The captions list people in numeric order, so when two
> people do the move in the same 1–2 s window, the model can't tell who was
> truly first and tends to default to person 1. When exact ordering of a
> near-simultaneous move matters, use Method B.

---

## Presenting the winner — `winner_snapshot`

Once a method returns a winner and timestamp:

1. Seek the video to that timestamp and grab the frame.
2. Run **YOLOX** (a separate ONNX person detector, also via `cv2.dnn`) to get
   real person bounding boxes, sorted left → right. Using a real detector avoids
   the boxes drifting onto walls or doors, which happened when box coordinates
   were requested from the VLM.
3. Box the **Nth detection from the left** (the winner) with an amber rectangle
   and a `WINNER: Person N @ ts` label, and save it as a JPEG. If the detector
   finds fewer than N people, it falls back to an equal-width column split so an
   image is still produced.

---

## Other method — deterministic "who crossed their arms first"

This path uses no language model at all. It runs a pose estimator on sampled
frames and decides "arms crossed" from geometry, giving an exact onset time per
person.

### B1. `pose_people` — 17 keypoints per person

Each frame is letterboxed to 640×640 and run through **YOLOv8n-pose** (ONNX, via
OpenCV's `cv2.dnn`, on CPU). For every detection above the confidence threshold
it returns the 17 COCO keypoints (shoulders, wrists, hips, …) in original pixel
coordinates. Overlapping detections are removed with non-max suppression, and
people are sorted left → right.

### B2. `arms_crossed` — the geometric test

Given one person's keypoints, arms are "crossed" when **both** hold:

- **Wrists pulled together.** The horizontal gap between the wrists, divided by
  shoulder width, is small. Arms at the sides give a ratio near 1; crossed
  wrists meet or swap over the chest, driving the ratio toward 0 or negative.
  Dividing by shoulder width makes the test scale-invariant (independent of how
  far the person is from the camera).
- **Wrists at chest height, not overhead.** Both wrists must sit at or below
  shoulder level, so a "hands up" pose (also a small gap, but raised) doesn't
  count.

Low-visibility keypoints are rejected first so the test isn't fooled by missing
joints.

### B3. `first_to_cross_arms` — time the onset per person

1. Sample the video every `dt` seconds (default 0.1 s) by seeking with
   `cv2.CAP_PROP_POS_MSEC` and running `pose_people`.
2. Build fixed **column centers** from the frame that saw the most people; this
   is the left-to-right reference for numbering.
3. Assign each detection in each frame to its nearest column, and record the
   **first** timestamp at which each person satisfies `arms_crossed`.
4. The winner is the person with the earliest onset.

Because there's no model in the timing loop, near-simultaneous crosses are
ordered correctly and the result is fully reproducible.

---

## Models used

Downloaded once and cached under `./models` (~50 MB total):

| Model | Role |
|-------|------|
| **YOLOX** (`yolox.onnx`, ~34 MB) | person detection for the winner snapshot |
| **YOLOv8n-pose** (`yolov8n-pose.onnx`, ~12 MB) | 17-keypoint pose for the deterministic arms-crossed timing |

Both run through OpenCV's `cv2.dnn` module on CPU — no GPU or extra deep-learning
framework required.

## Key parameters

| Parameter | Where | Default | Effect |
|-----------|-------|---------|--------|
| `chunk_duration` | `caption_video` | 1 s | Caption window = Method A's time resolution |
| `num_frames_per_chunk` | `caption_video` | 5 | Frames the VLM sees per window (hard cap 5) |
| `temperature` | `first_performer` | 0.0 | Kept at 0 for repeatable LLM answers |
| `dt` | `first_to_cross_arms` | 0.1 s | Frame sampling interval for pose timing |
| `conf_th` (pose) | `pose_people` | 0.5 | Minimum pose-detection confidence |
| `ratio_th` | `arms_crossed` | 0.55 | Wrist-gap / shoulder-width cutoff for "crossed" |
| `conf_th` (person) | `detect_people` | 0.35 | Minimum person-detection confidence (YOLOX) |

---

# VSS Backend Architecture

The notebook talks to a **VSS** backend over HTTP (`caption_video` → `/generate_vlm_captions`,
`first_performer` → the text NIM). This section documents how that backend is deployed
(`deploy/docker/local_deployment_single_gpu`).

## Version & shape

- **VSS engine 2.4.1** (`nvcr.io/nvidia/blueprint/vss-engine:2.4.1`), docker-compose.
- Backend API on **`:8100`**, web UI on **`:9100`**.
- One GPU container — **`via-server`**, pinned to **GPU 1** (`NVIDIA_VISIBLE_DEVICES=1`) — plus
  CPU-only infra containers: **neo4j** (graph DB), **milvus** (vector DB), **arango**, **minio**,
  **elasticsearch**.
- The VLM and the CA-RAG models run as **separate external NIM containers**, reached over
  `host.docker.internal`. They are *not* part of the compose file.

## Models on the text / CA-RAG side (`config.yaml`)

| Model | Port | Use |
|-------|------|-----|
| **nvidia/nemotron-3.5-lightning** (30b-a3b) | `:8007` | `chat_llm` + `summarization_llm` + `notification_llm` — graph ingestion & retrieval, caption→summary aggregation, alert notifications |
| **nvidia/llama-nemotron-embed-vl-1b-v2** | `:8006` | Embeddings for the neo4j graph DB and milvus vector DB |
| **nvidia/llama-nemotron-rerank-vl-1b-v2** | `:8005` | Reranker for retrieval |

These power **CA-RAG**: captions are ingested into the graph/vector DBs, then chat/summary
queries retrieve + rerank + reason over them.

## Vision model (`.env`)

`VLM_MODEL_TO_USE=openai-compat` → **`nvidia/cosmos3-nano-reasoner`** (served by the
`cosmos3-reasoner` NIM) at **`:38011`**. Its job is **dense captioning** of each video chunk:
VSS samples **≤5 frames per chunk** (the model caps a prompt at 5 images) and asks the VLM to
describe them. This is the model that receives the Set-of-Marks overlaid frames.

## CV detection + tracking (`DISABLE_CV_PIPELINE=false`)

Running inside `via-server` on GPU 1:

- **GroundingDINO** detector (auto-downloaded from NGC) finds people.
- **NvDCF tracker** (+ **ReID**, + **SAM2** engines built at first start) assigns each person a
  **stable tracking ID**.
- The IDs are **burned onto the frames** (Set-of-Marks) *before* the VLM sees them, so the VLM
  gets one view per frame: the numbered one.
- Detector cadence = `GDINO_INFERENCE_INTERVAL` (default 1 = detect every other frame); the
  tracker labels **every** frame in between, so the overlay is full-frame-rate.

Setup + troubleshooting detail: see `deploy/docker/local_deployment_single_gpu/CV_TRACKING_SETUP.md`.

## Scheme 1 — request pipeline

```
                                 ┌──────────────────── GPU 1 (via-server) ────────────────────┐
  video ──▶ decode ──▶ CV pipeline: GDINO detect + NvDCF track (+ReID) ──▶ Set-of-Marks overlay
                                 └──────────────────────────────┬──────────────────────────────┘
                                                                │ sample ≤5 frames/chunk
                                                                ▼
                                              VLM dense caption  (cosmos3, GPU 1, :38011)
                                                                │  timestamped captions
                                                                ▼
                        ┌──────────────────── GPU 0 (CA-RAG NIMs) ────────────────────┐
                        │  embed (:8006) ─▶ store        rerank (:8005) ─▶ retrieve    │
                        │            nemotron-3.5-lightning LLM (:8007) summarize/reason│
                        └───────────────────────────────┬──────────────────────────────┘
                                                         │        (graph=neo4j, vector=milvus — CPU)
                                                         ▼
                                             summary / chat answer ──▶ client (:8100 / UI :9100)
```

## Scheme 2 — model placement across the two GPUs

```
┌─────────────────── GPU 0  (~92 GB used — saturated) ───────────────────┐
│  nemotron-3.5-lightning  (vLLM)   :8007   ~77 GB   CA-RAG LLM           │
│  llama-nemotron-embed-vl-1b-v2    :8006   ~8 GB    embeddings           │
│  llama-nemotron-rerank-vl-1b-v2   :8005   ~6 GB    reranker             │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────── GPU 1  (~60 GB used · ~35 GB free) ─────────────────┐
│  cosmos3 VLM             (vLLM)   :38011  ~49 GB   dense captioning     │
│  via-server: decode + CV pipeline (GDINO/NvDCF/ReID/SAM2)  ~10 GB       │
└─────────────────────────────────────────────────────────────────────────┘

CPU-only containers: neo4j · milvus · arango · minio · elasticsearch
```

The cosmos NIM is memory-capped with `NIM_GPU_MEMORY_UTILIZATION=0.50` specifically to leave
headroom on GPU 1 for the CV pipeline's TensorRT engine builds and inference.

