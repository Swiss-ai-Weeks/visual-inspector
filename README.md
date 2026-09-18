# 🎮 MoveMatch

> **Watch the sequence. Reproduce the moves. Let AI decide who got it right.**

MoveMatch is an AI-powered game. The app shows a short sequence of moves; a camera films everyone at once; and computer vision decides **who is the first to perform the moves correctly and in order**. That player wins the round and climbs the leaderboard.

It started as a playful spin on the **Visual Compliance Inspector** challenge. Instead of asking *"did a worker correctly follow this procedure?"* we ask *"did this player correctly follow this movement sequence?"*. The underlying AI problem is the same (detect people, recognize actions, check their order against an expected procedure), turned into a game.

---

## 🕹️ The Game

| Step | What happens |
|------|--------------|
| **1. Watch** | The app shows a movement sequence, e.g. 👏 → 🙆 → 👇 |
| **2. Perform** | Players have a few seconds (**5–10 s**, set by difficulty) to reproduce it on camera |
| **3. Analyze** | Every player is tracked with a stable ID; a vision model describes what each ID does over time |
| **4. Verify** | The app finds who completed the sequence **in the right order, first** |
| **5. Rank** | It boxes the winner in a snapshot, shows the result, and updates the leaderboard |

**Example** — expected 👏 → 🙆 → 👇:

- 👏 → 🙆 → 👇 ✅ correct order
- 👏 → 👇 → 🙆 ❌ wrong order

Difficulty sets the number of moves in the sequence (1–5); players can also enter their names so the winner is announced by name.

---

## 🚀 Running It

MoveMatch is a small Flask app (`app/`) that offloads all perception to a running **VSS backend** (see the **VSS Backend** section below). You need:

- A **VSS backend** reachable at `:8100` (VLM captions) and a **text NIM** at `:38011` (reasoning) — see the deployment section below.
- **Python ≥ 3.11** with the deps in `pyproject.toml` (`flask`, `gunicorn`, `requests`, `opencv-python`, `numpy`).
- **ffmpeg** on `PATH` (clips are transcoded to H.264 MP4 before upload) and the **docker CLI** (the winner's CV metadata is copied out of the `via-server` container with `docker cp`).

Start the server:

```bash
cd app
./run.sh          # gunicorn on http://127.0.0.1:5100
```

`run.sh` reads a few environment variables (all optional):

| Variable | Default | Purpose |
|----------|---------|---------|
| `VSS_RTVLM_URL` | `http://127.0.0.1:8100` | VLM caption endpoint |
| `VSS_LLM_URL` | `http://127.0.0.1:38011` | text-NIM reasoner |
| `MOVEMATCH_VENV` | `/home/nvidia/Documents/.venv` | virtualenv to run under |
| `SECRET_KEY` | `dev-change-me` | Flask session secret |

Pages:

- `/` — **Play**: pick difficulty, get a random challenge, record/upload, see the result.
- `/leaderboard` — win rankings and a gallery of boxed winner snapshots.
- `/test` — **Free play**: type any moves as plain English and run the pipeline.

Results are stored locally in `app/movematch.db` (SQLite) and snapshots in `app/static/snapshots/`.

---

## 🧠 How It Works

The key challenge isn't detecting a person or a single pose — it's understanding **actions over time** and their **order**, per player. The pipeline:

```text
Browser (record / upload)
        │  video
        ▼
Flask app (app/app.py)  ──ffmpeg──▶  H.264 MP4
        │
        ▼
VSS backend :8100  ──▶  CV pipeline: GroundingDINO detect + NvDCF track
        │                    (stable per-person IDs burned onto the frames)
        ▼
cosmos3 VLM  ──▶  dense per-ID captions → timestamped timeline
        │
        ▼
text NIM :38011  ──▶  first ID to complete the sequence, in order
        │
        ▼
CV metadata bbox  ──▶  boxed winner snapshot (amber "WINNER @ ts")
        │
        ▼
SQLite  ──▶  result + leaderboard
```

For every player the system asks: did they perform the correct movements, in the correct order, within the time limit — and who got there first?

---

## 🔬 Technical Deep Dive

How does the app decides **which person is the first to complete an ordered sequence of moves** and then produces a still image with that "winner" boxed ?

The main module is `app/vss.py`, driven by `app/app.py`.

People are identified by a **stable tracker ID** assigned by the VSS CV pipeline (GroundingDINO detector + NvDCF tracker). The ID is **burned onto every frame the VLM sees** (Set-of-Marks), so the same "Person N" refers to the same body across the whole clip. IDs are **0-based** (Person 0 is a real person). The winner is returned as a tracker ID; for display it is later mapped to a left-to-right *starting* rank (see `start_order`).

The perception path is a single pipeline: a vision-language model *describes* the tracked video in words, then a text model *reasons* over those words to pick the winner. Three functions run in sequence (`caption_video` → `first_performer` → `winner_snapshot_cv`), orchestrated by `find_winner`.

`find_winner` returns:

```python
{"winner": int,       # tracker id, -1 == nobody did it
 "timestamp": float,  # seconds when they finish (None if nobody)
 "num_people": int,   # total people seen
 "cv_marker": int,    # marker used to locate this run's CV metadata
 "method": "vss",
 "timeline": list}    # [(start, end, text), ...] the captions
```

Test for "nobody" with `timestamp is None` (not truthiness), since tracker id 0 is a real person.

### 1. `caption_video` — turn the clip into a timestamped, per-ID timeline

1. Record `cv_marker` (the newest fused-CV-metadata mtime *before* this run) so the metadata this run produces can be located afterwards.
2. Upload the clip to the VSS backend (`POST /files` at `:8100`).
3. Ask it to densely caption the clip (`POST /generate_vlm_captions`), split into windows of `chunk_duration` seconds with `chunk_overlap_duration` seconds of overlap so a move straddling a boundary still lands inside one caption. Each window is summarized from at most **5 sampled frames** (cosmos3 caps a prompt at 5 images). `enable_cv_metadata=True` with `cv_pipeline_prompt="person"` runs the CV pipeline and overlays the stable tracker IDs.
4. Delete the uploaded file, strip any `<think>…</think>` reasoning the model leaked, and return `(captions, cv_marker)` where captions is sorted by start time as `[(start, end, text), …]`.

The caption prompt tells the VLM to describe every action, pose, and move, **always referring to a person by the ID drawn on them** — and to describe nothing if no IDs are present. This keeps the numbering consistent with the boxes drawn later.

**Why `chunk_duration` matters.** It is the timeline's temporal resolution. Make it too large and a brief action gets averaged into a static "everyone standing" caption and disappears.

### 2. `first_performer` — reason over the timeline

1. Flatten the captions into one text block: `[start - end] caption` per line.
2. Send **one** prompt to the text NIM (`POST /v1/chat/completions` at `:38011`, `temperature = 0` for repeatability). The prompt explains that each person has a fixed ID that stays constant over time, pastes the timeline, and asks for the **earliest** person to perform a single move or complete a list of moves *in order*.
3. The model must answer with JSON only; the function extracts the JSON, coerces string numbers to real types, and returns `{"winner": int, "timestamp": float|None, "num_people": int}` with `winner == -1` meaning nobody.

`moves` can be a single string (`"cross your arms"`) or an ordered list (`["cross arms", "turn around", "touch your head"]`). Ordering is enforced by the prompt wording, not by code — the model decides whether the moves appear in sequence.

### 3. Presenting the winner — CV metadata, not a separate detector

The winner box comes straight from the CV pipeline's own tracker output, so the box and the caption ID always agree. Three functions:

- **`fetch_cv_metadata`** — the CV pipeline writes a fused metadata JSON (id + bbox per frame) inside the `via-server` container, named with its own internal request id (not the caption API's response id). Since that directory isn't mounted, the file is copied out with `docker cp`, picking the newest fused file created **after** `cv_marker`.
- **`start_order`** — iterates frames in time order, records each tracker id's bbox-center x the first time it appears, and ranks ids by that x. This gives a stable left-to-right *starting* rank (used to name the winner "Person N"), even if people move or swap later in the clip.
- **`winner_snapshot_cv`** — picks the metadata frame nearest `timestamp` that contains the winner's id, seeks the video to `timestamp`, scales the stored bbox to the frame's real resolution, and draws an amber rectangle with a `WINNER @ ts s` label. Because the box is the tracker's own bounding box, it can't drift onto walls or doors the way VLM-requested coordinates did.

### Models used

The app itself loads **no local models** — all detection and tracking runs server-side inside the VSS `via-server` container:

| Component | Role |
|-----------|------|
| **GroundingDINO** (in `via-server`) | person detection |
| **NvDCF tracker** (+ ReID, SAM2) | stable per-person tracking IDs |
| **cosmos3-nano-reasoner** VLM (`:8100` → `:38011`) | dense per-ID captioning |
| **text NIM** (`:38011`, `/v1/chat/completions`) | reasoning over the timeline |

On the client side, `app/vss.py` only uses OpenCV (`cv2`) to read a frame and draw the winner box, and `docker cp` to fetch the CV metadata.

### Key parameters

| Parameter | Where | Default | Effect |
|-----------|-------|---------|--------|
| `chunk_duration` | `caption_video` (app sets `CHUNK_DURATION=1`) | 2 s (app: 1 s) | Caption window = the game's time resolution |
| `chunk_overlap_duration` | `caption_video` (app sets `CHUNK_OVERLAP=0`) | 1 s (app: 0 s) | Overlap so a move on a boundary stays in one caption |
| `num_frames_per_chunk` | `caption_video` | 5 | Frames the VLM sees per window (hard cap 5) |
| `temperature` | `first_performer` | 0.0 | Kept at 0 for repeatable LLM answers |
| `cv_pipeline_prompt` | `caption_video` | `"person"` | What the CV pipeline detects and tracks |

---

## 🖥️ VSS Backend

The app talks to a **VSS** backend over HTTP (`caption_video` → `/generate_vlm_captions`, `first_performer` → the text NIM). This section documents how that backend is deployed (`deploy/docker/local_deployment_single_gpu`).

### Version & shape

- **VSS engine 2.4.1** (`nvcr.io/nvidia/blueprint/vss-engine:2.4.1`), docker-compose.
- Backend API on **`:8100`**, web UI on **`:9100`**.
- One GPU container — **`via-server`**, pinned to **GPU 1** (`NVIDIA_VISIBLE_DEVICES=1`) — plus CPU-only infra containers: **neo4j** (graph DB), **milvus** (vector DB), **arango**, **minio**, **elasticsearch**.
- The VLM and the CA-RAG models run as **separate external NIM containers**, reached over `host.docker.internal`. They are *not* part of the compose file and must be deployed independently (more info on https://build.nvidia.com/nvidia/cosmos3-nano-reasoner/deploy, but use NIM_CACHE_PATH, the right free port coherent with config.yaml and the right gpu written as '"device=0"').

### Models on the text / CA-RAG side (`config.yaml`)

| Model | Port | Use |
|-------|------|-----|
| **nvidia/nemotron-3.5-lightning** (30b-a3b) | `:8007` | `chat_llm` + `summarization_llm` + `notification_llm` — graph ingestion & retrieval, caption→summary aggregation, alert notifications |
| **nvidia/llama-nemotron-embed-vl-1b-v2** | `:8006` | Embeddings for the neo4j graph DB and milvus vector DB |
| **nvidia/llama-nemotron-rerank-vl-1b-v2** | `:8005` | Reranker for retrieval |

These power **CA-RAG**: captions are ingested into the graph/vector DBs, then chat/summary queries retrieve + rerank + reason over them.

### Vision model (`.env`)

`VLM_MODEL_TO_USE=openai-compat` → **`nvidia/cosmos3-nano-reasoner`** (served by the `cosmos3-reasoner` NIM) at **`:38011`**. Its job is **dense captioning** of each video chunk: VSS samples **≤5 frames per chunk** (the model caps a prompt at 5 images) and asks the VLM to describe them. This is the model that receives the Set-of-Marks overlaid frames.

### CV detection + tracking (`DISABLE_CV_PIPELINE=false`)

Running inside `via-server` on GPU 1:

- **GroundingDINO** detector (auto-downloaded from NGC) finds people.
- **NvDCF tracker** (+ **ReID**, + **SAM2** engines built at first start) assigns each person a **stable tracking ID**.
- The IDs are **burned onto the frames** (Set-of-Marks) *before* the VLM sees them, so the VLM gets one view per frame: the numbered one.
- Detector cadence = `GDINO_INFERENCE_INTERVAL` (default 1 = detect every other frame); the tracker labels **every** frame in between, so the overlay is full-frame-rate.

Setup + troubleshooting detail: see `deploy/docker/local_deployment_single_gpu/CV_TRACKING_SETUP.md`.

### Scheme 1 — request pipeline

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

### Scheme 2 — model placement across the two GPUs

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

The cosmos NIM is memory-capped with `NIM_GPU_MEMORY_UTILIZATION=0.50` specifically to leave headroom on GPU 1 for the CV pipeline's TensorRT engine builds and inference. This might be optimized to give more free RAM to the cache of cosmos3 at its startup.

---

## 🗺️ Roadmap

**Next up:**

- 🔊 Audio instructions and synchronization
- ✋ Raise-hand player registration
- 🏆 Tournament rounds / top-80% qualification
- ⚡ Real-time (per-frame) movement validation
- 👤 Face recognition to identify and name the winner automatically

**Beyond the game:** the same "watch people, understand actions, check the order, determine the outcome" loop applies to **industrial procedures, workplace safety, sports training, education, and rehabilitation** — turning computer vision into an **AI game master** that watches everyone at once and calls the winner.

**Watch. Move. Get recognized. Climb the leaderboard. 🏆**
