# VSS 2.4.1 — Deployment Guide

How to stand up this Video Search & Summarization (VSS) deployment from nothing: clone the
code, build the custom minio image, set API keys, launch the four NVIDIA NIM models, and bring
up the VSS engine + databases.

> Target host used here: 2× NVIDIA H100 NVL (95 GB each). GPU 0 hosts the three text NIMs;
> GPU 1 hosts the vision model + the VSS engine (with its CV pipeline).

---

## 0. Prerequisites

- A multi-GPU Linux host with **Docker** + the **NVIDIA Container Toolkit** (`--runtime=nvidia`
  / `--gpus` working).
- An **NGC / build.nvidia.com** account (to pull the NIM images and generate an API key).
- A **Hugging Face** account + token (some model weights are pulled from HF).

---

## 1. Get the code

Clone the VSS fork/branch into `~/Documents/vss2/`:

```bash
cd ~/Documents/vss2
git clone -b test/vss2 https://github.com/mickaelbressieux/video-search-and-summarization.git
```

The deployment directory is:

```
~/Documents/vss2/video-search-and-summarization/deploy/docker/local_deployment_single_gpu
```

That folder holds `compose.yaml`, `config.yaml`, `.env`, and `guardrails/`.

---

## 2. Build the custom minio image

`compose.yaml` references `image: minio-local:source`, so that image must exist locally before
you bring the stack up. Clone the minio fork/branch and build it with that exact tag:

```bash
cd ~/Documents/vss2
git clone -b team/vss2 https://github.com/mickaelbressieux/minio.git
cd minio
# Build per the repo's instructions, then tag it as the image the compose expects:
docker build -t minio-local:source .
# (If the repo ships a Makefile target such as `make docker`, run that instead, then:
#  docker tag <built-image> minio-local:source )
```

Verify:

```bash
docker images | grep minio-local
```

---

## 3. Set the API keys

Edit the deployment `.env`:

```
~/Documents/vss2/video-search-and-summarization/deploy/docker/local_deployment_single_gpu/.env
```

Set your two secrets (keep the surrounding lines as-is):

```bash
export NGC_API_KEY=nvapi-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX   # from build.nvidia.com
export HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX                    # from huggingface.co
```

Notes:
- `NVIDIA_API_KEY="noapikeyset"` in `.env` is an intentional placeholder — the VLM endpoint is
  local, so no real key is needed there. Leave it.
- Generate the NGC key at **build.nvidia.com → your profile → "Get API Key"**.
- Treat `.env` as a secret file — do not commit it.

---

## 4. Log in to the NGC container registry

The NIM images live on `nvcr.io`. Authenticate with your NGC key (username is the literal
`$oauthtoken`):

```bash
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

All four NIMs share one weights cache on the host so re-launches are fast:

```bash
mkdir -p ~/.cache/nim
```

---

## 5. Launch the four NVIDIA NIM models

Find each model on **build.nvidia.com** (search the name) — the model page links the
`nvcr.io/nim/nvidia/...` image and shows the pull command. This deployment uses:

| Model (build.nvidia.com) | Role in VSS | Image (`nvcr.io/nim/nvidia/…`) | GPU | Host port |
|---|---|---|---|---|
| **nemotron-3.5-lightning** | CA-RAG LLM: chat + summarization + notification | `nemotron-3.5-lightning-30b-a3b:latest` | 0 | 8007 |
| **llama-nemotron-embed-vl-1b-v2** | embeddings (graph + vector DB) | `llama-nemotron-embed-vl-1b-v2:1.12.0` | 0 | 8006 |
| **llama-nemotron-rerank-vl-1b-v2** | reranker (retrieval) | `llama-nemotron-rerank-vl-1b-v2:latest` | 0 | 8005 |
| **cosmos3-reasoner** | VLM: dense captioning of video frames | `cosmos3-reasoner:latest` | 1 | 38011 |

> Ports are not arbitrary: `config.yaml` expects the LLM/embed/rerank on **8007 / 8006 / 8005**,
> and `.env` points the VLM at **38011**. Keep them.

Make sure `NGC_API_KEY` is exported in your shell first (`source .env` or export it), then:

### 5a. LLM — nemotron-3.5-lightning (GPU 0, :8007)

```bash
docker run -d --name nemotron-llm --runtime=nvidia --gpus '"device=0"' \
  --shm-size=16g -e NGC_API_KEY \
  -v ~/.cache/nim:/opt/nim/.cache \
  -p 8007:8000 \
  nvcr.io/nim/nvidia/nemotron-3.5-lightning-30b-a3b:latest
```

### 5b. Embeddings — llama-nemotron-embed-vl-1b-v2 (GPU 0, :8006)

```bash
docker run -d --name nemotron-embed --runtime=nvidia --gpus '"device=0"' \
  --shm-size=16g -e NGC_API_KEY \
  -v ~/.cache/nim:/opt/nim/.cache \
  -p 8006:8000 \
  nvcr.io/nim/nvidia/llama-nemotron-embed-vl-1b-v2:1.12.0
```

### 5c. Reranker — llama-nemotron-rerank-vl-1b-v2 (GPU 0, :8005)

```bash
docker run -d --name nemotron-rerank --runtime=nvidia --gpus '"device=0"' \
  --shm-size=16g -e NGC_API_KEY \
  -v ~/.cache/nim:/opt/nim/.cache \
  -p 8005:8000 \
  nvcr.io/nim/nvidia/llama-nemotron-rerank-vl-1b-v2:latest
```

### 5d. VLM — cosmos3-reasoner (GPU 1, :38011)

The cosmos NIM gets extra flags so it does **not** eat all of GPU 1 — the VSS CV pipeline needs
headroom on that GPU for its TensorRT engine builds (see `CV_TRACKING_SETUP.md`).

```bash
docker run -d --name cosmos3-reasoner --runtime=nvidia --gpus '"device=1"' \
  --shm-size=32g -e NGC_API_KEY \
  -e NIM_GPU_MEMORY_UTILIZATION=0.50 \
  -e NIM_MAX_MODEL_LEN=32768 \
  -v ~/.cache/nim:/opt/nim/.cache \
  -p 38011:8000 \
  nvcr.io/nim/nvidia/cosmos3-reasoner:latest
```

The first launch of each NIM downloads weights (can take several minutes to tens of minutes).
Watch with `docker logs -f <name>` until each reports it is serving.

---

## 6. Bring up the VSS engine + databases

From the deployment directory (compose reads `.env` automatically):

```bash
cd ~/Documents/vss2/video-search-and-summarization/deploy/docker/local_deployment_single_gpu
docker compose up -d
```

This starts:
- **via-server** — the VSS engine (pinned to GPU 1 via `NVIDIA_VISIBLE_DEVICES=1`), which does
  video decode + the CV detection/tracking pipeline and talks to the NIMs above.
- **neo4j** (graph DB), **milvus** (vector DB), **arango**, **minio** (the image from step 2),
  **elasticsearch** — all CPU-only.

For CV detection + tracking (Set-of-Marks) specifics, see
`../video-search-and-summarization/deploy/docker/local_deployment_single_gpu/CV_TRACKING_SETUP.md`.

---

## 7. Verify

```bash
# NIMs up:
curl -s http://localhost:8007/v1/models   # LLM
curl -s http://localhost:8006/v1/models   # embeddings
curl -s http://localhost:8005/v1/models   # reranker
curl -s http://localhost:38011/v1/models  # cosmos VLM

# VSS stack up:
docker compose ps                          # all services healthy
curl -s http://localhost:8100/health/ready # VSS backend ready
```

Then open the UI at **`http://<host>:9100`** (backend API on **`:8100`**).

---

## GPU placement recap

```
GPU 0  (~92 GB used)                      GPU 1  (~60 GB used, ~35 GB free)
  nemotron-3.5-lightning  :8007  ~77 GB     cosmos3 VLM            :38011  ~49 GB
  embed-vl-1b-v2          :8006  ~8 GB      via-server decode + CV pipeline ~10 GB
  rerank-vl-1b-v2         :8005  ~6 GB
```

CPU-only containers: neo4j · milvus · arango · minio · elasticsearch.
See `technical_explanation.md` (VSS Backend Architecture) for the full pipeline diagram.
