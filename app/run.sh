#!/usr/bin/env bash
# run.sh — start MoveMatch
set -euo pipefail
cd "$(dirname "$0")"

# RTVLM = VLM caption endpoint; LLM = text NIM reasoner (see vss.py).
export VSS_RTVLM_URL="${VSS_RTVLM_URL:-http://127.0.0.1:8100}"
export VSS_LLM_URL="${VSS_LLM_URL:-http://127.0.0.1:38011}"
export KEEP_UPLOADS="${KEEP_UPLOADS:-0}"

VENV="${MOVEMATCH_VENV:-/home/nvidia/Documents/.venv}"
# One worker + threads: the async job registry (app.py JOBS) lives in process
# memory, so every /jobs poll must land on the same worker. Threads still give
# us concurrent request handling while a job runs in a background thread.
exec "$VENV/bin/gunicorn" --timeout 1200 --graceful-timeout 60 --workers 1 --threads 8 --bind 127.0.0.1:5000 app:app
