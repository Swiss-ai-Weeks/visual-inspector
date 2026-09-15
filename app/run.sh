#!/usr/bin/env bash
# run.sh — start the Cosmos video-QA web app
set -euo pipefail
cd "$(dirname "$0")"

export COSMOS_NIM_URL="${COSMOS_NIM_URL:-http://127.0.0.1:8000}"
export COSMOS_NIM_MODEL="${COSMOS_NIM_MODEL:-nvidia/cosmos-reason2-2b}"
export VSS_AGENT_URL=http://127.0.0.1:8000
export VSS_HOST_UPLOAD_DIR=/home/nvidia/Documents/app/uploads
export VSS_AGENT_UPLOAD_DIR=/data/uploads
export KEEP_UPLOADS=1

exec gunicorn --timeout 1200 --graceful-timeout 60 --workers 2 --bind 127.0.0.1:5000 app:app