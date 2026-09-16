#!/usr/bin/env bash
# run.sh — start MoveMatch
set -euo pipefail
cd "$(dirname "$0")"

export VSS_AGENT_URL="${VSS_AGENT_URL:-http://127.0.0.1:8000}"
export VSS_HOST_UPLOAD_DIR="${VSS_HOST_UPLOAD_DIR:-/home/nvidia/Documents/app/uploads}"
export VSS_AGENT_UPLOAD_DIR="${VSS_AGENT_UPLOAD_DIR:-/data/uploads}"
export KEEP_UPLOADS="${KEEP_UPLOADS:-0}"

exec gunicorn --timeout 1200 --graceful-timeout 60 --workers 2 --bind 127.0.0.1:5000 app:app
