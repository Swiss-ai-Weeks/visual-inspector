#!/usr/bin/env bash
# run.sh — start MoveMatch
set -euo pipefail
cd "$(dirname "$0")"

# RTVLM = VLM caption endpoint; LLM = text NIM reasoner (see vss.py).
export VSS_RTVLM_URL="${VSS_RTVLM_URL:-http://127.0.0.1:8100}"
export VSS_LLM_URL="${VSS_LLM_URL:-http://127.0.0.1:38011}"
export KEEP_UPLOADS="${KEEP_UPLOADS:-0}"

# YOLO pipeline ONNX models (see yolo.py). Absolute so gunicorn workers find
# them regardless of CWD; default to the repo-root models dir.
export YOLOX_ONNX="${YOLOX_ONNX:-/home/nvidia/Documents/vss2/models/yolox.onnx}"
export YOLO_POSE_ONNX="${YOLO_POSE_ONNX:-/home/nvidia/Documents/vss2/models/yolov8n-pose.onnx}"

VENV="${MOVEMATCH_VENV:-/home/nvidia/Documents/.venv}"
exec "$VENV/bin/gunicorn" --timeout 1200 --graceful-timeout 60 --workers 2 --bind 127.0.0.1:12500 app:app
