#!/usr/bin/env bash
# run.sh — start MoveMatch with gunicorn (Linux/macOS).
# On Windows, or without gunicorn, use:  python app/app.py
set -euo pipefail
cd "$(dirname "$0")"

# RTVLM = VLM caption endpoint; LLM = text NIM reasoner (see vss.py).
export VSS_RTVLM_URL="${VSS_RTVLM_URL:-http://127.0.0.1:8100}"
export VSS_LLM_URL="${VSS_LLM_URL:-http://127.0.0.1:38011}"
export KEEP_UPLOADS="${KEEP_UPLOADS:-0}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5000}"
WEB_WORKERS="${WEB_WORKERS:-2}"

# Interpreter: $MOVEMATCH_VENV, else the repo's .venv, else python3 on PATH.
if [[ -n "${MOVEMATCH_VENV:-}" ]]; then
  PY="$MOVEMATCH_VENV/bin/python"
elif [[ -x ../.venv/bin/python ]]; then
  PY="../.venv/bin/python"
else
  PY="$(command -v python3 || command -v python)"
fi

if ! "$PY" -c 'import sys; assert sys.version_info >= (3, 11); import flask, gunicorn' 2>/dev/null; then
  echo "run.sh: $PY is not Python >= 3.11 with the MoveMatch dependencies installed." >&2
  echo "Set up the project virtualenv once, from the repo root:" >&2
  echo "  python3.12 -m venv .venv && .venv/bin/pip install .   (or: uv venv -p 3.12 .venv && uv pip install .)" >&2
  echo "or point MOVEMATCH_VENV at an existing virtualenv." >&2
  exit 1
fi

exec "$PY" -m gunicorn --timeout 1200 --graceful-timeout 60 \
  --workers "$WEB_WORKERS" --bind "$HOST:$PORT" app:app
