"""VSS Agent client — uploads videos to VST, then queries the agent."""

import json
import mimetypes
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

AGENT_URL = os.environ.get("VSS_AGENT_URL", "http://127.0.0.1:8000").rstrip("/")

_CANDIDATES = ("/chat", "/v1/chat", "/generate", "/v1/generate",
               "/v1/chat/completions", "/chat/completions")

_cached_endpoint: str | None = None


# --------------------------------------------------------------------------
# Agent route discovery
# --------------------------------------------------------------------------

def discover_endpoint() -> str:
    """Return the first supported POST chat/generate route on the agent."""
    global _cached_endpoint
    if _cached_endpoint:
        return _cached_endpoint

    spec = requests.get(f"{AGENT_URL}/openapi.json", timeout=15)
    spec.raise_for_status()
    paths = spec.json().get("paths", {})

    for candidate in _CANDIDATES:
        if candidate in paths and "post" in paths[candidate]:
            _cached_endpoint = candidate
            return candidate

    raise RuntimeError(f"No chat route found. Available: {sorted(paths)}")


def _build_payload(endpoint: str, prompt: str) -> dict[str, Any]:
    """Each route family expects a different body shape."""
    if "completions" in endpoint:
        return {
            "model": "vss-agent",
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
    if "chat" in endpoint:
        return {"messages": [{"role": "user", "content": prompt}]}
    return {"input_message": prompt}


def _extract_text(data: Any) -> str:
    """Pull the assistant text out of whichever envelope came back."""
    if isinstance(data, str):
        return data

    if isinstance(data, dict):
        choices = data.get("choices")
        if choices:
            first = choices[0]
            msg = first.get("message") or first.get("delta") or {}
            if msg.get("content"):
                return msg["content"]
            if first.get("text"):
                return first["text"]

        for key in ("value", "output", "answer",
                    "result", "response", "content"):
            if data.get(key) is not None:
                return _extract_text(data[key])

    return json.dumps(data, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------
# Video upload (agent -> VST -> complete)
# --------------------------------------------------------------------------

def upload_video(video_path: str) -> str:
    """Register a local video with VSS. Returns its sensorId."""
    src = Path(video_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(src)

    filename = src.name
    mime = mimetypes.guess_type(filename)[0] or "video/mp4"

    # 1. Ask the agent for the chunked-upload URL.
    init = requests.post(
        f"{AGENT_URL}/api/v1/videos",
        json={"filename": filename},
        timeout=(15, 60),
    )
    init.raise_for_status()
    upload_url = init.json().get("url")
    if not upload_url:
        raise RuntimeError(f"No upload url returned: {init.text}")

    # 2. Single-chunk POST to VST (nvstreamer protocol).
    identifier = uuid.uuid4().hex
    with src.open("rb") as handle:
        sent = requests.post(
            upload_url,
            headers={
                "nvstreamer-chunk-number": "1",
                "nvstreamer-total-chunks": "1",
                "nvstreamer-is-last-chunk": "true",
                "nvstreamer-identifier": identifier,
                "nvstreamer-file-name": filename,
            },
            files={"mediaFile": (filename, handle, mime)},
            data={
                "filename": filename,
                "metadata": json.dumps({
                    "timestamp": os.environ.get(
                        "VSS_UPLOAD_TS",
                        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                    )
                }),
            },
            timeout=(15, 900),
        )
    sent.raise_for_status()

    payload = sent.json()
    sensor_id = payload.get("sensorId")
    if not sensor_id:
        raise RuntimeError(f"Upload returned no sensorId: {sent.text}")

    # 3. Tell the agent the upload finished (enables search/embeddings).
    #    Non-fatal: video_understanding works without this fan-out, which
    #    502s when RTVI-CV / RTVI-embed are not part of the profile.
    payload["filename"] = filename
    try:
        done = requests.post(
            f"{AGENT_URL}/api/v1/videos/{sensor_id}/complete",
            json=payload,
            timeout=(15, 900),
        )
        done.raise_for_status()
    except requests.RequestException as exc:
        print(
            f"[warn] /complete failed ({exc}); continuing with "
            f"sensor_id={sensor_id}. Search/embeddings may be "
            "unavailable for this video."
        )

    return sensor_id


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def query_vss_agent_video(video_path: str, actions: list[str]) -> str:
    """Upload a video to VSS, detect per-person actions, return raw agent text.

    *actions* is the vocabulary of recognisable moves (e.g. ["clap", "wave"]).
    The prompt asks the agent to return structured JSON so the caller can do
    deterministic sequence matching in Python.
    """
    endpoint = discover_endpoint()
    sensor_id = upload_video(video_path)

    action_list = "\n".join(f"  - {a}" for a in actions)

    composed = (
        f"Analyse the video with sensor_id `{sensor_id}` "
        f"(uploaded file: {Path(video_path).name}).\n\n"
        "Use the video_understanding tool with that exact sensor_id. "
        "Set chunk_duration_secs=2 for fine temporal granularity.\n\n"
        "Task:\n"
        "1. Detect every person visible in the video. "
        "Number them left to right (1 = leftmost).\n"
        "2. For each person, list the actions they performed in "
        "chronological order. Only use action names from the list below "
        "(exactly as written):\n"
        f"{action_list}\n\n"
        "Return ONLY valid JSON in this exact format — no other text:\n"
        "{\n"
        '  "players": [\n'
        '    {"id": 1, "actions": ["action name", "action name"]},\n'
        '    {"id": 2, "actions": ["action name"]}\n'
        "  ]\n"
        "}\n\n"
        "If a person performed none of the listed actions, use an empty list."
    )

    response = requests.post(
        f"{AGENT_URL}{endpoint}",
        json=_build_payload(endpoint, composed),
        timeout=(15, 900),
    )
    response.raise_for_status()

    try:
        return _extract_text(response.json())
    except ValueError:
        return response.text


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print('usage: python3 process_video.py <video_path> \'["clap","wave"]\'')
        raise SystemExit(1)

    print(query_vss_agent_video(sys.argv[1], json.loads(sys.argv[2])))