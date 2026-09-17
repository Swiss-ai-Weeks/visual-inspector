"""Low-level VSS API client: video upload, /summarize, /chat."""

from __future__ import annotations

import json
import mimetypes
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

AGENT_URL = os.environ.get("VSS_AGENT_URL", "http://127.0.0.1:8100").rstrip("/")

_session = requests.Session()


def upload_video(video_path: str) -> str:
    """Upload a video to VSS via VST chunked protocol. Returns sensorId."""
    src = Path(video_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(src)

    filename = src.name
    mime = mimetypes.guess_type(filename)[0] or "video/mp4"

    init = _session.post(
        f"{AGENT_URL}/api/v1/videos",
        json={"filename": filename},
        timeout=(15, 60),
    )
    init.raise_for_status()
    upload_url = init.json().get("url")
    if not upload_url:
        raise RuntimeError(f"No upload url returned: {init.text}")

    identifier = uuid.uuid4().hex
    with src.open("rb") as handle:
        sent = _session.post(
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
                    "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                }),
            },
            timeout=(15, 900),
        )
    sent.raise_for_status()

    payload = sent.json()
    sensor_id = payload.get("sensorId")
    if not sensor_id:
        raise RuntimeError(f"Upload returned no sensorId: {sent.text}")

    payload["filename"] = filename
    try:
        done = _session.post(
            f"{AGENT_URL}/api/v1/videos/{sensor_id}/complete",
            json=payload,
            timeout=(15, 900),
        )
        done.raise_for_status()
    except requests.RequestException:
        pass

    return sensor_id


def summarize_video(
    sensor_id: str,
    caption_prompt: str,
    *,
    chunk_duration: int = 1,
    num_frames_per_chunk: int = 8,
    enable_cv_metadata: bool = True,
    cv_pipeline_prompt: str = "person",
    temperature: float = 0.05,
    caption_summarization_prompt: str | None = None,
    summary_aggregation_prompt: str | None = None,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Call VSS /summarize with CV pipeline + SOM enabled."""
    payload: dict[str, Any] = {
        "sensor_id": sensor_id,
        "prompt": caption_prompt,
        "chunk_duration": chunk_duration,
        "num_frames_per_chunk": num_frames_per_chunk,
        "enable_cv_metadata": enable_cv_metadata,
        "cv_pipeline_prompt": cv_pipeline_prompt,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if caption_summarization_prompt:
        payload["caption_summarization_prompt"] = caption_summarization_prompt
    if summary_aggregation_prompt:
        payload["summary_aggregation_prompt"] = summary_aggregation_prompt

    resp = _session.post(
        f"{AGENT_URL}/v1/summarize",
        json=payload,
        timeout=(15, 1800),
    )
    resp.raise_for_status()
    return resp.json()


def chat_query(sensor_id: str, question: str) -> str:
    """Ask a follow-up question via VSS /chat/completions (CA-RAG)."""
    resp = _session.post(
        f"{AGENT_URL}/v1/chat/completions",
        json={
            "model": "vss-agent",
            "messages": [{"role": "user", "content": question}],
            "sensor_id": sensor_id,
            "stream": False,
        },
        timeout=(15, 600),
    )
    resp.raise_for_status()
    data = resp.json()
    return _extract_text(data)


def query_agent(prompt: str) -> str:
    """Send a free-form prompt to the VSS agent /generate endpoint."""
    resp = _session.post(
        f"{AGENT_URL}/generate",
        json={"input_message": prompt},
        timeout=(15, 900),
    )
    resp.raise_for_status()
    return _extract_text(resp.json())


def delete_video(sensor_id: str) -> None:
    """Remove a video from VSS."""
    try:
        _session.delete(
            f"{AGENT_URL}/api/v1/videos/{sensor_id}",
            timeout=(10, 120),
        )
    except requests.RequestException:
        pass


def _extract_text(data: Any) -> str:
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
        for key in ("value", "output", "answer", "result", "response", "content"):
            if key in data and data[key] is not None:
                return _extract_text(data[key])
    return json.dumps(data, indent=2, ensure_ascii=False)
