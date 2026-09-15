"""Query the VSS Agent (port 8000) — the agentic layer above RT-VLM."""

import json
import os
from pathlib import Path
from typing import Any

import requests

AGENT_URL = os.environ.get("VSS_AGENT_URL", "http://127.0.0.1:8000").rstrip("/")
VST_URL = os.environ.get("VSS_VST_URL", "http://127.0.0.1:30000").rstrip("/")

# Preference order: richest schema first, OpenAI-compat last.
_CANDIDATES = ("/chat", "/v1/chat", "/generate", "/v1/generate",
               "/v1/chat/completions", "/chat/completions")


def discover_endpoint() -> str:
    """Return the first supported POST path on the running agent."""
    spec = requests.get(f"{AGENT_URL}/openapi.json", timeout=15)
    spec.raise_for_status()
    paths = spec.json().get("paths", {})

    available = [p for p in _CANDIDATES if p in paths and "post" in paths[p]]
    if not available:
        raise RuntimeError(
            f"No known chat route found. Available: {sorted(paths)}"
        )
    return available[0]


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
    # /generate uses the workflow's own schema
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
            if key in data and data[key] is not None:
                return _extract_text(data[key])

    return json.dumps(data, indent=2, ensure_ascii=False)


def query_vss_agent(prompt: str, endpoint: str | None = None) -> str:
    """Ask the VSS Agent a natural-language question."""
    endpoint = endpoint or discover_endpoint()

    response = requests.post(
        f"{AGENT_URL}{endpoint}",
        json=_build_payload(endpoint, prompt),
        timeout=(15, 900),  # agents chain many tool calls; be patient
    )
    response.raise_for_status()

    try:
        return _extract_text(response.json())
    except ValueError:
        return response.text


def query_vss_agent_stream(prompt: str) -> str:
    """Streaming variant — shows intermediate steps as they arrive."""
    endpoint = "/chat/stream"
    payload = _build_payload("/chat", prompt)

    chunks: list[str] = []
    with requests.post(
        f"{AGENT_URL}{endpoint}",
        json=payload,
        stream=True,
        timeout=(15, 900),
    ) as response:
        response.raise_for_status()

        for raw in response.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            data = raw[5:].strip()
            if data == "[DONE]":
                break
            try:
                piece = _extract_text(json.loads(data))
            except json.JSONDecodeError:
                piece = data
            print(piece, end="", flush=True)
            chunks.append(piece)

    print()
    return "".join(chunks)


def upload_to_vst(video_path: str) -> str:
    """Optional: push a local file into VST so the agent can reach it."""
    path = Path(video_path)
    with path.open("rb") as handle:
        uploaded = requests.post(
            f"{VST_URL}/api/v1/files",
            files={"file": (path.name, handle, "video/mp4")},
            timeout=(15, 600),
        )
    uploaded.raise_for_status()
    return uploaded.json().get("file_id", path.name)


if __name__ == "__main__":
    route = discover_endpoint()
    print(f"[agent route] {route}\n")

    # Always start here: the agent reports its real sensor names.
    print(query_vss_agent("What sensors are available?"))