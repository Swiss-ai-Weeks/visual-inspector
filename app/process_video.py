"""VSS Agent client — uploads videos to VST, then queries the agent."""

import json
import mimetypes
import os
import re
import uuid
from pathlib import Path
from typing import Any

import requests

AGENT_URL = os.environ.get("VSS_AGENT_URL", "http://127.0.0.1:8000").rstrip("/")
VST_URL = os.environ.get("VSS_VST_URL", "http://127.0.0.1:30000").rstrip("/")
UPLOAD_TIMESTAMP = os.environ.get("VSS_UPLOAD_TS", "2025-01-01T00:00:00")

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
# Sensor housekeeping
# --------------------------------------------------------------------------

def list_sensors() -> list[dict]:
    """Every sensor VST currently knows about.

    The agent's /api/v1/videos is POST-only (a GET returns 405), so the
    registry has to be read from VST directly.
    """
    try:
        listing = requests.get(
            f"{VST_URL}/vst/api/v1/sensor/list", timeout=(10, 60)
        )
        listing.raise_for_status()
        payload = listing.json()
    except (requests.RequestException, ValueError):
        return []
    return [item for item in payload if isinstance(item, dict)] \
        if isinstance(payload, list) else []


def _matching_sensors(stem: str) -> list[dict]:
    return [s for s in list_sensors() if stem in (s.get("name") or "")]


def purge_videos(filename: str) -> int:
    """Delete any sensor already registered under `filename`.

    RTVI-CV rejects a stream whose camera id already exists
    ("STREAM_ADD_FAIL, Duplicate Camera id") and RTVI-Embed rejects an asset id
    it already holds ("AssetAlreadyExists"), either of which silently costs us
    the CV metadata for the run. Clearing stale sensors first avoids that.

    A successful DELETE is not proof the sensor is gone, so the registry is
    re-read afterwards and survivors are reported: VST hands the same stream id
    back for a file of the same name, and its asset comes back with it.
    """
    stem = Path(filename).stem  # VST stores the name without the extension

    removed = 0
    for sensor in _matching_sensors(stem):
        sensor_id = sensor.get("sensorId")
        if sensor_id and delete_video(sensor_id):
            removed += 1

    survivors = _matching_sensors(stem)
    if survivors:
        ids = ", ".join(s.get("sensorId", "?") for s in survivors)
        print(
            f"[warn] {len(survivors)} sensor(s) named like '{stem}' survived the "
            f"purge ({ids}); /complete may fail with a duplicate id."
        )
    return removed


def drop_asset(asset_id: str) -> None:
    """Best-effort removal of everything the agent still holds under `asset_id`.

    The VSS asset store is keyed by the VST stream id, and deleting the sensor
    does not always drop it. The agent does not document an asset route, so the
    single-id DELETE routes it advertises are swept; the ones that do not know
    the id answer 404, which is harmless.
    """
    try:
        spec = requests.get(f"{AGENT_URL}/openapi.json", timeout=15).json()
    except (requests.RequestException, ValueError):
        return

    for route, ops in spec.get("paths", {}).items():
        if "delete" not in ops or route.count("{") != 1:
            continue
        if not re.search(r"video|file|asset|stream", route, re.I):
            continue
        url = AGENT_URL + re.sub(r"\{[^}]+\}", asset_id, route)
        try:
            requests.delete(url, timeout=(10, 120))
        except requests.RequestException:
            pass


def delete_video(sensor_id: str) -> bool:
    """Remove a sensor from VSS. Best-effort — never raises."""
    try:
        response = requests.delete(
            f"{AGENT_URL}/api/v1/videos/{sensor_id}", timeout=(10, 120)
        )
        return response.ok
    except requests.RequestException:
        return False


# --------------------------------------------------------------------------
# Video upload (agent -> VST -> complete)
# --------------------------------------------------------------------------

_COLLISION = re.compile(
    r"AssetAlreadyExists|Duplicate Camera id|already exists", re.I
)


def _complete(sensor_id: str, payload: dict) -> requests.Response | None:
    """POST /complete, returning the response even when it is an error.

    None means the call never got an answer (timeout, connection refused);
    /complete is slow enough that this is a real outcome, and it must not cost
    the caller the sensor id it already has.
    """
    try:
        return requests.post(
            f"{AGENT_URL}/api/v1/videos/{sensor_id}/complete",
            json=payload,
            timeout=(15, 900),
        )
    except requests.RequestException as exc:
        print(f"[warn] /complete did not answer: {exc}")
        return None


def upload_video(video_path: str) -> tuple[str, bool]:
    """Register a local video with VSS.

    Returns (sensor_id, cv_ready). `cv_ready` reports whether the /complete
    fan-out to RTVI-CV / RTVI-Embed succeeded — video_understanding works
    without it, but the CV tracking metadata does not exist unless it did.
    """
    src = Path(video_path).resolve()
    if not src.is_file():
        raise FileNotFoundError(src)

    filename = src.name
    mime = mimetypes.guess_type(filename)[0] or "video/mp4"

    purge_videos(filename)

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
                "metadata": json.dumps({"timestamp": UPLOAD_TIMESTAMP}),
            },
            timeout=(15, 900),
        )
    sent.raise_for_status()

    payload = sent.json()
    sensor_id = payload.get("sensorId")
    if not sensor_id:
        raise RuntimeError(f"Upload returned no sensorId: {sent.text}")

    # 3. Tell the agent the upload finished — this is what runs RTVI-CV
    #    (detection + tracking) and RTVI-Embed over the clip.
    payload["filename"] = filename
    done = _complete(sensor_id, payload)

    # A stale registration under this id fails the fan-out before any CV runs.
    # Clearing the asset and retrying once is cheap; re-uploading is not, and
    # would need a new filename to be of any use.
    if done is not None and not done.ok and _COLLISION.search(done.text or ""):
        print(
            f"[warn] /complete: id {sensor_id} is still registered from an "
            "earlier run; clearing it and retrying"
        )
        drop_asset(sensor_id)
        done = _complete(sensor_id, payload)

    if done is None or not done.ok:
        detail = "no response" if done is None else f"{done.status_code} {done.text[:500]}"
        print(
            f"[warn] /complete failed ({detail}); continuing with "
            f"sensor_id={sensor_id}. Tracking and search are unavailable "
            "for this video."
        )

    return sensor_id, bool(done is not None and done.ok)


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------

def ask_agent(sensor_id: str, prompt: str, filename: str = "") -> str:
    """Ask the agent a question about an already-uploaded sensor."""
    endpoint = discover_endpoint()

    origin = f" (uploaded file: {filename})" if filename else ""
    composed = (
        f"Analyse the video with sensor_id `{sensor_id}`{origin}.\n\n"
        f"User question:\n{prompt}\n\n"
        "Use the video_understanding tool with that exact sensor_id and "
        "omit start/end timestamps to cover the whole video. Answer only "
        "from what is visible, and include a timestamp for each observation."
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


def query_vss_agent_video(video_path: str, prompt: str) -> str:
    """Upload a video to VSS, then ask the agent about it."""
    sensor_id, _ = upload_video(video_path)
    return ask_agent(sensor_id, prompt, Path(video_path).name)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("usage: python3 process_video.py <video_path> <prompt>")
        raise SystemExit(1)

    print(query_vss_agent_video(sys.argv[1], sys.argv[2]))
