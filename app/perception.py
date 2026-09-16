"""Perception Agent: calls VSS /summarize with SOM overlay to extract
structured per-player action events from a video clip."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from models import ACTION_VOCAB, ActionEvent
from vss_client import summarize_video

log = logging.getLogger(__name__)

CAPTION_PROMPT = (
    "You see a short video chunk. Each person has a numeric ID overlaid "
    "on a colored mask. For EACH visible person ID, report the single "
    "dominant body action from this closed vocabulary ONLY:\n"
    + json.dumps(ACTION_VOCAB)
    + "\n\n"
    "Return strict JSON, no prose:\n"
    '{"events":[{"player_id":<int>,"action":"<vocab>","confidence":0.0-1.0}]}\n'
    'If a person is partially visible or the action is ambiguous, use "idle".'
)

MERGE_PROMPT = (
    "Concatenate all chunk JSON objects into one array ordered by timestamp. "
    "Do not summarize."
)


def extract_events(sensor_id: str) -> list[ActionEvent]:
    """Run the Perception Agent against a VSS-uploaded video.

    Returns a flat list of ActionEvents with timestamps inferred from
    chunk ordering (1s chunks).
    """
    raw = summarize_video(
        sensor_id,
        caption_prompt=CAPTION_PROMPT,
        chunk_duration=1,
        num_frames_per_chunk=8,
        enable_cv_metadata=True,
        cv_pipeline_prompt="person",
        temperature=0.05,
        caption_summarization_prompt=MERGE_PROMPT,
        summary_aggregation_prompt=MERGE_PROMPT,
        max_tokens=2048,
    )

    content = _get_content(raw)
    log.info("VSS raw content: %s", content[:500])
    return _parse_events(content)


def extract_events_via_agent(sensor_id: str, video_name: str, actions: list[str]) -> list[ActionEvent]:
    """Fallback: use the VSS agent /generate endpoint when /summarize
    doesn't support the CV pipeline flags yet."""
    from vss_client import query_agent

    action_list = json.dumps(actions)
    prompt = (
        f"Analyse the video with sensor_id `{sensor_id}` "
        f"(uploaded file: {video_name}).\n\n"
        "Use the video_understanding tool with that exact sensor_id. "
        "Set chunk_duration_secs=1 for fine temporal granularity.\n\n"
        "Task:\n"
        "1. Detect every person visible in the video. "
        "Number them left to right (1 = leftmost).\n"
        "2. For each person, list EACH action they performed in "
        "chronological order. Only use action names from this list "
        f"(exactly as written): {action_list}\n\n"
        "Return ONLY valid JSON in this exact format -- no other text:\n"
        '{"events":[\n'
        '  {"player_id":1,"action":"clap","confidence":0.9},\n'
        '  {"player_id":1,"action":"crouch","confidence":0.8}\n'
        "]}\n\n"
        "List events in chronological order. "
        'If a person did nothing recognizable, emit one event with action "idle".'
    )

    text = query_agent(prompt)
    log.info("Agent raw text: %s", text[:500])
    return _parse_events(text)


def _get_content(resp: dict[str, Any]) -> str:
    choices = resp.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        return msg.get("content", "")
    return json.dumps(resp)


def _parse_events(text: str) -> list[ActionEvent]:
    """Extract ActionEvent list from JSON embedded in LLM output."""
    events: list[ActionEvent] = []

    blocks = re.findall(r"\{[^{}]*\"events\"\s*:\s*\[.*?\]\s*\}", text, re.DOTALL)
    if not blocks:
        blocks = re.findall(r"\{[^{}]*\"players\"\s*:\s*\[.*?\]\s*\}", text, re.DOTALL)
        for block in blocks:
            events.extend(_parse_players_format(block))
        return events

    for block in blocks:
        try:
            data = json.loads(block)
            for i, ev in enumerate(data.get("events", [])):
                events.append(ActionEvent(
                    player_id=int(ev.get("player_id", 0)),
                    action=str(ev.get("action", "idle")).strip().lower(),
                    confidence=float(ev.get("confidence", 0.5)),
                    t_start=float(ev.get("t_start", i)),
                    t_end=float(ev.get("t_end", i + 1)),
                ))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            log.warning("Failed to parse event block: %s", exc)

    return events


def _parse_players_format(block: str) -> list[ActionEvent]:
    """Parse the older {players: [{id, actions}]} format as a fallback."""
    events: list[ActionEvent] = []
    try:
        data = json.loads(block)
        for p in data.get("players", []):
            pid = int(p.get("id", 0))
            for i, action in enumerate(p.get("actions", [])):
                events.append(ActionEvent(
                    player_id=pid,
                    action=str(action).strip().lower(),
                    confidence=0.7,
                    t_start=float(i),
                    t_end=float(i + 1),
                ))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        log.warning("Failed to parse players block: %s", exc)
    return events
