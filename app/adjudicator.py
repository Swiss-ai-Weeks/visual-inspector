"""Adjudicator Agent (Agent 4) -- LLM-based, on-demand only.

Fires when the Perception Agent returns low-confidence events.
Uses VSS /chat/completions (CA-RAG temporal reasoning) to ask
targeted questions about specific time windows.
"""

from __future__ import annotations

import json
import logging

from models import ACTION_VOCAB, ActionEvent
from vss_client import chat_query

log = logging.getLogger(__name__)


def adjudicate(
    sensor_id: str,
    uncertain_events: list[ActionEvent],
) -> list[ActionEvent]:
    """Re-examine low-confidence events by querying VSS CA-RAG.

    Returns updated events with revised action and confidence.
    """
    refined: list[ActionEvent] = []

    for ev in uncertain_events:
        question = (
            f"Between {ev.t_start:.1f}s and {ev.t_end:.1f}s, "
            f"what action did Person {ev.player_id} perform? "
            f"Choose exactly one from: {json.dumps(ACTION_VOCAB)}. "
            f"Reply with JSON: "
            f'{{"player_id": {ev.player_id}, "action": "<choice>", "confidence": 0.0-1.0}}'
        )

        try:
            answer = chat_query(sensor_id, question)
            log.info("Adjudicator answer for P%d [%.1f-%.1f]: %s",
                     ev.player_id, ev.t_start, ev.t_end, answer[:200])
            parsed = _parse_answer(answer, ev)
            refined.append(parsed)
        except Exception as exc:
            log.warning("Adjudicator failed for P%d: %s", ev.player_id, exc)
            refined.append(ev)

    return refined


def _parse_answer(text: str, original: ActionEvent) -> ActionEvent:
    """Try to extract a revised ActionEvent from the adjudicator's answer."""
    import re

    match = re.search(r"\{[^}]*\"action\"\s*:\s*\"([^\"]+)\"[^}]*\}", text)
    if not match:
        return original

    try:
        block = json.loads(match.group())
        action = str(block.get("action", "idle")).strip().lower()
        if action not in ACTION_VOCAB:
            action = original.action
        confidence = float(block.get("confidence", 0.6))
        return ActionEvent(
            player_id=original.player_id,
            action=action,
            confidence=confidence,
            t_start=original.t_start,
            t_end=original.t_end,
        )
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return original
