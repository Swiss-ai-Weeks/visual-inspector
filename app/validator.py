"""Sequence Validator (Agent 3) -- fully deterministic, no LLM.

Checks whether each player performed the expected action sequence
in the correct order within the time limit.
"""

from __future__ import annotations

from itertools import groupby

from models import ActionEvent, PlayerResult


CONFIDENCE_THRESHOLD = 0.5
TIME_LIMIT = 5.0


def validate(
    events: list[ActionEvent],
    expected: list[str],
    t0: float = 0.0,
    *,
    time_limit: float = TIME_LIMIT,
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    lenient: bool = False,
) -> list[PlayerResult]:
    """Validate per-player action sequences against the expected sequence.

    Returns a PlayerResult for each detected player_id.
    """
    per_player: dict[int, list[ActionEvent]] = {}
    for ev in events:
        per_player.setdefault(ev.player_id, []).append(ev)

    results: list[PlayerResult] = []
    for pid in sorted(per_player):
        evts = sorted(per_player[pid], key=lambda e: e.t_start)
        evts = _dedupe_consecutive(evts)
        evts = [e for e in evts if e.action != "idle" and e.confidence > confidence_threshold]

        seq = [e.action for e in evts]
        time_ok = not evts or (evts[-1].t_end - t0) <= time_limit

        if lenient:
            ok_order = _lcs_length(seq, expected) == len(expected)
            score = _lcs_length(seq, expected) / max(len(expected), 1)
        else:
            ok_order = seq[:len(expected)] == expected
            score = 1.0 if ok_order else _partial_score(seq, expected)

        results.append(PlayerResult(
            player_id=pid,
            name=f"Player {pid}",
            events=evts,
            detected_sequence=seq,
            passed=ok_order and time_ok,
            time_ok=time_ok,
            score=score,
        ))

    return results


def needs_adjudication(events: list[ActionEvent]) -> list[ActionEvent]:
    """Return events whose confidence is below the threshold."""
    return [e for e in events if 0.0 < e.confidence <= CONFIDENCE_THRESHOLD]


def _dedupe_consecutive(events: list[ActionEvent]) -> list[ActionEvent]:
    """Collapse runs of the same action from consecutive 1s chunks."""
    deduped: list[ActionEvent] = []
    for action, group in groupby(events, key=lambda e: e.action):
        items = list(group)
        merged = ActionEvent(
            player_id=items[0].player_id,
            action=action,
            confidence=max(e.confidence for e in items),
            t_start=items[0].t_start,
            t_end=items[-1].t_end,
        )
        deduped.append(merged)
    return deduped


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _partial_score(detected: list[str], expected: list[str]) -> float:
    """Score 0-1 based on how many of the expected actions appear in order."""
    if not expected:
        return 1.0
    return _lcs_length(detected, expected) / len(expected)
