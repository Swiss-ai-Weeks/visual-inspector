from __future__ import annotations

from dataclasses import dataclass, field


# Natural-language moves the VSS path can find. Free text is fine anywhere the
# user types their own; this pool is only what the Play page draws random
# challenges from.
MOVE_POOL = [
    "cross your arms",
    "raise one arm",
    "put your hands on your head",
    "crouch down",
    "turn around",
    "clap your hands",
    "jump",
    "touch your head",
    "wave",
    "point at the camera",
]

# Best-effort emoji for display. Matched by keyword against a (free-text) move;
# falls back to a target when nothing matches.
_EMOJI_KEYWORDS = [
    (("cross", "arm"), "\U0001f645"),
    (("hands", "head"), "\U0001f646"),
    (("touch", "head"), "\U0001f9d1"),
    (("raise", "arm"), "\U0001f91a"),
    (("crouch",), "\U0001f9ce"),
    (("turn",), "\U0001f504"),
    (("clap",), "\U0001f44f"),
    (("jump",), "\U0001f3c3"),
    (("wave",), "\U0001f44b"),
    (("point",), "\U0001f449"),
]


def move_emoji(move: str) -> str:
    low = move.lower()
    for keys, emoji in _EMOJI_KEYWORDS:
        if all(k in low for k in keys):
            return emoji
    return "\U0001f3af"  # 🎯


@dataclass
class GameResult:
    """Outcome of one round: who first performed the move / completed the sequence.

    Mirrors the notebook's ``{winner, timestamp, num_people}`` result, plus the
    human-facing bits the templates need.
    """
    moves: list[str]
    winner_number: int = 0
    winner_name: str | None = None
    timestamp: float | None = None
    num_people: int = 0
    method: str = "vss"                     # "vss" | "pose"
    winner_image: str | None = None
    timeline: list[tuple] = field(default_factory=list)  # [(start, end, text)]

    @property
    def has_winner(self) -> bool:
        # winner is a 0-based tracker id, so "nobody" is signalled by no timestamp.
        return self.timestamp is not None
