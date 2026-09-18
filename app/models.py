from __future__ import annotations

from dataclasses import dataclass, field


ACTION_VOCAB = [
    "clap",
    "hands_on_head",
    "crouch",
    "jump",
    "raise_one_arm",
    "turn_around",
    "idle",
]

SEQUENCE_ACTIONS = ["clap", "crouch", "hands_on_head", "turn_around"]

ACTION_EMOJIS = {
    "clap": "\U0001f44f",
    "hands_on_head": "\U0001f646",
    "crouch": "\U0001f9ce",
    "jump": "\U0001f3c3",
    "raise_one_arm": "\U0001f91a",
    "turn_around": "\U0001f504",
    "idle": "\U0001f9cd",
}


@dataclass
class ActionEvent:
    player_id: int
    action: str
    confidence: float = 1.0
    t_start: float = 0.0
    t_end: float = 0.0


@dataclass
class PlayerResult:
    player_id: int
    name: str
    events: list[ActionEvent] = field(default_factory=list)
    detected_sequence: list[str] = field(default_factory=list)
    passed: bool = False
    time_ok: bool = False
    score: float = 0.0


@dataclass
class GameResult:
    expected: list[str]
    players: list[PlayerResult] = field(default_factory=list)
    winner: PlayerResult | None = None
    raw_events: list[ActionEvent] = field(default_factory=list)
    adjudicated: bool = False
    winner_image: str | None = None
