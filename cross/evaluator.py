"""Evaluator protocol — shared types for gates and sentinels.

Gates evaluate individual tool calls inline (synchronous, blocking).
Sentinels observe the event stream over time (asynchronous, periodic).
Both return EvaluationResponse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

PROTOCOL_VERSION = "0.1"


class Action(Enum):
    """Evaluator verdict, ordered by severity. Chain uses max()."""

    ABSTAIN = 0
    ALLOW = 1
    ALERT = 2
    ESCALATE = 3
    BLOCK = 4  # gates only: prevent this tool call
    HALT_SESSION = 5  # sentinels only: stop the session


@dataclass
class GateRequest:
    """Input to a gate evaluator — a single tool call to evaluate."""

    version: str = PROTOCOL_VERSION
    tool_use_id: str = ""
    tool_name: str = ""
    tool_input: Any = None
    session_id: str = ""
    agent: str = ""
    timestamp: float = 0.0
    cwd: str = ""
    # Context
    recent_tools: list[dict[str, Any]] = field(default_factory=list)
    user_intent: str = ""
    tool_index_in_message: int = 0
    tool_count_in_message: int = 0
    # Set by chain when invoking review gate (stage 2)
    prior_result: "EvaluationResponse | None" = None


@dataclass
class SentinelRequest:
    """Input to a sentinel evaluator — a window of events to review."""

    version: str = PROTOCOL_VERSION
    session_id: str = ""
    agent: str = ""
    timestamp: float = 0.0
    # The event window
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    session_summary: str = ""
    window_start: float = 0.0
    window_end: float = 0.0
    user_intent: str = ""


@dataclass
class EvaluationResponse:
    """Output from any evaluator."""

    version: str = PROTOCOL_VERSION
    action: Action = Action.ABSTAIN
    reason: str = ""
    rule_id: str = ""
    confidence: float = 1.0
    evaluator: str = ""
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Evaluator:
    """Base class for all evaluators."""

    name: str = ""

    def __init__(self, name: str = ""):
        self.name = name or self.__class__.__name__


class Gate(Evaluator):
    """Inline evaluator. Checks each tool call before release."""

    timeout_ms: float = 50.0
    on_error: Action = Action.ALLOW

    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        """Evaluate a single tool call. Must be fast."""
        return EvaluationResponse(action=Action.ABSTAIN, evaluator=self.name)


class Sentinel(Evaluator):
    """Observational, asynchronous evaluator. Watches events over time."""

    async def observe(self, event: Any) -> None:
        """Receive an event. Accumulate state. Never blocks the stream."""
        pass

    async def evaluate(self, request: SentinelRequest) -> EvaluationResponse:
        """Review a window of events. Called periodically or on trigger."""
        return EvaluationResponse(action=Action.ABSTAIN, evaluator=self.name)
