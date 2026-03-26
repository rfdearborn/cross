from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable


@dataclass
class RequestEvent:
    """Fired when a request is received from the client."""

    method: str
    path: str
    model: str | None = None
    messages_count: int = 0
    stream: bool = False
    tool_names: list[str] = field(default_factory=list)
    last_message_role: str | None = None
    last_message_preview: str | None = None
    raw_body: dict[str, Any] | None = None
    agent: str = ""
    session_id: str = ""


@dataclass
class ToolUseEvent:
    """Fired when a tool_use block is fully assembled from the SSE stream."""

    name: str
    tool_use_id: str
    input: dict[str, Any] = field(default_factory=dict)
    script_contents: dict[str, str] | None = None  # resolved script file contents (path -> source)
    agent: str = ""  # source agent (for routing notifications)
    session_id: str = ""


@dataclass
class TextEvent:
    """Fired when a complete text block is assembled."""

    text: str
    agent: str = ""
    session_id: str = ""


@dataclass
class MessageStartEvent:
    """Fired on message_start SSE event."""

    message_id: str
    model: str
    agent: str = ""
    session_id: str = ""


@dataclass
class MessageDeltaEvent:
    """Fired on message_delta SSE event (end of message)."""

    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    agent: str = ""
    session_id: str = ""


@dataclass
class ErrorEvent:
    """Fired on API errors."""

    status_code: int
    body: str


@dataclass
class GateDecisionEvent:
    """Fired when the gate chain makes a decision about a tool_use."""

    tool_use_id: str
    tool_name: str
    action: str  # "allow", "block", "alert", "escalate", "abstain"
    reason: str = ""
    rule_id: str = ""
    evaluator: str = ""
    confidence: float = 1.0
    tool_input: dict[str, Any] | None = None  # included so sentinel can see what was blocked
    script_contents: dict[str, str] | None = None  # resolved script file contents (path -> source)
    agent: str = ""  # source agent (for routing notifications)
    session_id: str = ""
    # Context for follow-up conversations ("Ask about this evaluation...")
    recent_tools: list[dict[str, Any]] | None = None
    user_intent: str = ""
    conversation_context: list[dict[str, str]] | None = None
    # Original LLM evaluation conversation (for seeding follow-ups)
    eval_system_prompt: str = ""
    eval_user_message: str = ""
    eval_response_text: str = ""


@dataclass
class GateRetryEvent:
    """Fired when a blocked tool call triggers a retry."""

    tool_use_id: str
    tool_name: str
    reason: str
    retry_number: int
    max_retries: int


@dataclass
class SentinelReviewEvent:
    """Fired when the sentinel completes a periodic review."""

    action: str  # "allow", "alert", "escalate", "halt_session"
    summary: str = ""
    concerns: str = ""
    event_count: int = 0
    evaluator: str = ""
    review_id: str = ""  # unique ID for conversation threading
    event_window_text: str = ""  # formatted event window for conversation context
    agent: str = ""  # dominant agent in the review window (for Slack routing)
    session_id: str = ""
    # Original LLM evaluation conversation (for seeding follow-ups)
    eval_system_prompt: str = ""
    eval_user_message: str = ""
    eval_response_text: str = ""


@dataclass
class PermissionPromptEvent:
    """Fired when Claude Code shows a native permission prompt in a PTY session."""

    session_id: str
    tool_desc: str = ""
    allow_all_label: str = "Allow all (session)"


@dataclass
class PermissionResolvedEvent:
    """Fired when a permission prompt is resolved from any surface."""

    session_id: str
    action: str = ""  # "approve", "allow_all", "deny"
    resolver: str = ""  # "slack", "dashboard", "terminal", "cli"


CrossEvent = (
    RequestEvent
    | ToolUseEvent
    | TextEvent
    | MessageStartEvent
    | MessageDeltaEvent
    | ErrorEvent
    | GateDecisionEvent
    | GateRetryEvent
    | SentinelReviewEvent
    | PermissionPromptEvent
    | PermissionResolvedEvent
)

EventHandler = Callable[[CrossEvent], Awaitable[None]]


class EventBus:
    def __init__(self):
        self._handlers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler):
        self._handlers.append(handler)

    async def publish(self, event: CrossEvent):
        for handler in self._handlers:
            try:
                await handler(event)
            except Exception as e:
                # Don't let a plugin crash the proxy
                import logging

                logging.getLogger("cross.events").exception(f"Plugin error handling {type(event).__name__}: {e}")
