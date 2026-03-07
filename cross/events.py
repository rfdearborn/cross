from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


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


@dataclass
class ToolUseEvent:
    """Fired when a tool_use block is fully assembled from the SSE stream."""
    name: str
    tool_use_id: str
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextEvent:
    """Fired when a complete text block is assembled."""
    text: str


@dataclass
class MessageStartEvent:
    """Fired on message_start SSE event."""
    message_id: str
    model: str


@dataclass
class MessageDeltaEvent:
    """Fired on message_delta SSE event (end of message)."""
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ErrorEvent:
    """Fired on API errors."""
    status_code: int
    body: str


CrossEvent = RequestEvent | ToolUseEvent | TextEvent | MessageStartEvent | MessageDeltaEvent | ErrorEvent

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
