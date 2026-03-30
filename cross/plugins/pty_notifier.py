"""PTY terminal notifications and interactive prompts.

For gate ESCALATE events (where the proxy is blocking, waiting for human
approval), renders an interactive prompt in the terminal with arrow-key
navigation.  For informational events (BLOCK, ALERT, sentinel reviews),
sends OSC 9 terminal notifications that don't interfere with the TUI.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Awaitable, Callable

from cross.ansi import format_notification
from cross.config import settings
from cross.events import CrossEvent, GateDecisionEvent, SentinelReviewEvent

logger = logging.getLogger("cross.plugins.pty_notifier")

_DEDUP_TTL = 5.0  # seconds
_recent: dict[str, float] = {}

# Async callbacks set by daemon
_notify_callback: Callable[[str, str], Awaitable[None]] | None = None
_prompt_callback: Callable[[str, dict], Awaitable[None]] | None = None


def set_notify_callback(callback: Callable[[str, str], Awaitable[None]]):
    global _notify_callback
    _notify_callback = callback


def set_prompt_callback(callback: Callable[[str, dict], Awaitable[None]]):
    global _prompt_callback
    _prompt_callback = callback


def _dedup(key: str) -> bool:
    """Return True if this key was seen within the TTL (i.e. should skip)."""
    now = time.time()
    stale = [k for k, t in _recent.items() if now - t > _DEDUP_TTL]
    for k in stale:
        del _recent[k]
    if key in _recent:
        return True
    _recent[key] = now
    return False


def _format_tool_summary(event: GateDecisionEvent) -> str:
    """Build a short body string showing what the tool call does."""
    parts: list[str] = []
    if event.tool_input:
        # Show the most relevant field (command for Bash, file_path for Write, etc.)
        for key in ("command", "file_path", "path", "query", "url"):
            if key in event.tool_input:
                val = str(event.tool_input[key])
                parts.append(val[:120] if len(val) > 120 else val)
                break
        else:
            # Fallback: compact JSON
            compact = json.dumps(event.tool_input, separators=(",", ":"))
            parts.append(compact[:120])
    if event.reason:
        parts.append(event.reason[:200])
    return "\n".join(parts)


_GATE_NOTIFY_STYLES = {
    "block": "error",
    "halt_session": "error",
    "alert": "alert",
}

_SENTINEL_STYLES = {
    "halt_session": "error",
    "escalate": "warning",
    "alert": "alert",
}


async def handle_event(event: CrossEvent):
    """EventBus handler — interactive prompts for escalations, OSC 9 for the rest."""
    if not settings.pty_notifications_enabled:
        return

    session_id = getattr(event, "session_id", "")
    if not session_id:
        return

    # Gate ESCALATE → interactive prompt (proxy is blocking, waiting for human)
    if isinstance(event, GateDecisionEvent) and event.action == "escalate":
        if not _prompt_callback:
            return
        if _dedup(f"gate:{event.tool_use_id}"):
            return
        prompt_data = {
            "type": "prompt",
            "prompt_id": event.tool_use_id,
            "title": f"Gate ESCALATE \u2014 {event.tool_name}",
            "body": _format_tool_summary(event),
            "options": ["Allow", "Deny"],
        }
        try:
            await _prompt_callback(session_id, prompt_data)
        except Exception as e:
            logger.debug(f"Failed to send PTY prompt: {e}")
        return

    # Other gate events → OSC 9 notification (informational)
    if isinstance(event, GateDecisionEvent) and event.action in _GATE_NOTIFY_STYLES:
        if not _notify_callback:
            return
        if _dedup(f"gate:{event.tool_use_id}"):
            return
        title = f"Gate {event.action.upper()}: {event.tool_name}"
        body = event.reason or ""
        style = _GATE_NOTIFY_STYLES[event.action]
        banner = format_notification(title, body, style=style)
        try:
            await _notify_callback(session_id, banner)
        except Exception as e:
            logger.debug(f"Failed to send PTY notification: {e}")
        return

    # Sentinel events → OSC 9 notification
    if isinstance(event, SentinelReviewEvent) and event.action in _SENTINEL_STYLES:
        if not _notify_callback:
            return
        if _dedup(f"sentinel:{event.review_id or 'no-id'}"):
            return
        title = f"Sentinel {event.action.upper()}"
        body = event.summary or event.concerns or ""
        style = _SENTINEL_STYLES[event.action]
        banner = format_notification(title, body, style=style)
        try:
            await _notify_callback(session_id, banner)
        except Exception as e:
            logger.debug(f"Failed to send PTY notification: {e}")
