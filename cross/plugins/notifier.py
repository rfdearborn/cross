"""Native desktop notifications for gate escalations and sentinel alerts.

Uses terminal-notifier on macOS (click opens dashboard). Suppressed when
a dashboard tab is open (browser notifications handle it instead, so
clicking focuses the existing tab rather than opening a new one).
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from typing import Callable

from cross.config import settings
from cross.events import CrossEvent, GateDecisionEvent, SentinelReviewEvent

logger = logging.getLogger("cross.plugins.notifier")

_IS_MACOS = platform.system() == "Darwin"
_TERMINAL_NOTIFIER = shutil.which("terminal-notifier") if _IS_MACOS else None

# Set by the daemon to check if browser clients are connected
_has_browser_clients: Callable[[], bool] | None = None


def is_available() -> bool:
    """Return True if native desktop notifications are enabled and supported."""
    from cross.config import settings

    return settings.native_notifications_enabled and _TERMINAL_NOTIFIER is not None


def set_browser_check(check: Callable[[], bool]):
    """Register a callback that returns True when dashboard tabs are open."""
    global _has_browser_clients
    _has_browser_clients = check


def _dashboard_url() -> str:
    return f"http://localhost:{settings.listen_port}/cross/dashboard"


def _notify(title: str, body: str):
    """Send a native desktop notification via terminal-notifier."""
    if not _TERMINAL_NOTIFIER:
        return
    try:
        subprocess.Popen(
            [
                _TERMINAL_NOTIFIER,
                "-title",
                title,
                "-message",
                body,
                "-open",
                _dashboard_url(),
                "-group",
                "cross",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError:
        logger.debug("Failed to send desktop notification")


async def handle_event(event: CrossEvent):
    """EventBus handler — send native notifications for escalations and alerts."""
    # If a dashboard tab is open, let browser notifications handle it
    if _has_browser_clients and _has_browser_clients():
        return

    if isinstance(event, GateDecisionEvent) and event.action == "escalate":
        tool = event.tool_name or "unknown tool"
        reason = event.reason or ""
        body = f"{tool}: {reason}" if reason else tool
        _notify("cross — approval needed", body)

    elif isinstance(event, SentinelReviewEvent) and event.action and event.action != "allow":
        body = event.summary or event.concerns or ""
        _notify(f"cross — sentinel {event.action}", body)
