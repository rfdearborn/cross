"""Persistent event store — JSONL-backed bounded buffer of recent events.

Subscribes to EventBus, serializes events to disk, and provides a query
interface for the dashboard and other consumers.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import deque
from dataclasses import asdict
from typing import Any

from cross.events import CrossEvent

logger = logging.getLogger("cross.event_store")

_MAX_EVENTS = 100


def _default_path() -> str:
    from cross.config import settings

    return os.path.join(os.path.expanduser(settings.config_dir), "events.jsonl")


def event_to_dict(event: CrossEvent) -> dict[str, Any]:
    """Convert a CrossEvent dataclass to a JSON-serializable dict."""
    d = asdict(event)
    d["event_type"] = type(event).__name__
    d["ts"] = time.time()
    # Remove raw_body from RequestEvent — too large for storage
    d.pop("raw_body", None)
    return d


class EventStore:
    """Bounded, JSONL-persisted event buffer."""

    def __init__(self, path: str | None = None, max_events: int = _MAX_EVENTS):
        if path is None:
            path = _default_path()
        self._max_events = max_events
        self._path = path
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)

        # Load existing events from disk
        self._load()
        # Truncate file to bounded size
        self._truncate()

        # Open for appending
        os.makedirs(os.path.dirname(self._path), exist_ok=True)
        self._file = open(self._path, "a")

        if self._events:
            logger.info(f"Loaded {len(self._events)} events from {self._path}")

    async def handle_event(self, event: CrossEvent):
        """EventBus handler — serialize, store, and persist."""
        event_dict = event_to_dict(event)
        self._events.append(event_dict)
        self._file.write(json.dumps(event_dict) + "\n")
        self._file.flush()

    def get_events(self) -> list[dict[str, Any]]:
        """Return recent events as a list (oldest first)."""
        return list(self._events)

    def _load(self):
        """Load events from JSONL file."""
        try:
            with open(self._path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self._events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            pass

    def _truncate(self):
        """Truncate the JSONL file to keep only the last max_events entries."""
        try:
            with open(self._path) as f:
                lines = f.readlines()
        except FileNotFoundError:
            return
        if len(lines) <= self._max_events:
            return
        with open(self._path, "w") as f:
            f.writelines(lines[-self._max_events :])
