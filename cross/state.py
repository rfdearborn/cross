"""Daemon state persistence — survives cross restarts.

Persists session registry, project CWDs, gate agents, and sentinel event
window to ~/.cross/state.json so that agent monitoring is maintained
across daemon restarts.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

logger = logging.getLogger("cross.state")

_STATE_VERSION = 1


def _default_path() -> str:
    from cross.config import settings

    return os.path.join(os.path.expanduser(settings.config_dir), "state.json")


def save_state(
    *,
    sessions: dict[str, dict[str, Any]],
    project_cwds: dict[str, str],
    gate_agents: set[str],
    sentinel_events: list[dict[str, Any]] | None = None,
    halted_sessions: dict[str, str] | None = None,
    path: str | None = None,
) -> None:
    """Persist daemon state to disk.  Safe against crashes (atomic write)."""
    if path is None:
        path = _default_path()

    state = {
        "version": _STATE_VERSION,
        "saved_at": time.time(),
        "sessions": sessions,
        "project_cwds": project_cwds,
        "gate_agents": sorted(gate_agents),
    }
    if sentinel_events is not None:
        state["sentinel_events"] = sentinel_events
    if halted_sessions:
        state["halted_sessions"] = halted_sessions

    tmp_path = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp_path, "w") as f:
            json.dump(state, f)
        os.replace(tmp_path, path)  # atomic on POSIX
    except OSError as e:
        logger.warning(f"Failed to save state: {e}")
        # Clean up temp file on failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def load_state(path: str | None = None) -> dict[str, Any]:
    """Load persisted daemon state from disk.

    Returns a dict with keys: sessions, project_cwds, gate_agents,
    sentinel_events.  All default to empty if file missing or corrupt.
    """
    if path is None:
        path = _default_path()

    empty: dict[str, Any] = {
        "sessions": {},
        "project_cwds": {},
        "gate_agents": set(),
        "sentinel_events": [],
        "halted_sessions": {},
    }

    try:
        with open(path) as f:
            raw = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        if not isinstance(e, FileNotFoundError):
            logger.warning(f"Failed to load state from {path}: {e}")
        return empty

    if not isinstance(raw, dict) or raw.get("version") != _STATE_VERSION:
        logger.warning(f"Ignoring state file with unknown version: {raw.get('version')}")
        return empty

    # Restore sessions
    sessions: dict[str, dict[str, Any]] = {}
    for sid, sdata in raw.get("sessions", {}).items():
        if isinstance(sdata, dict):
            sessions[sid] = sdata

    # Restore project CWDs
    project_cwds = raw.get("project_cwds", {})
    if not isinstance(project_cwds, dict):
        project_cwds = {}

    # Restore gate agents
    gate_agents_raw = raw.get("gate_agents", [])
    gate_agents = set(gate_agents_raw) if isinstance(gate_agents_raw, list) else set()

    # Restore sentinel events — keep only the most recent N
    from cross.config import settings

    raw_events = [ev for ev in raw.get("sentinel_events", []) if isinstance(ev, dict)]
    sentinel_events = raw_events[-settings.llm_sentinel_max_events :]

    # Restore halted sessions
    halted_sessions = raw.get("halted_sessions", {})
    if not isinstance(halted_sessions, dict):
        halted_sessions = {}

    restored = {
        "sessions": sessions,
        "project_cwds": project_cwds,
        "gate_agents": gate_agents,
        "sentinel_events": sentinel_events,
        "halted_sessions": halted_sessions,
    }

    count_sessions = len(sessions)
    count_sentinel = len(sentinel_events)
    if count_sessions or count_sentinel:
        logger.info(
            f"Restored state: {count_sessions} sessions, "
            f"{len(project_cwds)} project CWDs, "
            f"{len(gate_agents)} gate agents, "
            f"{count_sentinel} sentinel events"
        )

    return restored


def clear_state(path: str | None = None) -> None:
    """Remove persisted state file."""
    if path is None:
        path = _default_path()
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning(f"Failed to clear state: {e}")
