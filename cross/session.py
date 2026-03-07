"""Session registry — tracks active agent sessions."""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from cross.pty_wrapper import PTYSession


@dataclass
class SessionInfo:
    """Metadata for a tracked session."""

    session_id: str
    agent: str  # e.g. "claude", "codex"
    argv: list[str]
    project: str  # derived from cwd or explicit
    cwd: str
    started_at: float
    ended_at: float | None = None
    exit_code: int | None = None
    pty_session: PTYSession | None = field(default=None, repr=False)


class SessionRegistry:
    """In-memory registry of active and recent sessions."""

    def __init__(self):
        self._sessions: dict[str, SessionInfo] = {}

    def create(self, agent: str, argv: list[str], cwd: str | None = None) -> SessionInfo:
        """Register a new session."""
        session_id = uuid.uuid4().hex[:12]
        work_dir = cwd or os.getcwd()
        project = _detect_project(work_dir)

        info = SessionInfo(
            session_id=session_id,
            agent=agent,
            argv=argv,
            project=project,
            cwd=work_dir,
            started_at=time.time(),
            pty_session=PTYSession(),
        )
        self._sessions[session_id] = info
        return info

    def get(self, session_id: str) -> SessionInfo | None:
        return self._sessions.get(session_id)

    def active(self) -> list[SessionInfo]:
        return [s for s in self._sessions.values() if s.ended_at is None]

    def complete(self, session_id: str, exit_code: int):
        info = self._sessions.get(session_id)
        if info:
            info.ended_at = time.time()
            info.exit_code = exit_code


# Global singleton
registry = SessionRegistry()


def _detect_project(cwd: str) -> str:
    """Derive a project name from the working directory.

    Walks up looking for a git root, falls back to the directory name.
    """
    path = Path(cwd).resolve()
    for parent in [path, *path.parents]:
        if (parent / ".git").exists():
            return parent.name
        if parent == parent.parent:
            break
    return path.name
