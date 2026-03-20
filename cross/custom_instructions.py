"""Custom instructions — hot-reloadable user instructions for gate and sentinel prompts.

Loads instructions from a markdown file (default: ~/.cross/instructions.md).
Re-reads the file on each access if the mtime has changed, so edits take
effect without restarting the daemon.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("cross.custom_instructions")


class CustomInstructions:
    """Lazily loads and caches a custom instructions file, hot-reloading on mtime change."""

    def __init__(self, path: str | Path = "~/.cross/instructions.md"):
        self._path = Path(path).expanduser()
        self._last_mtime: float = 0.0
        self._content: str = ""
        self._load()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def content(self) -> str:
        """Return the current instructions, reloading if the file changed."""
        self._maybe_reload()
        return self._content

    def save(self, text: str) -> None:
        """Write new instructions to disk (creates parent dirs if needed)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(text, encoding="utf-8")
        self._last_mtime = self._path.stat().st_mtime
        self._content = text
        logger.info(f"Custom instructions saved ({len(text)} chars)")

    def _load(self) -> None:
        """Read the file if it exists."""
        if not self._path.exists():
            self._content = ""
            self._last_mtime = 0.0
            return
        try:
            self._content = self._path.read_text(encoding="utf-8")
            self._last_mtime = self._path.stat().st_mtime
            if self._content:
                logger.info(f"Loaded custom instructions from {self._path} ({len(self._content)} chars)")
        except OSError as e:
            logger.warning(f"Failed to read custom instructions from {self._path}: {e}")
            self._content = ""

    def _maybe_reload(self) -> None:
        """Reload if the file's mtime has changed (or if the file was created/deleted)."""
        if not self._path.exists():
            if self._content:
                logger.info("Custom instructions file removed, clearing")
                self._content = ""
                self._last_mtime = 0.0
            return
        try:
            current_mtime = self._path.stat().st_mtime
        except OSError:
            return
        if current_mtime != self._last_mtime:
            logger.info("Custom instructions file changed, reloading...")
            self._load()


def format_instructions_block(instructions: str) -> str:
    """Wrap instructions in a clearly delimited block for inclusion in system prompts.

    Returns empty string if instructions are empty/whitespace.
    """
    text = instructions.strip()
    if not text:
        return ""
    return (
        "\n\n--- Custom Instructions (from the user who deployed this monitoring system) ---\n"
        + text
        + "\n--- End Custom Instructions ---"
    )
