"""Custom instructions — hot-reloadable user instructions for gate and sentinel prompts.

Loads instructions from a markdown file (default: ~/.cross/instructions.md).
Re-reads the file on each access if the mtime has changed, so edits take
effect without restarting the daemon.

Supports per-project instructions via ``<project>/.cross/instructions.md``.
Project instructions are additive to global instructions, but global
instructions take precedence when there is a conflict.
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


class ProjectInstructionsCache:
    """Cache of per-project instruction files, keyed by project root.

    Looks for ``<cwd>/.cross/instructions.md``.  Hot-reloads when the file
    changes, just like the global :class:`CustomInstructions`.
    """

    def __init__(self):
        self._cache: dict[str, tuple[float, str]] = {}  # path -> (mtime, content)

    def get(self, cwd: str) -> str:
        """Return project instructions for *cwd*, or ``""`` if none exist."""
        if not cwd:
            return ""
        path = Path(cwd) / ".cross" / "instructions.md"
        cached = self._cache.get(str(path))

        if not path.exists():
            if cached:
                del self._cache[str(path)]
            return ""

        try:
            mtime = path.stat().st_mtime
        except OSError:
            return cached[1] if cached else ""

        if cached and cached[0] == mtime:
            return cached[1]

        try:
            content = path.read_text(encoding="utf-8")
            self._cache[str(path)] = (mtime, content)
            logger.info(f"Loaded project instructions from {path} ({len(content)} chars)")
            return content
        except OSError as e:
            logger.warning(f"Failed to read project instructions from {path}: {e}")
            return cached[1] if cached else ""


def merge_instructions(global_content: str, project_content: str) -> str:
    """Merge global and project instructions.

    Both are included; global instructions appear first and are marked as
    higher-priority so the LLM knows to prefer them on conflict.
    """
    g = global_content.strip()
    p = project_content.strip()
    if not g and not p:
        return ""
    if not p:
        return g
    if not g:
        return p
    return (
        g
        + "\n\n--- Project-Specific Instructions (lower priority than global) ---\n"
        + p
        + "\n--- End Project-Specific Instructions ---"
    )


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
