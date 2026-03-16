"""Script content resolver — reads the source of scripts that agents execute.

When a Bash command runs a script file (e.g., `python bad_script.py`), this
module resolves the file path and reads its contents so that gate and sentinel
LLMs can review what the script actually does, not just its filename.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger("cross.script_resolver")

# Max bytes to read from a script file (avoid loading huge binaries)
_MAX_SCRIPT_BYTES = 50_000

# Interpreters whose first positional argument is typically a script file
_INTERPRETER_RE = re.compile(
    r"""
    (?:^|&&|\|\||;|\|)\s*       # start of command or chained command
    (?:sudo\s+)?                # optional sudo
    (?:
        python[23w]?(?:\.\d+)?  # python, python3, python3.11, pythonw
      | node                    # node
      | ruby                    # ruby
      | perl                    # perl
      | php                     # php
      | Rscript                 # R
      | lua                     # lua
      | (?:ba|z|fi|c|da|a|tc|k)?sh  # shell interpreters
    )
    \s+                         # required space before args
    """,
    re.VERBOSE,
)

# Flags that mean "not running a script file" — stop looking after these
_NO_SCRIPT_FLAGS = re.compile(r"^-[cme]$")

# Flags that take a value argument (skip these and their values)
_FLAG_WITH_VALUE_RE = re.compile(r"^-[WXEIOQ]$|^--\w+=")

# File extensions that are clearly scripts (not binaries)
_SCRIPT_EXTENSIONS = {
    ".py", ".js", ".ts", ".mjs", ".cjs", ".rb", ".pl", ".pm",
    ".php", ".lua", ".r", ".R", ".sh", ".bash", ".zsh",
    ".ksh", ".fish", ".tcl", ".awk",
}


def _looks_like_script_path(token: str) -> bool:
    """Heuristic: does this token look like a file path (not a flag or module)?"""
    if token.startswith("-"):
        return False
    # Has a script extension
    _, ext = os.path.splitext(token)
    if ext.lower() in _SCRIPT_EXTENSIONS:
        return True
    # Contains a path separator (./foo, ../bar, /path/to/script)
    if "/" in token or token.startswith("."):
        return True
    return False


def extract_script_paths(command: str) -> list[str]:
    """Extract probable script file paths from a bash command string.

    Returns a list of candidate paths (may be relative).
    """
    paths: list[str] = []

    for match in _INTERPRETER_RE.finditer(command):
        # Get everything after the interpreter
        rest = command[match.end():]
        tokens = rest.split()

        i = 0
        while i < len(tokens):
            token = tokens[i]
            # Stop at shell operators
            if token in ("&&", "||", ";", "|", ">", ">>", "<", "2>", "2>>"):
                break
            # Skip flags
            if token.startswith("-"):
                # -c, -m, -e mean "not running a script file" — stop looking
                if _NO_SCRIPT_FLAGS.match(token):
                    break
                # Some flags consume the next token as a value
                if _FLAG_WITH_VALUE_RE.match(token) and i + 1 < len(tokens):
                    i += 2
                    continue
                i += 1
                continue
            # First non-flag token is the script path
            if _looks_like_script_path(token):
                paths.append(token)
            break

    return paths


def resolve_script_contents(
    command: str,
    cwd: str = "",
) -> dict[str, str]:
    """Resolve and read script files referenced in a bash command.

    Args:
        command: The bash command string.
        cwd: Working directory for resolving relative paths.

    Returns:
        Dict mapping resolved file paths to their contents (truncated).
        Empty dict if no scripts found or files unreadable.
    """
    paths = extract_script_paths(command)
    if not paths:
        return {}

    results: dict[str, str] = {}
    cwd_path = Path(cwd) if cwd else Path.cwd()

    for script_path in paths:
        try:
            p = Path(script_path)
            if not p.is_absolute():
                p = cwd_path / p
            p = p.resolve()

            if not p.is_file():
                continue
            if p.stat().st_size > _MAX_SCRIPT_BYTES:
                results[str(p)] = f"[file too large: {p.stat().st_size} bytes, limit {_MAX_SCRIPT_BYTES}]"
                continue

            content = p.read_text(errors="replace")
            results[str(p)] = content
        except (OSError, ValueError) as e:
            logger.debug(f"Could not read script {script_path}: {e}")

    return results
