"""ANSI escape code stripping and terminal output cleaning."""

from __future__ import annotations

import re

# Matches ANSI escape sequences comprehensively
_ANSI_RE = re.compile(
    r"\x1b"           # ESC character
    r"(?:"
    r"\[[0-9;?]*[A-Za-z]"    # CSI sequences: ESC [ params letter
    r"|\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC sequences
    r"|[()][AB012]"           # Character set selection
    r"|[=>]"                  # Keypad modes
    r"|."                     # Other single-char escapes
    r")"
)

# Leftover partial CSI params (e.g. "38;2;248;242m" without the ESC[)
# Require 2+ semicolon-separated numbers to avoid false positives on normal text
_PARTIAL_CSI_RE = re.compile(r"\d+(?:;\d+)+[mGKHJABCDfsu]")

# Control characters to strip (except newline, tab, carriage return)
_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1a\x1c-\x1f\x7f]")

# Box drawing and decorative characters that clutter output
_DECORATION_RE = re.compile(r"[╌─━┌┐└┘├┤┬┴┼╭╮╯╰│║═]+")


def strip_ansi(data: bytes) -> str:
    """Strip ANSI escape codes, control characters, and decoration from terminal output."""
    text = data.decode("utf-8", errors="replace")
    text = _ANSI_RE.sub("", text)
    text = _PARTIAL_CSI_RE.sub("", text)
    text = _CTRL_RE.sub("", text)
    text = _DECORATION_RE.sub("", text)
    # Collapse multiple spaces/blank lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text
