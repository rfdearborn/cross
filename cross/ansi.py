"""ANSI escape code stripping and terminal output cleaning."""

from __future__ import annotations

import re

# Matches ANSI escape sequences comprehensively
_ANSI_RE = re.compile(
    r"\x1b"  # ESC character
    r"(?:"
    r"\[[0-9;?]*[A-Za-z]"  # CSI sequences: ESC [ params letter
    r"|\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC sequences
    r"|[()][AB012]"  # Character set selection
    r"|[=>]"  # Keypad modes
    r"|."  # Other single-char escapes
    r")"
)

# Leftover partial CSI params (e.g. "38;2;248;242m" without the ESC[)
# Require 2+ semicolon-separated numbers to avoid false positives on normal text
_PARTIAL_CSI_RE = re.compile(r"\d+(?:;\d+)+[mGKHJABCDfsu]")

# Control characters to strip (except newline, tab, carriage return)
_CTRL_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1a\x1c-\x1f\x7f]")

# Box drawing and decorative characters that clutter output
_DECORATION_RE = re.compile(r"[╌─━┌┐└┘├┤┬┴┼╭╮╯╰│║═]+")


# Icons used in OSC 9 notification text
_NOTIFICATION_ICONS: dict[str, str] = {
    "error": "\U0001f6d1",  # stop sign
    "warning": "\u26a0\ufe0f",  # warning
    "alert": "\U0001f514",  # bell
}


def format_notification(title: str, body: str = "", style: str = "error") -> str:
    """Format a terminal notification using OSC 9.

    OSC 9 is processed by the terminal emulator (iTerm2, Kitty, Ghostty,
    Warp, Windows Terminal) and shown as a native toast/tab notification.
    It does NOT interfere with full-screen TUI apps like Claude Code or
    Codex, which is why we use this instead of inline ANSI banners.

    Returns an OSC 9 escape sequence ready to write to the terminal.
    """
    icon = _NOTIFICATION_ICONS.get(style, _NOTIFICATION_ICONS["error"])
    text = f"{icon} cross: {title}"
    if body:
        if len(body) > 200:
            body = body[:200] + "..."
        text += f" — {body}"
    # OSC 9 ; <text> BEL — iTerm2/Kitty/Ghostty terminal notification
    return f"\033]9;{text}\a"


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
