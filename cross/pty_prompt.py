"""Interactive terminal prompts for cross escalation events.

Renders a simple vertical menu at the bottom of the terminal.
Arrow keys navigate, Enter selects, Escape dismisses.
"""

from __future__ import annotations

import fcntl
import os
import struct
import termios

_BOLD = "\033[1m"
_DIM = "\033[2m"
_INVERSE = "\033[7m"
_RESET = "\033[0m"
_CLEAR_LINE = "\033[2K"
_YELLOW = "\033[33m"
_SAVE_CURSOR = "\0337"
_RESTORE_CURSOR = "\0338"


def _get_terminal_rows(fd: int) -> int:
    try:
        data = fcntl.ioctl(fd, termios.TIOCGWINSZ, b"\x00" * 8)
        rows = struct.unpack("HHHH", data)[0]
        return rows or 24
    except OSError:
        return 24


def render_prompt(
    title: str,
    body: str,
    options: list[str],
    selected: int,
    fd: int,
) -> int:
    """Render the interactive prompt at the bottom of the terminal.

    Returns the number of lines rendered (for later cleanup).
    """
    lines: list[str] = []
    lines.append(f"  {_BOLD}{_YELLOW}\u26a0  cross: {title}{_RESET}")
    if body:
        for bline in body.split("\n")[:3]:
            lines.append(f"  {_DIM}{bline[:80]}{_RESET}")
    lines.append("")
    for i, opt in enumerate(options):
        if i == selected:
            lines.append(f"  {_INVERSE}{_BOLD} \u25b8 {opt} {_RESET}")
        else:
            lines.append(f"  {_DIM}   {opt}{_RESET}")

    num_lines = len(lines)
    rows = _get_terminal_rows(fd)
    start_row = max(1, rows - num_lines)

    buf = _SAVE_CURSOR
    for i, line in enumerate(lines):
        buf += f"\033[{start_row + i};1H{_CLEAR_LINE}{line}"
    os.write(fd, buf.encode())
    return num_lines


def clear_prompt(num_lines: int, fd: int):
    """Clear the prompt area and restore cursor."""
    rows = _get_terminal_rows(fd)
    start_row = max(1, rows - num_lines)

    buf = ""
    for i in range(num_lines):
        buf += f"\033[{start_row + i};1H{_CLEAR_LINE}"
    buf += _RESTORE_CURSOR
    os.write(fd, buf.encode())


def parse_key(data: bytes) -> str:
    """Parse raw terminal input into a key action."""
    if len(data) >= 3 and data[:3] == b"\x1b[A":
        return "up"
    if len(data) >= 3 and data[:3] == b"\x1b[B":
        return "down"
    if data in (b"\r", b"\n"):
        return "enter"
    if data == b"\x1b":
        return "escape"
    if data in (b"q", b"Q"):
        return "quit"
    return ""
