"""Shared helpers for parsing Claude Code PTY output."""

import re


def is_permission_prompt(text: str) -> bool:
    """Detect actual Claude Code permission prompts vs TUI redraws.

    Requires the specific "Do you want to" prompt structure that Claude Code
    always shows, rather than broad pattern matching that catches redraws.
    """
    # "Do you want to" — may appear garbled as "Doyouwant" or "Do you want t"
    if re.search(r"Do\s*you\s*want\s*t", text, re.IGNORECASE):
        return True
    # Numbered options structure: "1. Yes" + "3. No" together
    if re.search(r"1\.\s*Yes", text) and re.search(r"3\.\s*No", text):
        return True
    return False


def extract_allow_all(text: str) -> str | None:
    """Extract the 'allow all...' option text from PTY output.

    PTY text may have spaces preserved or garbled together (no spaces),
    so we try both patterns.
    """
    # Clean text: "allow all edits in Downloads/"
    m = re.search(r"allow all (\w+) in (\S+/)", text, re.IGNORECASE)
    if m:
        return f"Allow all {m.group(1)} in {m.group(2)}"

    # Garbled text (no spaces): "allowalleditsinDownloads/"
    # Non-greedy \w+? backtracks until "in" matches
    m = re.search(r"allowall(\w+?)in(\S+?/)", text, re.IGNORECASE)
    if m:
        return f"Allow all {m.group(1)} in {m.group(2)}"

    # Bash commands (no directory)
    if re.search(r"allow\s*all\s*bash", text, re.IGNORECASE):
        return "Allow all Bash"

    return None
