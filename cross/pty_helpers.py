"""Shared helpers for parsing Claude Code PTY output."""

import re


def is_permission_prompt(text: str) -> bool:
    """Detect actual Claude Code permission prompts vs TUI content.

    Requires BOTH the "Do you want to" question AND the numbered options
    structure together.  Either signal alone can appear in normal conversation
    text (e.g. the assistant discussing permission prompts), so both must be
    present to avoid false positives.
    """
    has_question = bool(re.search(r"Do\s*you\s*want\s*t", text, re.IGNORECASE))
    has_options = bool(re.search(r"1\.\s*Yes", text)) and bool(re.search(r"3\.\s*No", text))
    return has_question and has_options


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
