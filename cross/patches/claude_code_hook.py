#!/usr/bin/env python3
"""cross gate hook for Claude Code (PreToolUse).

Installed as a Claude Code hook in ~/.claude/settings.json.
Reads tool call JSON from stdin, POSTs to cross daemon's /cross/api/gate
endpoint, and blocks if the gate denies the call.

Fails open (allows) if the cross daemon is unreachable.
Skips gating for sessions already proxied via `cross wrap` to avoid
double-gating.
"""

import json
import os
import sys
import urllib.error
import urllib.request

CROSS_PORT = os.environ.get("CROSS_LISTEN_PORT", "2767")
GATE_URL = f"http://localhost:{CROSS_PORT}/cross/api/gate"
TIMEOUT_S = 300  # 5 minutes — allows for human escalation review

MAX_CONV_TURNS = int(os.environ.get("CROSS_LLM_GATE_CONTEXT_TURNS", "5"))
MAX_CHARS_PER_TURN = 300
MAX_INTENT_CHARS = 500
_SKIP_PREFIXES = ("<system-reminder>", "[Request interrupted by user]", "Conversation info")


def _is_already_proxied() -> bool:
    """Check if this session is already routed through cross's proxy."""
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
    return "localhost" in base_url or "127.0.0.1" in base_url


def _extract_text(content) -> str:
    """Extract text from a message content field (string or list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return " ".join(parts)
    return ""


def _read_transcript(transcript_path: str) -> tuple[list[dict[str, str]], str]:
    """Read the Claude Code transcript JSONL and extract conversation context + user intent.

    Returns (conversation_context, user_intent).
    """
    if not transcript_path or not os.path.isfile(transcript_path):
        return [], ""

    try:
        # Read all lines — we'll scan backward from the end
        with open(transcript_path) as f:
            lines = f.readlines()
    except (OSError, IOError):
        return [], ""

    turns: list[dict[str, str]] = []
    user_intent = ""

    for line in reversed(lines):
        if len(turns) >= MAX_CONV_TURNS and user_intent:
            break
        try:
            obj = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue

        msg = obj.get("message")
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        if role not in ("user", "assistant"):
            continue

        text = _extract_text(msg.get("content", ""))
        if not text:
            continue
        if any(text.startswith(p) for p in _SKIP_PREFIXES):
            continue

        if len(turns) < MAX_CONV_TURNS:
            turns.append({"role": role, "text": text[:MAX_CHARS_PER_TURN]})

        if not user_intent and role == "user":
            user_intent = text[:MAX_INTENT_CHARS]

    turns.reverse()  # chronological order
    return turns, user_intent


def main() -> None:
    # Skip if already proxied via `cross wrap`
    if _is_already_proxied():
        sys.exit(0)

    try:
        raw = sys.stdin.read()
        if not raw.strip():
            sys.exit(0)
        hook_event = json.loads(raw)
    except (json.JSONDecodeError, IOError):
        sys.exit(0)  # Fail open

    tool_name = hook_event.get("tool_name", "")
    tool_input = hook_event.get("tool_input", {})
    session_id = hook_event.get("session_id", "")
    cwd = hook_event.get("cwd", "")
    transcript_path = hook_event.get("transcript_path", "")

    conversation_context, user_intent = _read_transcript(transcript_path)

    payload = json.dumps(
        {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "agent": "claude (desktop)",
            "session_id": session_id,
            "cwd": cwd,
            "conversation_context": conversation_context,
            "user_intent": user_intent,
        }
    ).encode()

    req = urllib.request.Request(
        GATE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
            result = json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        # Daemon unreachable — fail open
        sys.exit(0)

    action = result.get("action", "ALLOW").upper()
    reason = result.get("reason", "")

    if action in ("BLOCK", "HALT_SESSION", "ESCALATE"):
        # Exit 2 = block the tool call; reason on stderr becomes Claude's feedback
        print(f"[cross] {reason}", file=sys.stderr)
        sys.exit(2)

    # ALLOW or unknown — let the tool execute
    sys.exit(0)


if __name__ == "__main__":
    main()
