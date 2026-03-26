#!/usr/bin/env python3
"""cross gate hook for OpenAI Codex CLI (PreToolUse).

Installed as a Codex hook in ~/.codex/hooks.json.
Codex uses a Claude Code-compatible hooks system: PreToolUse events
receive tool call JSON on stdin and block by exiting with code 2.

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


def _should_skip() -> bool:
    """Check if this session should skip gating."""
    # Already proxied via `cross wrap`
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    if "localhost" in base_url or "127.0.0.1" in base_url:
        return True
    # Internal cross LLM call (gate/sentinel subprocess)
    if os.environ.get("CROSS_INTERNAL"):
        return True
    return False


def main() -> None:
    # Skip gating for proxied sessions and internal cross LLM calls
    if _should_skip():
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

    payload = json.dumps(
        {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "agent": "codex",
            "session_id": session_id,
            "cwd": cwd,
            "conversation_context": [],
            "user_intent": "",
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
        # Exit 2 = block the tool call; reason on stderr becomes agent feedback
        print(f"[cross] {reason}", file=sys.stderr)
        sys.exit(2)

    # ALLOW or unknown — let the tool execute
    sys.exit(0)


if __name__ == "__main__":
    main()
