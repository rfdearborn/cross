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


def _is_already_proxied() -> bool:
    """Check if this session is already routed through cross's proxy."""
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
    return "localhost" in base_url or "127.0.0.1" in base_url


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

    payload = json.dumps(
        {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "agent": "claude (desktop)",
            "session_id": session_id,
            "cwd": cwd,
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
