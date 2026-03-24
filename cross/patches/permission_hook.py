#!/usr/bin/env python3
"""cross permission hook for Claude Code (PermissionRequest).

Installed as a Claude Code hook in ~/.claude/settings.json.
When a permission prompt appears, notifies the cross daemon so it can
relay to Slack/email/dashboard after a delay (giving the user time to
approve in terminal first).

Does NOT control the permission decision — just notifies.
"""

import json
import os
import sys
import urllib.error
import urllib.request

CROSS_PORT = os.environ.get("CROSS_LISTEN_PORT", "2767")


def _get_cross_session_id() -> str:
    """Extract the cross session_id from ANTHROPIC_BASE_URL.

    cross wrap embeds the session_id in the URL: http://localhost:PORT/s/{session_id}
    Falls back to the Claude Code session_id if not wrapped.
    """
    base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
    if "/s/" in base_url:
        return base_url.split("/s/")[1].rstrip("/")
    return ""


def main() -> None:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            sys.exit(0)
        hook_event = json.loads(raw)
    except (json.JSONDecodeError, IOError):
        sys.exit(0)

    # Use the cross session_id (from ANTHROPIC_BASE_URL), not the Claude Code UUID
    session_id = _get_cross_session_id() or hook_event.get("session_id", "")
    tool_name = hook_event.get("tool_name", "")
    tool_input = hook_event.get("tool_input", {})

    # Build a tool description matching the format used elsewhere in cross
    tool_desc = ""
    if tool_name == "Bash":
        cmd = tool_input.get("command", "") if isinstance(tool_input, dict) else ""
        tool_desc = f"`Bash`: `{cmd[:80]}`"
    elif tool_name in ("Read", "Write", "Edit"):
        path = tool_input.get("file_path", "") if isinstance(tool_input, dict) else ""
        tool_desc = f"`{tool_name}`: `{path}`"
    elif tool_name:
        tool_desc = f"`{tool_name}`"

    url = f"http://localhost:{CROSS_PORT}/cross/sessions/{session_id}/permission"
    payload = json.dumps(
        {
            "tool_name": tool_name,
            "tool_input": tool_input,
            "tool_desc": tool_desc,
        }
    ).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        urllib.request.urlopen(req, timeout=5)
    except (urllib.error.URLError, OSError):
        pass  # Daemon unreachable — don't block the permission prompt

    # Always exit 0 — this hook is notification-only, doesn't control decisions
    sys.exit(0)


if __name__ == "__main__":
    main()
