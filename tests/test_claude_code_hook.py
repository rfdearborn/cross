"""Tests for the Claude Code PreToolUse hook script."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOK_SCRIPT = str(Path(__file__).parent.parent / "cross" / "patches" / "claude_code_hook.py")


def _run_hook(tool_name: str = "Bash", tool_input: dict | None = None, env_override: dict | None = None):
    """Run the hook script as a subprocess, returning (exit_code, stdout, stderr)."""
    hook_input = json.dumps(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input or {},
            "session_id": "test-session",
            "cwd": "/tmp",
        }
    )

    import os

    env = dict(os.environ)
    # Ensure we're not already proxied (default test state)
    env.pop("ANTHROPIC_BASE_URL", None)
    # Point to a port that's definitely not running cross
    env["CROSS_LISTEN_PORT"] = "19999"
    if env_override:
        env.update(env_override)

    proc = subprocess.run(
        [sys.executable, HOOK_SCRIPT],
        input=hook_input,
        capture_output=True,
        text=True,
        env=env,
        timeout=5,
    )
    return proc.returncode, proc.stdout, proc.stderr


class TestHookSkipIfProxied:
    def test_skips_when_base_url_is_localhost(self):
        exit_code, _, _ = _run_hook(env_override={"ANTHROPIC_BASE_URL": "http://localhost:2767"})
        assert exit_code == 0

    def test_skips_when_base_url_is_127(self):
        exit_code, _, _ = _run_hook(env_override={"ANTHROPIC_BASE_URL": "http://127.0.0.1:2767"})
        assert exit_code == 0


class TestHookFailOpen:
    def test_allows_when_daemon_unreachable(self):
        """When cross daemon is not running, the hook should fail open (allow)."""
        exit_code, _, _ = _run_hook(
            tool_name="Bash",
            tool_input={"command": "ls"},
        )
        assert exit_code == 0

    def test_allows_on_empty_stdin(self):
        """Empty stdin should be handled gracefully."""
        import os

        env = dict(os.environ)
        env.pop("ANTHROPIC_BASE_URL", None)
        env["CROSS_LISTEN_PORT"] = "19999"

        proc = subprocess.run(
            [sys.executable, HOOK_SCRIPT],
            input="",
            capture_output=True,
            text=True,
            env=env,
            timeout=5,
        )
        assert proc.returncode == 0


class TestHookGateIntegration:
    """Tests that require a running cross daemon (integration)."""

    @pytest.fixture
    def gate_server(self):
        """Start a minimal HTTP server that mimics the gate API."""
        import http.server
        import threading

        responses = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                self.rfile.read(length)  # consume body
                resp = responses[0] if responses else {"action": "ALLOW", "reason": "ok"}
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(resp).encode())

            def log_message(self, *args):
                pass  # Suppress logs

        server = http.server.HTTPServer(("localhost", 0), Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        yield server, port, responses
        server.shutdown()

    def test_allows_on_allow_response(self, gate_server):
        server, port, responses = gate_server
        responses.append({"action": "ALLOW", "reason": "ok"})

        exit_code, _, _ = _run_hook(
            tool_name="Bash",
            tool_input={"command": "ls"},
            env_override={"CROSS_LISTEN_PORT": str(port)},
        )
        assert exit_code == 0

    def test_blocks_on_block_response(self, gate_server):
        server, port, responses = gate_server
        responses.append({"action": "BLOCK", "reason": "Dangerous command"})

        exit_code, _, stderr = _run_hook(
            tool_name="Bash",
            tool_input={"command": "rm -rf /"},
            env_override={"CROSS_LISTEN_PORT": str(port)},
        )
        assert exit_code == 2
        assert "Dangerous command" in stderr

    def test_blocks_on_escalate_response(self, gate_server):
        server, port, responses = gate_server
        responses.append({"action": "ESCALATE", "reason": "Needs review"})

        exit_code, _, stderr = _run_hook(
            tool_name="Bash",
            tool_input={"command": "sudo something"},
            env_override={"CROSS_LISTEN_PORT": str(port)},
        )
        assert exit_code == 2
        assert "Needs review" in stderr

    def test_blocks_on_halt_session_response(self, gate_server):
        server, port, responses = gate_server
        responses.append({"action": "HALT_SESSION", "reason": "Session halted"})

        exit_code, _, stderr = _run_hook(
            tool_name="Bash",
            tool_input={"command": "danger"},
            env_override={"CROSS_LISTEN_PORT": str(port)},
        )
        assert exit_code == 2
        assert "Session halted" in stderr

    def test_sends_correct_payload(self, gate_server):
        """Verify the hook sends the right data to the gate API."""
        import http.server
        import threading

        received = []

        class CapturingHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                body = json.loads(self.rfile.read(length))
                received.append(body)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"action": "ALLOW", "reason": "ok"}).encode())

            def log_message(self, *args):
                pass

        server = http.server.HTTPServer(("localhost", 0), CapturingHandler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        try:
            _run_hook(
                tool_name="Write",
                tool_input={"file_path": "/tmp/test.txt", "content": "hello"},
                env_override={"CROSS_LISTEN_PORT": str(port)},
            )
        finally:
            server.shutdown()

        assert len(received) == 1
        payload = received[0]
        assert payload["tool_name"] == "Write"
        assert payload["tool_input"]["file_path"] == "/tmp/test.txt"
        assert payload["agent"] == "claude (desktop)"
        assert payload["session_id"] == "test-session"
