"""Tests for the Claude Code PreToolUse hook script."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

HOOK_SCRIPT = str(Path(__file__).parent.parent / "cross" / "patches" / "claude_code_hook.py")

# Import hook functions for unit testing — path manipulation must precede import
sys.path.insert(0, str(Path(__file__).parent.parent / "cross" / "patches"))
from claude_code_hook import _extract_text, _read_transcript  # noqa: E402


def _run_hook(
    tool_name: str = "Bash",
    tool_input: dict | None = None,
    env_override: dict | None = None,
    transcript_path: str = "",
):
    """Run the hook script as a subprocess, returning (exit_code, stdout, stderr)."""
    hook_input = json.dumps(
        {
            "hook_event_name": "PreToolUse",
            "tool_name": tool_name,
            "tool_input": tool_input or {},
            "session_id": "test-session",
            "cwd": "/tmp",
            "transcript_path": transcript_path,
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


class TestHookSkip:
    def test_skips_when_base_url_is_localhost(self):
        exit_code, _, _ = _run_hook(env_override={"ANTHROPIC_BASE_URL": "http://localhost:2767"})
        assert exit_code == 0

    def test_skips_when_base_url_is_127(self):
        exit_code, _, _ = _run_hook(env_override={"ANTHROPIC_BASE_URL": "http://127.0.0.1:2767"})
        assert exit_code == 0

    def test_skips_when_cross_internal(self):
        exit_code, _, _ = _run_hook(env_override={"CROSS_INTERNAL": "1"})
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
        # Without a transcript, context fields should be empty
        assert payload["conversation_context"] == []
        assert payload["user_intent"] == ""

    def test_sends_conversation_context_from_transcript(self, gate_server):
        """Verify the hook extracts and sends conversation context from transcript."""
        import http.server
        import tempfile
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

        # Create a fake transcript JSONL
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            transcript_path = f.name
            msg = {"type": "user", "message": {"role": "user", "content": "Please fix the bug in auth.py"}}
            f.write(json.dumps(msg) + "\n")
            msg = {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I'll look at auth.py now."}],
                },
            }
            f.write(json.dumps(msg) + "\n")
            msg = {"type": "user", "message": {"role": "user", "content": "Yes go ahead"}}
            f.write(json.dumps(msg) + "\n")

        try:
            _run_hook(
                tool_name="Read",
                tool_input={"file_path": "/tmp/auth.py"},
                env_override={"CROSS_LISTEN_PORT": str(port)},
                transcript_path=transcript_path,
            )
        finally:
            server.shutdown()
            os.unlink(transcript_path)

        assert len(received) == 1
        payload = received[0]
        assert payload["user_intent"] == "Yes go ahead"
        assert len(payload["conversation_context"]) == 3
        assert payload["conversation_context"][0]["role"] == "user"
        assert "fix the bug" in payload["conversation_context"][0]["text"]
        assert payload["conversation_context"][1]["role"] == "assistant"
        assert payload["conversation_context"][2]["role"] == "user"


class TestExtractText:
    def test_string_content(self):
        assert _extract_text("hello world") == "hello world"

    def test_list_content_with_text_blocks(self):
        content = [
            {"type": "text", "text": "Part one."},
            {"type": "tool_use", "id": "t1"},
            {"type": "text", "text": "Part two."},
        ]
        assert _extract_text(content) == "Part one. Part two."

    def test_empty_list(self):
        assert _extract_text([]) == ""

    def test_non_text_blocks_only(self):
        content = [{"type": "tool_result", "content": "ok"}]
        assert _extract_text(content) == ""

    def test_none_returns_empty(self):
        assert _extract_text(None) == ""


class TestReadTranscript:
    def _write_transcript(self, messages):
        """Write a list of (role, content) tuples as a JSONL transcript file."""
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for role, content in messages:
            line = json.dumps({"type": role, "message": {"role": role, "content": content}})
            f.write(line + "\n")
        f.close()
        return f.name

    def test_extracts_recent_turns(self):
        path = self._write_transcript(
            [
                ("user", "First message"),
                ("assistant", "First reply"),
                ("user", "Second message"),
                ("assistant", "Second reply"),
            ]
        )
        try:
            turns, intent = _read_transcript(path)
            assert len(turns) == 4
            assert turns[0]["text"] == "First message"
            assert turns[-1]["text"] == "Second reply"
            assert intent == "Second message"
        finally:
            os.unlink(path)

    def test_caps_at_max_turns(self):
        messages = []
        for i in range(20):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append((role, f"Message {i}"))
        path = self._write_transcript(messages)
        try:
            turns, intent = _read_transcript(path)
            assert len(turns) == 5  # MAX_CONV_TURNS
        finally:
            os.unlink(path)

    def test_truncates_long_text(self):
        long_text = "x" * 1000
        path = self._write_transcript([("user", long_text)])
        try:
            turns, intent = _read_transcript(path)
            assert len(turns[0]["text"]) == 300  # MAX_CHARS_PER_TURN
            assert len(intent) == 500  # MAX_INTENT_CHARS
        finally:
            os.unlink(path)

    def test_skips_system_reminders(self):
        path = self._write_transcript(
            [
                ("user", "<system-reminder>some system text</system-reminder>"),
                ("user", "Real message"),
            ]
        )
        try:
            turns, intent = _read_transcript(path)
            assert len(turns) == 1
            assert turns[0]["text"] == "Real message"
            assert intent == "Real message"
        finally:
            os.unlink(path)

    def test_handles_missing_file(self):
        turns, intent = _read_transcript("/nonexistent/path.jsonl")
        assert turns == []
        assert intent == ""

    def test_handles_empty_path(self):
        turns, intent = _read_transcript("")
        assert turns == []
        assert intent == ""

    def test_handles_array_content(self):
        content = [{"type": "text", "text": "Array content message"}]
        path = self._write_transcript([("user", content)])
        try:
            turns, intent = _read_transcript(path)
            assert turns[0]["text"] == "Array content message"
            assert intent == "Array content message"
        finally:
            os.unlink(path)

    def test_skips_non_message_lines(self):
        """Lines without a message.role of user/assistant should be skipped."""
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        f.write(json.dumps({"type": "system", "message": {"role": "system", "content": "You are helpful"}}) + "\n")
        f.write(json.dumps({"type": "custom-title", "message": "some title"}) + "\n")
        f.write(json.dumps({"type": "user", "message": {"role": "user", "content": "Hello"}}) + "\n")
        f.close()
        try:
            turns, intent = _read_transcript(f.name)
            assert len(turns) == 1
            assert turns[0]["text"] == "Hello"
        finally:
            os.unlink(f.name)
