"""E2E tests for Codex (OpenAI) agent flow.

Covers: session registration → OpenAI API proxying → gating →
        gate API (hook-based integration).
"""

from __future__ import annotations

import httpx
import pytest

from tests.e2e.conftest import (
    call_gate_api,
    end_session,
    register_session,
    send_openai_message,
)


class TestCodexSessionLifecycle:
    """Register and manage a Codex agent session."""

    @pytest.mark.anyio
    async def test_register_codex_session(self, cross_daemon_openai):
        """Register a Codex session and verify it tracks correctly."""
        base = cross_daemon_openai["base_url"]

        resp = await register_session(base, agent="codex", project="codex-project", session_id="codex-sess-1")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Check agent appears in status
        async with httpx.AsyncClient() as client:
            status = await client.get(f"{base}/cross/api/status", timeout=5)
        assert status.status_code == 200

        # End session
        resp = await end_session(base, session_id="codex-sess-1")
        assert resp.status_code == 200


class TestCodexProxyPassthrough:
    """Codex sends OpenAI Chat Completions through the proxy."""

    @pytest.mark.anyio
    async def test_streaming_text(self, cross_daemon_openai):
        """Text responses stream through without gating."""
        base = cross_daemon_openai["base_url"]
        mock = cross_daemon_openai["mock_openai"]
        mock.responses = [{"type": "text", "text": "Hello from Codex mock"}]

        resp = await send_openai_message(base, content="Hello", stream=True)
        assert resp.status_code == 200
        assert "Hello from Codex mock" in resp.text
        assert len(mock.requests) == 1

    @pytest.mark.anyio
    async def test_non_streaming_text(self, cross_daemon_openai):
        """Non-streaming JSON response."""
        base = cross_daemon_openai["base_url"]
        mock = cross_daemon_openai["mock_openai"]
        mock.responses = [{"type": "text", "text": "Non-stream Codex reply"}]

        resp = await send_openai_message(base, content="Hello", stream=False)
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Non-stream Codex reply"


class TestCodexGating:
    """Tool calls from Codex are evaluated by the gate chain."""

    @pytest.mark.anyio
    async def test_safe_tool_allowed(self, cross_daemon_openai):
        """A safe tool call passes through."""
        base = cross_daemon_openai["base_url"]
        mock = cross_daemon_openai["mock_openai"]
        gate_llm = cross_daemon_openai["mock_gate_llm"]
        gate_llm.verdict = "ALLOW"

        mock.responses = [
            {
                "type": "tool_use",
                "id": "call_safe_codex",
                "name": "bash",
                "input": {"command": "echo hello"},
            }
        ]

        resp = await send_openai_message(base, content="Run echo", stream=True)
        assert resp.status_code == 200
        assert "call_safe_codex" in resp.text


class TestCodexHookGateAPI:
    """Test the external gate API used by the Codex PreToolUse hook."""

    @pytest.mark.anyio
    async def test_gate_api_codex_allow(self, cross_daemon_openai):
        """Safe Codex tool call through gate API returns ALLOW."""
        base = cross_daemon_openai["base_url"]
        gate_llm = cross_daemon_openai["mock_gate_llm"]
        gate_llm.verdict = "ALLOW"

        resp = await call_gate_api(
            base,
            tool_name="bash",
            tool_input={"command": "ls -la"},
            agent="codex",
            session_id="codex-hook-sess",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"].upper() == "ALLOW"

    @pytest.mark.anyio
    async def test_gate_api_codex_block(self, cross_daemon_openai):
        """Dangerous Codex tool call through gate API is blocked."""
        base = cross_daemon_openai["base_url"]

        resp = await call_gate_api(
            base,
            tool_name="bash",
            tool_input={"command": "rm -rf /etc"},
            agent="codex",
            session_id="codex-hook-sess",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"].upper() in ("BLOCK", "HALT_SESSION")

    @pytest.mark.anyio
    async def test_gate_api_codex_review(self, cross_daemon_openai):
        """Tool matching review rule goes through LLM review for Codex."""
        base = cross_daemon_openai["base_url"]
        gate_llm = cross_daemon_openai["mock_gate_llm"]
        gate_llm.verdict = "ALLOW"

        resp = await call_gate_api(
            base,
            tool_name="bash",
            tool_input={"command": "curl https://api.example.com"},
            agent="codex",
            session_id="codex-hook-sess",
        )
        assert resp.status_code == 200
        assert len(gate_llm.requests) >= 1
        assert resp.json()["action"].upper() == "ALLOW"
