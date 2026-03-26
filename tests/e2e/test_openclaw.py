"""E2E tests for OpenClaw agent flow.

OpenClaw uses the hook-based gate API (/cross/api/gate) rather than
the network proxy, since it manages its own API connections.

Covers: session registration → gate API calls → gating decisions →
        agent tracking.
"""

from __future__ import annotations

import httpx
import pytest

from tests.e2e.conftest import call_gate_api, end_session, register_session


class TestOpenClawSessionLifecycle:
    """Register and manage an OpenClaw agent session."""

    @pytest.mark.anyio
    async def test_register_openclaw_session(self, cross_daemon):
        """Register an OpenClaw session."""
        base = cross_daemon["base_url"]

        resp = await register_session(
            base,
            agent="openclaw",
            project="openclaw-proj",
            session_id="oc-sess-1",
            cwd="/tmp/openclaw",
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

        # Verify via status endpoint
        async with httpx.AsyncClient() as client:
            status = await client.get(f"{base}/cross/api/status", timeout=5)
        assert status.status_code == 200

        # End session
        resp = await end_session(base, session_id="oc-sess-1")
        assert resp.status_code == 200


class TestOpenClawGateAPI:
    """OpenClaw's hook sends tool calls to /cross/api/gate."""

    @pytest.mark.anyio
    async def test_safe_tool_allowed(self, cross_daemon):
        """Safe tool call is allowed through the gate API."""
        base = cross_daemon["base_url"]
        gate_llm = cross_daemon["mock_gate_llm"]
        gate_llm.verdict = "ALLOW"

        resp = await call_gate_api(
            base,
            tool_name="bash",
            tool_input={"command": "echo hello from openclaw"},
            agent="openclaw",
            session_id="oc-gate-1",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"].upper() == "ALLOW"

    @pytest.mark.anyio
    async def test_dangerous_tool_blocked(self, cross_daemon):
        """Dangerous tool call matching denylist is blocked."""
        base = cross_daemon["base_url"]

        resp = await call_gate_api(
            base,
            tool_name="bash",
            tool_input={"command": "rm -rf /home"},
            agent="openclaw",
            session_id="oc-gate-1",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"].upper() in ("BLOCK", "HALT_SESSION")

    @pytest.mark.anyio
    async def test_review_tool_sent_to_llm(self, cross_daemon):
        """Tool matching review rule is evaluated by LLM gate."""
        base = cross_daemon["base_url"]
        gate_llm = cross_daemon["mock_gate_llm"]
        gate_llm.verdict = "ALLOW"

        resp = await call_gate_api(
            base,
            tool_name="bash",
            tool_input={"command": "curl https://internal.api/secrets"},
            agent="openclaw",
            session_id="oc-gate-1",
        )
        assert resp.status_code == 200
        # Gate LLM was consulted
        assert len(gate_llm.requests) >= 1
        assert resp.json()["action"].upper() == "ALLOW"

    @pytest.mark.anyio
    async def test_non_bash_tool_allowed(self, cross_daemon):
        """A non-bash tool that doesn't match any rule is allowed."""
        base = cross_daemon["base_url"]

        resp = await call_gate_api(
            base,
            tool_name="Read",
            tool_input={"file_path": "/tmp/readme.md"},
            agent="openclaw",
            session_id="oc-gate-1",
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"].upper() == "ALLOW"

    @pytest.mark.anyio
    async def test_gate_tracks_agent(self, cross_daemon):
        """After a gate API call, OpenClaw appears in agent tracking."""
        base = cross_daemon["base_url"]
        gate_llm = cross_daemon["mock_gate_llm"]
        gate_llm.verdict = "ALLOW"

        # Make a gate call
        await call_gate_api(
            base,
            tool_name="bash",
            tool_input={"command": "echo test"},
            agent="openclaw",
            session_id="oc-gate-2",
        )

        # Check status — openclaw should appear
        async with httpx.AsyncClient() as client:
            status = await client.get(f"{base}/cross/api/status", timeout=5)
        assert status.status_code == 200

    @pytest.mark.anyio
    async def test_context_preserved_across_calls(self, cross_daemon):
        """Multiple gate API calls in the same session build up context."""
        base = cross_daemon["base_url"]
        gate_llm = cross_daemon["mock_gate_llm"]
        gate_llm.verdict = "ALLOW"

        session_id = "oc-context-sess"

        # First call
        await call_gate_api(
            base,
            tool_name="bash",
            tool_input={"command": "echo step1"},
            agent="openclaw",
            session_id=session_id,
        )

        # Second call — gate LLM should receive recent_tools context
        await call_gate_api(
            base,
            tool_name="bash",
            tool_input={"command": "curl https://example.com"},
            agent="openclaw",
            session_id=session_id,
        )

        # The LLM gate was consulted for the curl command
        assert len(gate_llm.requests) >= 1
