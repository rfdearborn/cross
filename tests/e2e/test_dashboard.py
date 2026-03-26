"""E2E tests for the web dashboard.

Covers: dashboard page load → status API → events API →
        pending approvals → WebSocket event stream →
        instructions API → conversation API.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest
import websockets

from tests.e2e.conftest import (
    call_gate_api,
    send_anthropic_message,
)


class TestDashboardPages:
    """HTML pages are served correctly."""

    @pytest.mark.anyio
    async def test_root_redirects_to_dashboard(self, cross_daemon):
        base = cross_daemon["base_url"]
        async with httpx.AsyncClient(follow_redirects=False) as client:
            resp = await client.get(f"{base}/", timeout=5)
        assert resp.status_code in (301, 302, 307)
        assert "/cross/dashboard" in resp.headers.get("location", "")

    @pytest.mark.anyio
    async def test_dashboard_page_loads(self, cross_daemon):
        base = cross_daemon["base_url"]
        async with httpx.AsyncClient(follow_redirects=True) as client:
            resp = await client.get(f"{base}/cross/dashboard", timeout=5)
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "cross" in resp.text.lower()

    @pytest.mark.anyio
    async def test_settings_page_loads(self, cross_daemon):
        base = cross_daemon["base_url"]
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base}/cross/settings", timeout=5)
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")


class TestDashboardAPIs:
    """REST API endpoints for the dashboard."""

    @pytest.mark.anyio
    async def test_status_endpoint(self, cross_daemon):
        base = cross_daemon["base_url"]
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base}/cross/api/status", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert "monitored" in data or "agents" in data or "status" in data

    @pytest.mark.anyio
    async def test_events_endpoint(self, cross_daemon):
        """Events endpoint returns historical events."""
        base = cross_daemon["base_url"]

        # Generate some events first
        mock = cross_daemon["mock_anthropic"]
        mock.responses = [{"type": "text", "text": "Test event"}]
        await send_anthropic_message(base, content="Generate event", stream=True)

        # Small delay for event processing
        await asyncio.sleep(0.2)

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base}/cross/api/events", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    @pytest.mark.anyio
    async def test_pending_endpoint_empty(self, cross_daemon):
        """No pending approvals initially."""
        base = cross_daemon["base_url"]
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base}/cross/api/pending", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, (list, dict))

    @pytest.mark.anyio
    async def test_pending_permissions_endpoint_empty(self, cross_daemon):
        base = cross_daemon["base_url"]
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base}/cross/api/pending-permissions", timeout=5)
        assert resp.status_code == 200

    @pytest.mark.anyio
    async def test_instructions_get_and_put(self, cross_daemon):
        """Get and set custom instructions via API."""
        base = cross_daemon["base_url"]

        async with httpx.AsyncClient() as client:
            # GET initial
            resp = await client.get(f"{base}/cross/api/instructions", timeout=5)
            assert resp.status_code == 200

            # PUT new instructions
            resp = await client.put(
                f"{base}/cross/api/instructions",
                json={"content": "Be extra careful with file deletions."},
                timeout=5,
            )
            assert resp.status_code == 200

            # GET updated
            resp = await client.get(f"{base}/cross/api/instructions", timeout=5)
            assert resp.status_code == 200


class TestDashboardWebSocket:
    """WebSocket event stream delivers real-time events."""

    @pytest.mark.anyio
    async def test_ws_receives_events(self, cross_daemon):
        """Connect to the WS and verify events arrive when traffic flows."""
        base = cross_daemon["base_url"]
        ws_url = base.replace("http://", "ws://") + "/cross/api/ws"

        received = []

        async def listen():
            async with websockets.connect(ws_url) as ws:
                # Collect events for a short period
                try:
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=2)
                        received.append(json.loads(msg))
                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                    pass

        # Start listener
        listener = asyncio.create_task(listen())

        # Wait for WS to connect
        await asyncio.sleep(0.3)

        # Generate traffic
        mock = cross_daemon["mock_anthropic"]
        mock.responses = [{"type": "text", "text": "WS test event"}]
        await send_anthropic_message(base, content="WS test", stream=True)

        # Wait for events to arrive
        await asyncio.sleep(0.5)
        await listener

        # Should have received at least a RequestEvent
        assert len(received) >= 1
        event_types = {e.get("event_type") for e in received}
        assert "RequestEvent" in event_types


class TestDashboardAfterGating:
    """Dashboard reflects gating decisions."""

    @pytest.mark.anyio
    async def test_events_include_gate_decisions(self, cross_daemon):
        """After a gate call, events endpoint shows the decision."""
        base = cross_daemon["base_url"]
        gate_llm = cross_daemon["mock_gate_llm"]
        gate_llm.verdict = "ALLOW"

        await call_gate_api(
            base,
            tool_name="bash",
            tool_input={"command": "curl https://example.com"},
            agent="claude",
        )

        await asyncio.sleep(0.3)

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base}/cross/api/events", timeout=5)
        assert resp.status_code == 200
        events = resp.json()
        gate_events = [e for e in events if e.get("event_type") == "GateDecisionEvent"]
        assert len(gate_events) >= 1

    @pytest.mark.anyio
    async def test_events_include_tool_use(self, cross_daemon):
        """Tool use events appear in the event feed."""
        base = cross_daemon["base_url"]
        gate_llm = cross_daemon["mock_gate_llm"]
        gate_llm.verdict = "ALLOW"

        await call_gate_api(
            base,
            tool_name="Write",
            tool_input={"file_path": "/tmp/out.txt", "content": "hello"},
            agent="openclaw",
        )

        await asyncio.sleep(0.3)

        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{base}/cross/api/events", timeout=5)
        events = resp.json()
        tool_events = [e for e in events if e.get("event_type") == "ToolUseEvent"]
        assert len(tool_events) >= 1
        assert tool_events[-1]["name"] == "Write"
