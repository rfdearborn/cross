"""Tests for the dashboard plugin and HTTP endpoints."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cross.events import (
    GateDecisionEvent,
    RequestEvent,
    SentinelReviewEvent,
    ToolUseEvent,
)
from cross.plugins.dashboard import DashboardPlugin, _event_to_dict

# ---------------------------------------------------------------------------
# DashboardPlugin unit tests
# ---------------------------------------------------------------------------


class TestEventToDict:
    """Test the _event_to_dict helper."""

    def test_tool_use_event(self):
        ev = ToolUseEvent(name="Bash", tool_use_id="tu_1", input={"command": "ls"})
        d = _event_to_dict(ev)
        assert d["event_type"] == "ToolUseEvent"
        assert d["name"] == "Bash"
        assert d["tool_use_id"] == "tu_1"
        assert "ts" in d

    def test_gate_decision_event(self):
        ev = GateDecisionEvent(
            tool_use_id="tu_2",
            tool_name="Write",
            action="block",
            reason="dangerous",
        )
        d = _event_to_dict(ev)
        assert d["event_type"] == "GateDecisionEvent"
        assert d["action"] == "block"

    def test_request_event_strips_raw_body(self):
        ev = RequestEvent(method="POST", path="/v1/messages", raw_body={"big": "data"})
        d = _event_to_dict(ev)
        assert "raw_body" not in d
        assert d["method"] == "POST"

    def test_sentinel_review_event(self):
        ev = SentinelReviewEvent(
            action="alert",
            summary="looks fine",
            concerns="none",
            event_count=5,
            evaluator="llm_sentinel",
        )
        d = _event_to_dict(ev)
        assert d["event_type"] == "SentinelReviewEvent"
        assert d["event_count"] == 5


class TestDashboardPluginEventHandling:
    """Test event handling, pending tracking, and resolution."""

    @pytest.mark.anyio
    async def test_events_stored(self):
        plugin = DashboardPlugin()
        ev = ToolUseEvent(name="Read", tool_use_id="tu_10", input={})
        await plugin.handle_event(ev)

        events = plugin.get_events()
        assert len(events) == 1
        assert events[0]["name"] == "Read"

    @pytest.mark.anyio
    async def test_max_events_bounded(self):
        plugin = DashboardPlugin()
        for i in range(150):
            ev = ToolUseEvent(name=f"tool_{i}", tool_use_id=f"tu_{i}", input={})
            await plugin.handle_event(ev)

        events = plugin.get_events()
        assert len(events) == 100  # bounded by _MAX_EVENTS

    @pytest.mark.anyio
    async def test_escalate_adds_to_pending(self):
        plugin = DashboardPlugin()
        ev = GateDecisionEvent(
            tool_use_id="tu_esc",
            tool_name="Bash",
            action="escalate",
            reason="needs review",
            tool_input={"command": "rm -rf /"},
        )
        await plugin.handle_event(ev)

        pending = plugin.get_pending()
        assert len(pending) == 1
        assert pending[0]["tool_use_id"] == "tu_esc"

    @pytest.mark.anyio
    async def test_allow_removes_from_pending(self):
        plugin = DashboardPlugin()
        # First escalate
        await plugin.handle_event(GateDecisionEvent(tool_use_id="tu_res", tool_name="Bash", action="escalate"))
        assert len(plugin.get_pending()) == 1

        # Then allow
        await plugin.handle_event(GateDecisionEvent(tool_use_id="tu_res", tool_name="Bash", action="allow"))
        assert len(plugin.get_pending()) == 0

    @pytest.mark.anyio
    async def test_block_removes_from_pending(self):
        plugin = DashboardPlugin()
        await plugin.handle_event(GateDecisionEvent(tool_use_id="tu_blk", tool_name="Bash", action="escalate"))
        await plugin.handle_event(GateDecisionEvent(tool_use_id="tu_blk", tool_name="Bash", action="block"))
        assert len(plugin.get_pending()) == 0

    @pytest.mark.anyio
    async def test_non_escalate_gate_not_pending(self):
        plugin = DashboardPlugin()
        await plugin.handle_event(GateDecisionEvent(tool_use_id="tu_ok", tool_name="Read", action="allow"))
        assert len(plugin.get_pending()) == 0

    @pytest.mark.anyio
    async def test_resolve_calls_callback(self):
        callback = MagicMock()
        plugin = DashboardPlugin(resolve_approval_callback=callback)

        # Add a pending escalation
        await plugin.handle_event(GateDecisionEvent(tool_use_id="tu_cb", tool_name="Bash", action="escalate"))

        plugin.resolve("tu_cb", True, "testuser")

        callback.assert_called_once_with("tu_cb", True, "testuser")
        assert len(plugin.get_pending()) == 0

    @pytest.mark.anyio
    async def test_resolve_without_callback(self):
        plugin = DashboardPlugin()
        await plugin.handle_event(GateDecisionEvent(tool_use_id="tu_nc", tool_name="Bash", action="escalate"))
        # Should not raise even without callback
        plugin.resolve("tu_nc", False)
        assert len(plugin.get_pending()) == 0

    @pytest.mark.anyio
    async def test_broadcast_sends_to_ws_clients(self):
        plugin = DashboardPlugin()
        mock_ws = AsyncMock()
        plugin._ws_clients.add(mock_ws)

        ev = ToolUseEvent(name="Edit", tool_use_id="tu_bc", input={})
        await plugin.handle_event(ev)

        mock_ws.send_text.assert_awaited_once()
        sent_data = json.loads(mock_ws.send_text.call_args[0][0])
        assert sent_data["name"] == "Edit"

    @pytest.mark.anyio
    async def test_broadcast_removes_failed_clients(self):
        plugin = DashboardPlugin()
        good_ws = AsyncMock()
        bad_ws = AsyncMock()
        bad_ws.send_text.side_effect = RuntimeError("closed")
        plugin._ws_clients.add(good_ws)
        plugin._ws_clients.add(bad_ws)

        ev = ToolUseEvent(name="Read", tool_use_id="tu_fail", input={})
        await plugin.handle_event(ev)

        assert bad_ws not in plugin._ws_clients
        assert good_ws in plugin._ws_clients

    @pytest.mark.anyio
    async def test_ws_handler_accepts_and_cleans_up(self):
        from starlette.websockets import WebSocketDisconnect

        plugin = DashboardPlugin()
        mock_ws = AsyncMock()
        mock_ws.receive_text = AsyncMock(side_effect=WebSocketDisconnect())

        await plugin.ws_handler(mock_ws)

        mock_ws.accept.assert_awaited_once()
        assert mock_ws not in plugin._ws_clients


# ---------------------------------------------------------------------------
# Daemon HTTP route tests
# ---------------------------------------------------------------------------


def _mock_settings(**overrides):
    defaults = {
        "gating_enabled": False,
        "rules_dir": "~/.cross/rules.d",
        "llm_gate_enabled": False,
        "llm_gate_model": "anthropic/claude-haiku-4-5",
        "llm_gate_api_key": "",
        "llm_gate_base_url": "",
        "llm_gate_temperature": 0.0,
        "llm_gate_max_tokens": 256,
        "llm_gate_reasoning": "",
        "llm_gate_timeout_ms": 30000,
        "llm_gate_threshold": "block",
        "llm_sentinel_enabled": False,
        "llm_sentinel_model": "anthropic/claude-sonnet-4-6",
        "llm_sentinel_api_key": "",
        "llm_sentinel_base_url": "",
        "llm_sentinel_temperature": 0.0,
        "llm_sentinel_max_tokens": 1024,
        "llm_sentinel_reasoning": "medium",
        "llm_sentinel_interval_seconds": 60,
        "slack_bot_token": "",
        "slack_app_token": "",
        "listen_port": 2767,
        "log_file": "/dev/null",
    }
    defaults.update(overrides)
    mock = MagicMock()
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


class TestDashboardRoutes:
    """Test the dashboard HTTP route handlers in daemon.py."""

    @pytest.mark.anyio
    async def test_dashboard_page_returns_html(self):
        import cross.daemon as daemon

        mock_request = AsyncMock()
        response = await daemon.dashboard_page(mock_request)
        assert response.status_code == 200
        assert response.media_type == "text/html"
        assert b"cross" in response.body

    @pytest.mark.anyio
    async def test_api_events_with_dashboard(self):
        import cross.daemon as daemon

        plugin = DashboardPlugin()
        await plugin.handle_event(ToolUseEvent(name="Read", tool_use_id="tu_e1", input={}))
        daemon._dashboard = plugin

        mock_request = AsyncMock()
        response = await daemon.api_events(mock_request)
        assert response.status_code == 200
        body = json.loads(response.body)
        assert len(body) == 1
        assert body[0]["name"] == "Read"

    @pytest.mark.anyio
    async def test_api_events_without_dashboard(self):
        import cross.daemon as daemon

        daemon._dashboard = None
        mock_request = AsyncMock()
        response = await daemon.api_events(mock_request)
        assert response.status_code == 200
        assert json.loads(response.body) == []

    @pytest.mark.anyio
    async def test_api_pending_with_escalation(self):
        import cross.daemon as daemon

        plugin = DashboardPlugin()
        await plugin.handle_event(
            GateDecisionEvent(tool_use_id="tu_p1", tool_name="Bash", action="escalate", reason="dangerous")
        )
        daemon._dashboard = plugin

        mock_request = AsyncMock()
        response = await daemon.api_pending(mock_request)
        assert response.status_code == 200
        body = json.loads(response.body)
        assert len(body) == 1
        assert body[0]["tool_use_id"] == "tu_p1"

    @pytest.mark.anyio
    async def test_api_pending_empty(self):
        import cross.daemon as daemon

        daemon._dashboard = DashboardPlugin()
        mock_request = AsyncMock()
        response = await daemon.api_pending(mock_request)
        assert response.status_code == 200
        assert json.loads(response.body) == []

    @pytest.mark.anyio
    async def test_api_resolve_pending_approve(self):
        import cross.daemon as daemon

        callback = MagicMock()
        plugin = DashboardPlugin(resolve_approval_callback=callback)
        await plugin.handle_event(GateDecisionEvent(tool_use_id="tu_r1", tool_name="Bash", action="escalate"))
        daemon._dashboard = plugin

        mock_request = AsyncMock()
        mock_request.path_params = {"tool_use_id": "tu_r1"}
        mock_request.json.return_value = {"approved": True, "username": "alice"}

        response = await daemon.api_resolve_pending(mock_request)
        assert response.status_code == 200
        body = json.loads(response.body)
        assert body["approved"] is True

        callback.assert_called_once_with("tu_r1", True, "alice")

    @pytest.mark.anyio
    async def test_api_resolve_pending_deny(self):
        import cross.daemon as daemon

        callback = MagicMock()
        plugin = DashboardPlugin(resolve_approval_callback=callback)
        await plugin.handle_event(GateDecisionEvent(tool_use_id="tu_r2", tool_name="Write", action="escalate"))
        daemon._dashboard = plugin

        mock_request = AsyncMock()
        mock_request.path_params = {"tool_use_id": "tu_r2"}
        mock_request.json.return_value = {"approved": False, "username": "bob"}

        response = await daemon.api_resolve_pending(mock_request)
        assert response.status_code == 200
        body = json.loads(response.body)
        assert body["approved"] is False

        callback.assert_called_once_with("tu_r2", False, "bob")

    @pytest.mark.anyio
    async def test_api_resolve_pending_no_dashboard(self):
        import cross.daemon as daemon

        daemon._dashboard = None
        mock_request = AsyncMock()
        mock_request.path_params = {"tool_use_id": "tu_none"}
        mock_request.json.return_value = {"approved": True}

        response = await daemon.api_resolve_pending(mock_request)
        assert response.status_code == 500


class TestDashboardRouteRegistration:
    """Test that dashboard routes are registered in the Starlette app."""

    def test_dashboard_routes_present(self):
        from cross.daemon import app

        paths = [r.path for r in app.routes]
        assert "/cross/dashboard" in paths
        assert "/cross/api/events" in paths
        assert "/cross/api/pending" in paths
        assert "/cross/api/pending/{tool_use_id}/resolve" in paths

    def test_dashboard_ws_route_present(self):
        from cross.daemon import app

        paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/cross/api/ws" in paths

    def test_proxy_catchall_still_last(self):
        from cross.daemon import app

        last_route = app.routes[-1]
        assert last_route.path == "/{path:path}"


class TestDashboardStartup:
    """Test that dashboard plugin is initialized on startup."""

    @pytest.mark.anyio
    async def test_dashboard_always_active(self):
        """Dashboard should be initialized even without Slack tokens."""
        import cross.daemon as daemon

        daemon._gate_chain = None
        daemon._sentinel = None
        daemon._slack = None
        daemon._dashboard = None

        with (
            patch("cross.daemon.event_bus") as mock_bus,
            patch("cross.daemon.LoggerPlugin") as mock_logger_cls,
            patch("cross.daemon.settings", _mock_settings()),
            patch("cross.proxy.resolve_gate_approval"),
        ):
            mock_logger_cls.return_value = MagicMock()
            mock_bus.subscribe = MagicMock()
            await daemon.on_startup()

        assert daemon._dashboard is not None
        assert isinstance(daemon._dashboard, DashboardPlugin)


# ---------------------------------------------------------------------------
# CLI pending command tests
# ---------------------------------------------------------------------------


class TestPendingCLI:
    """Test the cross pending CLI command."""

    def test_pending_list_empty(self):
        from cross.cli import _run_pending

        args = MagicMock()
        args.pending_action = None

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = []

        with patch("cross.cli.httpx.get", return_value=mock_resp):
            result = _run_pending(args)

        assert result == 0

    def test_pending_list_with_items(self, capsys):
        from cross.cli import _run_pending

        args = MagicMock()
        args.pending_action = None

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [
            {
                "tool_use_id": "tu_cli1",
                "tool_name": "Bash",
                "reason": "dangerous command",
                "tool_input": {"command": "rm -rf /"},
            }
        ]

        with patch("cross.cli.httpx.get", return_value=mock_resp):
            result = _run_pending(args)

        assert result == 0
        output = capsys.readouterr().out
        assert "tu_cli1" in output
        assert "Bash" in output
        assert "dangerous command" in output

    def test_pending_daemon_not_running(self, capsys):
        from cross.cli import _run_pending

        args = MagicMock()
        args.pending_action = None

        with patch("cross.cli.httpx.get", side_effect=MagicMock(side_effect=__import__("httpx").ConnectError(""))):
            result = _run_pending(args)

        assert result == 1
        assert "daemon not running" in capsys.readouterr().err

    def test_pending_approve(self, capsys):
        from cross.cli import _run_pending

        args = MagicMock()
        args.pending_action = "approve"
        args.tool_use_id = "tu_approve"

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("cross.cli.httpx.post", return_value=mock_resp):
            result = _run_pending(args)

        assert result == 0
        assert "Approved" in capsys.readouterr().out

    def test_pending_deny(self, capsys):
        from cross.cli import _run_pending

        args = MagicMock()
        args.pending_action = "deny"
        args.tool_use_id = "tu_deny"

        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch("cross.cli.httpx.post", return_value=mock_resp):
            result = _run_pending(args)

        assert result == 0
        assert "Denied" in capsys.readouterr().out

    def test_pending_resolve_daemon_not_running(self, capsys):
        from cross.cli import _run_pending

        args = MagicMock()
        args.pending_action = "approve"
        args.tool_use_id = "tu_fail"

        with patch("cross.cli.httpx.post", side_effect=__import__("httpx").ConnectError("")):
            result = _run_pending(args)

        assert result == 1
        assert "daemon not running" in capsys.readouterr().err

    def test_pending_resolve_error_response(self, capsys):
        from cross.cli import _run_pending

        args = MagicMock()
        args.pending_action = "approve"
        args.tool_use_id = "tu_err"

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch("cross.cli.httpx.post", return_value=mock_resp):
            result = _run_pending(args)

        assert result == 1
