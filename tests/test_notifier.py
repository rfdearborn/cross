"""Tests for the native desktop notification plugin."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cross.events import GateDecisionEvent, SentinelReviewEvent, ToolUseEvent
from cross.plugins import notifier


@pytest.fixture(autouse=True)
def _reset_notifier():
    """Reset module-level state between tests."""
    original = notifier._has_browser_clients
    yield
    notifier._has_browser_clients = original


def _mock_settings(**overrides):
    mock = MagicMock()
    defaults = {"native_notifications_enabled": False}
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


class TestIsAvailable:
    def test_available_when_enabled_and_installed(self):
        with (
            patch.object(notifier, "_TERMINAL_NOTIFIER", "/usr/local/bin/terminal-notifier"),
            patch("cross.config.settings", _mock_settings(native_notifications_enabled=True)),
        ):
            assert notifier.is_available() is True

    def test_unavailable_when_terminal_notifier_missing(self):
        with (
            patch.object(notifier, "_TERMINAL_NOTIFIER", None),
            patch("cross.config.settings", _mock_settings(native_notifications_enabled=True)),
        ):
            assert notifier.is_available() is False

    def test_unavailable_when_disabled_in_config(self):
        with (
            patch.object(notifier, "_TERMINAL_NOTIFIER", "/usr/local/bin/terminal-notifier"),
            patch("cross.config.settings", _mock_settings(native_notifications_enabled=False)),
        ):
            assert notifier.is_available() is False


class TestNotify:
    @patch("subprocess.Popen")
    def test_calls_terminal_notifier(self, mock_popen):
        with patch.object(notifier, "_TERMINAL_NOTIFIER", "/usr/local/bin/terminal-notifier"):
            notifier._notify("test title", "test body")
        mock_popen.assert_called_once()
        args = mock_popen.call_args[0][0]
        assert args[0] == "/usr/local/bin/terminal-notifier"
        assert "-title" in args
        assert "test title" in args
        assert "-message" in args
        assert "test body" in args
        assert "-open" in args
        assert "-group" in args

    @patch("subprocess.Popen")
    def test_skips_when_not_available(self, mock_popen):
        with patch.object(notifier, "_TERMINAL_NOTIFIER", None):
            notifier._notify("title", "body")
        mock_popen.assert_not_called()

    @patch("subprocess.Popen", side_effect=OSError("not found"))
    def test_handles_os_error(self, mock_popen):
        with patch.object(notifier, "_TERMINAL_NOTIFIER", "/usr/local/bin/terminal-notifier"):
            # Should not raise
            notifier._notify("title", "body")


class TestHandleEvent:
    @pytest.mark.anyio
    @patch.object(notifier, "_notify")
    async def test_escalation_triggers_notification(self, mock_notify):
        with patch.object(notifier, "_TERMINAL_NOTIFIER", "/usr/local/bin/terminal-notifier"):
            ev = GateDecisionEvent(
                tool_use_id="tu_1",
                tool_name="Bash",
                action="escalate",
                reason="dangerous command",
            )
            await notifier.handle_event(ev)
        mock_notify.assert_called_once_with("cross — approval needed", "Bash: dangerous command")

    @pytest.mark.anyio
    @patch.object(notifier, "_notify")
    async def test_sentinel_alert_triggers_notification(self, mock_notify):
        with patch.object(notifier, "_TERMINAL_NOTIFIER", "/usr/local/bin/terminal-notifier"):
            ev = SentinelReviewEvent(
                action="alert",
                summary="suspicious activity",
                concerns="credential access",
                event_count=5,
                evaluator="llm_sentinel",
            )
            await notifier.handle_event(ev)
        mock_notify.assert_called_once_with("cross — sentinel alert", "suspicious activity")

    @pytest.mark.anyio
    @patch.object(notifier, "_notify")
    async def test_allow_events_skipped(self, mock_notify):
        ev = GateDecisionEvent(tool_use_id="tu_2", tool_name="Read", action="allow")
        await notifier.handle_event(ev)
        mock_notify.assert_not_called()

    @pytest.mark.anyio
    @patch.object(notifier, "_notify")
    async def test_sentinel_allow_skipped(self, mock_notify):
        ev = SentinelReviewEvent(action="allow", summary="all good")
        await notifier.handle_event(ev)
        mock_notify.assert_not_called()

    @pytest.mark.anyio
    @patch.object(notifier, "_notify")
    async def test_non_gate_events_skipped(self, mock_notify):
        ev = ToolUseEvent(name="Read", tool_use_id="tu_3", input={})
        await notifier.handle_event(ev)
        mock_notify.assert_not_called()


class TestBrowserClientSuppression:
    @pytest.mark.anyio
    @patch.object(notifier, "_notify")
    async def test_suppressed_when_browser_clients_connected(self, mock_notify):
        notifier.set_browser_check(lambda: True)
        ev = GateDecisionEvent(
            tool_use_id="tu_4",
            tool_name="Bash",
            action="escalate",
            reason="blocked",
        )
        await notifier.handle_event(ev)
        mock_notify.assert_not_called()

    @pytest.mark.anyio
    @patch.object(notifier, "_notify")
    async def test_fires_when_no_browser_clients(self, mock_notify):
        with patch.object(notifier, "_TERMINAL_NOTIFIER", "/usr/local/bin/terminal-notifier"):
            notifier.set_browser_check(lambda: False)
            ev = GateDecisionEvent(
                tool_use_id="tu_5",
                tool_name="Bash",
                action="escalate",
                reason="blocked",
            )
            await notifier.handle_event(ev)
        mock_notify.assert_called_once()

    @pytest.mark.anyio
    @patch.object(notifier, "_notify")
    async def test_fires_when_no_check_registered(self, mock_notify):
        with patch.object(notifier, "_TERMINAL_NOTIFIER", "/usr/local/bin/terminal-notifier"):
            notifier._has_browser_clients = None
            ev = GateDecisionEvent(
                tool_use_id="tu_6",
                tool_name="Bash",
                action="escalate",
                reason="blocked",
            )
            await notifier.handle_event(ev)
        mock_notify.assert_called_once()
