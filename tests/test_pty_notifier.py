"""Tests for PTY terminal notification plugin."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from cross.events import GateDecisionEvent, SentinelReviewEvent, TextEvent, ToolUseEvent
from cross.plugins import pty_notifier


@pytest.fixture(autouse=True)
def _reset_plugin(monkeypatch):
    """Reset plugin state and wire up mock callbacks for each test."""
    pty_notifier._recent.clear()
    pty_notifier._notify_callback = None
    pty_notifier._prompt_callback = None
    monkeypatch.setattr("cross.plugins.pty_notifier.settings.pty_notifications_enabled", True)
    yield


@pytest.fixture
def mock_notify():
    cb = AsyncMock()
    pty_notifier.set_notify_callback(cb)
    return cb


@pytest.fixture
def mock_prompt():
    cb = AsyncMock()
    pty_notifier.set_prompt_callback(cb)
    return cb


# --- Gate ESCALATE → interactive prompt ---


class TestGateEscalatePrompt:
    @pytest.mark.anyio
    async def test_escalate_sends_prompt(self, mock_notify, mock_prompt):
        event = GateDecisionEvent(
            tool_use_id="t1",
            tool_name="Bash",
            action="escalate",
            reason="needs review",
            tool_input={"command": "rm -rf /"},
            session_id="s1",
        )
        await pty_notifier.handle_event(event)
        # Should use prompt, not notify
        mock_prompt.assert_awaited_once()
        mock_notify.assert_not_awaited()

    @pytest.mark.anyio
    async def test_escalate_prompt_data(self, mock_prompt):
        event = GateDecisionEvent(
            tool_use_id="t1",
            tool_name="Bash",
            action="escalate",
            reason="test rule",
            tool_input={"command": "echo hello"},
            session_id="s1",
        )
        await pty_notifier.handle_event(event)
        args = mock_prompt.call_args[0]
        assert args[0] == "s1"
        prompt_data = args[1]
        assert prompt_data["type"] == "prompt"
        assert prompt_data["prompt_id"] == "t1"
        assert "Bash" in prompt_data["title"]
        assert "echo hello" in prompt_data["body"]
        assert "test rule" in prompt_data["body"]
        assert prompt_data["options"] == ["Allow", "Deny"]

    @pytest.mark.anyio
    async def test_escalate_without_prompt_callback_skips(self, mock_notify):
        """If no prompt callback is set, escalate events are silently skipped."""
        event = GateDecisionEvent(
            tool_use_id="t1",
            tool_name="Bash",
            action="escalate",
            reason="test",
            session_id="s1",
        )
        await pty_notifier.handle_event(event)
        mock_notify.assert_not_awaited()


# --- Gate non-escalate → OSC 9 notification ---


class TestGateNotifications:
    @pytest.mark.anyio
    async def test_block_fires_notification(self, mock_notify):
        event = GateDecisionEvent(
            tool_use_id="t1",
            tool_name="Bash",
            action="block",
            reason="dangerous command",
            session_id="s1",
        )
        await pty_notifier.handle_event(event)
        mock_notify.assert_awaited_once()
        args = mock_notify.call_args
        assert args[0][0] == "s1"
        assert "Gate BLOCK" in args[0][1]
        assert "Bash" in args[0][1]

    @pytest.mark.anyio
    async def test_halt_session_fires_notification(self, mock_notify):
        event = GateDecisionEvent(
            tool_use_id="t3",
            tool_name="Bash",
            action="halt_session",
            reason="session halted",
            session_id="s1",
        )
        await pty_notifier.handle_event(event)
        mock_notify.assert_awaited_once()
        assert "Gate HALT_SESSION" in mock_notify.call_args[0][1]

    @pytest.mark.anyio
    async def test_alert_fires_notification(self, mock_notify):
        event = GateDecisionEvent(
            tool_use_id="t4",
            tool_name="Bash",
            action="alert",
            reason="suspicious",
            session_id="s1",
        )
        await pty_notifier.handle_event(event)
        mock_notify.assert_awaited_once()
        assert "Gate ALERT" in mock_notify.call_args[0][1]

    @pytest.mark.anyio
    async def test_allow_skipped(self, mock_notify, mock_prompt):
        event = GateDecisionEvent(
            tool_use_id="t5",
            tool_name="Bash",
            action="allow",
            session_id="s1",
        )
        await pty_notifier.handle_event(event)
        mock_notify.assert_not_awaited()
        mock_prompt.assert_not_awaited()

    @pytest.mark.anyio
    async def test_abstain_skipped(self, mock_notify, mock_prompt):
        event = GateDecisionEvent(
            tool_use_id="t6",
            tool_name="Bash",
            action="abstain",
            session_id="s1",
        )
        await pty_notifier.handle_event(event)
        mock_notify.assert_not_awaited()
        mock_prompt.assert_not_awaited()


# --- Sentinel events ---


class TestSentinelNotifications:
    @pytest.mark.anyio
    async def test_escalate_fires(self, mock_notify):
        event = SentinelReviewEvent(
            action="escalate",
            summary="suspicious pattern",
            session_id="s1",
            review_id="r1",
        )
        await pty_notifier.handle_event(event)
        mock_notify.assert_awaited_once()
        assert "Sentinel ESCALATE" in mock_notify.call_args[0][1]

    @pytest.mark.anyio
    async def test_halt_session_fires(self, mock_notify):
        event = SentinelReviewEvent(
            action="halt_session",
            summary="danger",
            session_id="s1",
            review_id="r2",
        )
        await pty_notifier.handle_event(event)
        mock_notify.assert_awaited_once()
        assert "Sentinel HALT_SESSION" in mock_notify.call_args[0][1]

    @pytest.mark.anyio
    async def test_alert_fires(self, mock_notify):
        event = SentinelReviewEvent(
            action="alert",
            summary="worth noting",
            session_id="s1",
            review_id="r3",
        )
        await pty_notifier.handle_event(event)
        mock_notify.assert_awaited_once()

    @pytest.mark.anyio
    async def test_allow_skipped(self, mock_notify):
        event = SentinelReviewEvent(
            action="allow",
            summary="all good",
            session_id="s1",
        )
        await pty_notifier.handle_event(event)
        mock_notify.assert_not_awaited()


# --- Skipped cases ---


class TestSkippedCases:
    @pytest.mark.anyio
    async def test_non_escalation_event_skipped(self, mock_notify, mock_prompt):
        event = ToolUseEvent(name="Bash", tool_use_id="t1", session_id="s1")
        await pty_notifier.handle_event(event)
        mock_notify.assert_not_awaited()
        mock_prompt.assert_not_awaited()

    @pytest.mark.anyio
    async def test_text_event_skipped(self, mock_notify, mock_prompt):
        event = TextEvent(text="hello", session_id="s1")
        await pty_notifier.handle_event(event)
        mock_notify.assert_not_awaited()
        mock_prompt.assert_not_awaited()

    @pytest.mark.anyio
    async def test_missing_session_id_skipped(self, mock_notify, mock_prompt):
        event = GateDecisionEvent(
            tool_use_id="t1",
            tool_name="Bash",
            action="escalate",
            reason="bad",
            session_id="",
        )
        await pty_notifier.handle_event(event)
        mock_notify.assert_not_awaited()
        mock_prompt.assert_not_awaited()

    @pytest.mark.anyio
    async def test_disabled_config_skips(self, mock_notify, mock_prompt, monkeypatch):
        monkeypatch.setattr("cross.plugins.pty_notifier.settings.pty_notifications_enabled", False)
        event = GateDecisionEvent(
            tool_use_id="t1",
            tool_name="Bash",
            action="escalate",
            reason="bad",
            session_id="s1",
        )
        await pty_notifier.handle_event(event)
        mock_notify.assert_not_awaited()
        mock_prompt.assert_not_awaited()

    @pytest.mark.anyio
    async def test_no_callbacks_skips(self):
        pty_notifier._notify_callback = None
        pty_notifier._prompt_callback = None
        event = GateDecisionEvent(
            tool_use_id="t1",
            tool_name="Bash",
            action="block",
            reason="bad",
            session_id="s1",
        )
        await pty_notifier.handle_event(event)
        # No error, just silently skips


# --- Deduplication ---


class TestDeduplication:
    @pytest.mark.anyio
    async def test_duplicate_gate_suppressed(self, mock_notify):
        event = GateDecisionEvent(
            tool_use_id="dup1",
            tool_name="Bash",
            action="block",
            reason="bad",
            session_id="s1",
        )
        await pty_notifier.handle_event(event)
        await pty_notifier.handle_event(event)
        assert mock_notify.await_count == 1

    @pytest.mark.anyio
    async def test_duplicate_escalate_suppressed(self, mock_prompt):
        event = GateDecisionEvent(
            tool_use_id="dup2",
            tool_name="Bash",
            action="escalate",
            reason="bad",
            session_id="s1",
        )
        await pty_notifier.handle_event(event)
        await pty_notifier.handle_event(event)
        assert mock_prompt.await_count == 1

    @pytest.mark.anyio
    async def test_different_tool_use_ids_both_fire(self, mock_notify):
        e1 = GateDecisionEvent(
            tool_use_id="a",
            tool_name="Bash",
            action="block",
            reason="bad",
            session_id="s1",
        )
        e2 = GateDecisionEvent(
            tool_use_id="b",
            tool_name="Bash",
            action="block",
            reason="bad",
            session_id="s1",
        )
        await pty_notifier.handle_event(e1)
        await pty_notifier.handle_event(e2)
        assert mock_notify.await_count == 2

    @pytest.mark.anyio
    async def test_expired_dedup_allows_repeat(self, mock_notify):
        event = GateDecisionEvent(
            tool_use_id="exp1",
            tool_name="Bash",
            action="block",
            reason="bad",
            session_id="s1",
        )
        await pty_notifier.handle_event(event)
        # Expire the entry
        pty_notifier._recent["gate:exp1"] = time.time() - 10
        await pty_notifier.handle_event(event)
        assert mock_notify.await_count == 2

    @pytest.mark.anyio
    async def test_duplicate_sentinel_suppressed(self, mock_notify):
        event = SentinelReviewEvent(
            action="escalate",
            summary="pattern",
            session_id="s1",
            review_id="r1",
        )
        await pty_notifier.handle_event(event)
        await pty_notifier.handle_event(event)
        assert mock_notify.await_count == 1


# --- Tool summary formatting ---


class TestToolSummary:
    def test_command_field(self):
        event = GateDecisionEvent(
            tool_use_id="t1",
            tool_name="Bash",
            action="escalate",
            reason="test rule",
            tool_input={"command": "echo hello", "description": "test"},
            session_id="s1",
        )
        summary = pty_notifier._format_tool_summary(event)
        assert "echo hello" in summary
        assert "test rule" in summary

    def test_file_path_field(self):
        event = GateDecisionEvent(
            tool_use_id="t1",
            tool_name="Write",
            action="escalate",
            reason="writing file",
            tool_input={"file_path": "/etc/passwd", "content": "..."},
            session_id="s1",
        )
        summary = pty_notifier._format_tool_summary(event)
        assert "/etc/passwd" in summary

    def test_no_tool_input(self):
        event = GateDecisionEvent(
            tool_use_id="t1",
            tool_name="Bash",
            action="escalate",
            reason="some reason",
            session_id="s1",
        )
        summary = pty_notifier._format_tool_summary(event)
        assert "some reason" in summary

    def test_fallback_json(self):
        event = GateDecisionEvent(
            tool_use_id="t1",
            tool_name="Custom",
            action="escalate",
            reason="",
            tool_input={"foo": "bar"},
            session_id="s1",
        )
        summary = pty_notifier._format_tool_summary(event)
        assert "foo" in summary
