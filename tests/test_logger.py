"""Tests for the JSONL structured logger plugin."""

import json
import logging
from unittest.mock import patch

import pytest

from cross.events import (
    ErrorEvent,
    GateDecisionEvent,
    MessageDeltaEvent,
    MessageStartEvent,
    RequestEvent,
    SentinelReviewEvent,
    TextEvent,
    ToolUseEvent,
)
from cross.plugins.logger import LoggerPlugin


@pytest.fixture()
def logger_plugin(tmp_path):
    """Create a LoggerPlugin that writes to a temp directory."""
    log_file = tmp_path / "cross.log"
    with patch("cross.plugins.logger.settings") as mock_settings:
        mock_settings.log_file = str(log_file)
        plugin = LoggerPlugin()
    return plugin, log_file


def _read_records(log_file):
    """Read all JSONL records from the log file."""
    lines = log_file.read_text().strip().splitlines()
    return [json.loads(line) for line in lines]


class TestLoggerInit:
    def test_creates_log_directory(self, tmp_path):
        log_file = tmp_path / "subdir" / "nested" / "cross.log"
        with patch("cross.plugins.logger.settings") as mock_settings:
            mock_settings.log_file = str(log_file)
            LoggerPlugin()
        assert log_file.parent.exists()

    def test_opens_log_file(self, logger_plugin):
        plugin, log_file = logger_plugin
        assert log_file.exists()
        # File should be writable (plugin._file is open)
        assert not plugin._file.closed

    def test_logs_init_message(self, tmp_path, caplog):
        log_file = tmp_path / "cross.log"
        with patch("cross.plugins.logger.settings") as mock_settings:
            mock_settings.log_file = str(log_file)
            with caplog.at_level(logging.INFO, logger="cross.plugins.logger"):
                LoggerPlugin()
        assert any("Logging events to" in msg for msg in caplog.messages)


class TestWriteMethod:
    def test_writes_valid_jsonl(self, logger_plugin):
        plugin, log_file = logger_plugin
        plugin._write({"type": "test", "data": "hello"})
        records = _read_records(log_file)
        assert len(records) == 1
        assert records[0]["type"] == "test"
        assert records[0]["data"] == "hello"

    def test_adds_timestamp(self, logger_plugin):
        plugin, log_file = logger_plugin
        plugin._write({"type": "test"})
        records = _read_records(log_file)
        assert "ts" in records[0]
        # Should be ISO format UTC
        assert "T" in records[0]["ts"]

    def test_multiple_writes_produce_separate_lines(self, logger_plugin):
        plugin, log_file = logger_plugin
        plugin._write({"type": "a"})
        plugin._write({"type": "b"})
        plugin._write({"type": "c"})
        records = _read_records(log_file)
        assert len(records) == 3
        assert [r["type"] for r in records] == ["a", "b", "c"]


class TestRequestEvent:
    @pytest.mark.anyio
    async def test_logs_request(self, logger_plugin):
        plugin, log_file = logger_plugin
        event = RequestEvent(
            method="POST",
            path="/v1/messages",
            model="claude-sonnet-4-5-20250514",
            messages_count=3,
            stream=True,
            tool_names=["Bash", "Read", "Write"],
            last_message_role="user",
            last_message_preview="Write a test",
        )
        await plugin.handle(event)
        records = _read_records(log_file)
        assert len(records) == 1
        r = records[0]
        assert r["type"] == "request"
        assert r["method"] == "POST"
        assert r["path"] == "/v1/messages"
        assert r["model"] == "claude-sonnet-4-5-20250514"
        assert r["messages_count"] == 3
        assert r["stream"] is True
        assert r["tools"] == ["Bash", "Read", "Write"]
        assert r["tools_count"] == 3
        assert r["last_message_role"] == "user"
        assert r["last_message_preview"] == "Write a test"

    @pytest.mark.anyio
    async def test_tools_truncated_to_15(self, logger_plugin):
        plugin, log_file = logger_plugin
        tool_names = [f"tool_{i}" for i in range(20)]
        event = RequestEvent(
            method="POST",
            path="/v1/messages",
            tool_names=tool_names,
        )
        await plugin.handle(event)
        records = _read_records(log_file)
        assert len(records[0]["tools"]) == 15
        assert records[0]["tools_count"] == 20

    @pytest.mark.anyio
    async def test_empty_tools_no_tools_in_console_log(self, logger_plugin, caplog):
        plugin, _ = logger_plugin
        event = RequestEvent(method="GET", path="/health", tool_names=[])
        with caplog.at_level(logging.INFO, logger="cross.plugins.logger"):
            await plugin.handle(event)
        # Console log should not include "tools=" when empty
        log_msgs = [m for m in caplog.messages if "REQUEST" in m]
        assert len(log_msgs) == 1
        assert "tools=" not in log_msgs[0]

    @pytest.mark.anyio
    async def test_request_console_includes_tools_count(self, logger_plugin, caplog):
        plugin, _ = logger_plugin
        event = RequestEvent(
            method="POST",
            path="/v1/messages",
            model="test",
            messages_count=1,
            stream=False,
            tool_names=["Bash"],
        )
        with caplog.at_level(logging.INFO, logger="cross.plugins.logger"):
            await plugin.handle(event)
        log_msgs = [m for m in caplog.messages if "REQUEST" in m]
        assert "tools=1" in log_msgs[0]


class TestMessageStartEvent:
    @pytest.mark.anyio
    async def test_logs_message_start(self, logger_plugin):
        plugin, log_file = logger_plugin
        event = MessageStartEvent(message_id="msg_abc123", model="claude-sonnet-4-5-20250514")
        await plugin.handle(event)
        records = _read_records(log_file)
        assert len(records) == 1
        r = records[0]
        assert r["type"] == "message_start"
        assert r["message_id"] == "msg_abc123"
        assert r["model"] == "claude-sonnet-4-5-20250514"

    @pytest.mark.anyio
    async def test_console_log(self, logger_plugin, caplog):
        plugin, _ = logger_plugin
        event = MessageStartEvent(message_id="msg_xyz", model="test-model")
        with caplog.at_level(logging.INFO, logger="cross.plugins.logger"):
            await plugin.handle(event)
        assert any("message_start" in m and "msg_xyz" in m for m in caplog.messages)


class TestToolUseEvent:
    @pytest.mark.anyio
    async def test_logs_tool_use(self, logger_plugin):
        plugin, log_file = logger_plugin
        event = ToolUseEvent(
            name="Bash",
            tool_use_id="tu_123",
            input={"command": "ls -la"},
        )
        await plugin.handle(event)
        records = _read_records(log_file)
        assert len(records) == 1
        r = records[0]
        assert r["type"] == "tool_use"
        assert r["name"] == "Bash"
        assert r["tool_use_id"] == "tu_123"
        assert r["input"] == {"command": "ls -la"}

    @pytest.mark.anyio
    async def test_large_input_truncated_in_console(self, logger_plugin, caplog):
        plugin, _ = logger_plugin
        large_input = {"command": "x" * 300}
        event = ToolUseEvent(name="Bash", tool_use_id="tu_456", input=large_input)
        with caplog.at_level(logging.INFO, logger="cross.plugins.logger"):
            await plugin.handle(event)
        log_msgs = [m for m in caplog.messages if "tool_use" in m]
        assert len(log_msgs) == 1
        assert log_msgs[0].endswith("...")

    @pytest.mark.anyio
    async def test_small_input_not_truncated_in_console(self, logger_plugin, caplog):
        plugin, _ = logger_plugin
        event = ToolUseEvent(name="Read", tool_use_id="tu_789", input={"path": "/a.py"})
        with caplog.at_level(logging.INFO, logger="cross.plugins.logger"):
            await plugin.handle(event)
        log_msgs = [m for m in caplog.messages if "tool_use" in m]
        assert not log_msgs[0].endswith("...")

    @pytest.mark.anyio
    async def test_full_input_in_file(self, logger_plugin):
        """File log should contain the full input, not truncated."""
        plugin, log_file = logger_plugin
        large_input = {"command": "x" * 300}
        event = ToolUseEvent(name="Bash", tool_use_id="tu_big", input=large_input)
        await plugin.handle(event)
        records = _read_records(log_file)
        assert records[0]["input"]["command"] == "x" * 300


class TestTextEvent:
    @pytest.mark.anyio
    async def test_logs_text(self, logger_plugin):
        plugin, log_file = logger_plugin
        event = TextEvent(text="Hello world")
        await plugin.handle(event)
        records = _read_records(log_file)
        assert len(records) == 1
        assert records[0]["type"] == "text"
        assert records[0]["text"] == "Hello world"

    @pytest.mark.anyio
    async def test_long_text_truncated_in_console(self, logger_plugin, caplog):
        plugin, _ = logger_plugin
        event = TextEvent(text="A" * 300)
        with caplog.at_level(logging.INFO, logger="cross.plugins.logger"):
            await plugin.handle(event)
        log_msgs = [m for m in caplog.messages if "text:" in m]
        assert log_msgs[0].endswith("...")

    @pytest.mark.anyio
    async def test_short_text_not_truncated_in_console(self, logger_plugin, caplog):
        plugin, _ = logger_plugin
        event = TextEvent(text="short text")
        with caplog.at_level(logging.INFO, logger="cross.plugins.logger"):
            await plugin.handle(event)
        log_msgs = [m for m in caplog.messages if "text:" in m]
        assert not log_msgs[0].endswith("...")

    @pytest.mark.anyio
    async def test_full_text_in_file(self, logger_plugin):
        plugin, log_file = logger_plugin
        long_text = "B" * 500
        event = TextEvent(text=long_text)
        await plugin.handle(event)
        records = _read_records(log_file)
        assert records[0]["text"] == long_text


class TestMessageDeltaEvent:
    @pytest.mark.anyio
    async def test_logs_message_delta(self, logger_plugin):
        plugin, log_file = logger_plugin
        event = MessageDeltaEvent(stop_reason="end_turn", output_tokens=150)
        await plugin.handle(event)
        records = _read_records(log_file)
        assert len(records) == 1
        r = records[0]
        assert r["type"] == "message_delta"
        assert r["stop_reason"] == "end_turn"
        assert r["output_tokens"] == 150

    @pytest.mark.anyio
    async def test_tool_use_stop_reason(self, logger_plugin):
        plugin, log_file = logger_plugin
        event = MessageDeltaEvent(stop_reason="tool_use", output_tokens=42)
        await plugin.handle(event)
        records = _read_records(log_file)
        assert records[0]["stop_reason"] == "tool_use"

    @pytest.mark.anyio
    async def test_console_log(self, logger_plugin, caplog):
        plugin, _ = logger_plugin
        event = MessageDeltaEvent(stop_reason="end_turn", output_tokens=99)
        with caplog.at_level(logging.INFO, logger="cross.plugins.logger"):
            await plugin.handle(event)
        assert any("end_turn" in m and "99" in m for m in caplog.messages)


class TestErrorEvent:
    @pytest.mark.anyio
    async def test_logs_error(self, logger_plugin):
        plugin, log_file = logger_plugin
        event = ErrorEvent(status_code=429, body="Rate limited")
        await plugin.handle(event)
        records = _read_records(log_file)
        assert len(records) == 1
        r = records[0]
        assert r["type"] == "error"
        assert r["status_code"] == 429
        assert r["body"] == "Rate limited"

    @pytest.mark.anyio
    async def test_error_logged_as_warning(self, logger_plugin, caplog):
        plugin, _ = logger_plugin
        event = ErrorEvent(status_code=500, body="Server error")
        with caplog.at_level(logging.WARNING, logger="cross.plugins.logger"):
            await plugin.handle(event)
        assert any("ERROR 500" in m for m in caplog.messages)

    @pytest.mark.anyio
    async def test_long_body_truncated_in_console(self, logger_plugin, caplog):
        plugin, _ = logger_plugin
        event = ErrorEvent(status_code=400, body="X" * 400)
        with caplog.at_level(logging.WARNING, logger="cross.plugins.logger"):
            await plugin.handle(event)
        # The body[:200] in logger.warning truncates it
        log_msgs = [m for m in caplog.messages if "ERROR 400" in m]
        assert len(log_msgs) == 1
        # Console message should be shorter than the full 400-char body
        assert len(log_msgs[0]) < 400


class TestGateDecisionEvent:
    @pytest.mark.anyio
    async def test_logs_gate_decision(self, logger_plugin):
        plugin, log_file = logger_plugin
        event = GateDecisionEvent(
            tool_use_id="tu_abc",
            tool_name="Bash",
            action="block",
            reason="Destructive command: rm -rf /",
            rule_id="destructive-rm",
            evaluator="DenylistGate",
        )
        await plugin.handle(event)
        records = _read_records(log_file)
        assert len(records) == 1
        r = records[0]
        assert r["type"] == "gate_decision"
        assert r["tool_use_id"] == "tu_abc"
        assert r["tool_name"] == "Bash"
        assert r["action"] == "block"
        assert r["reason"] == "Destructive command: rm -rf /"
        assert r["rule_id"] == "destructive-rm"
        assert r["evaluator"] == "DenylistGate"

    @pytest.mark.anyio
    async def test_allow_decision(self, logger_plugin):
        plugin, log_file = logger_plugin
        event = GateDecisionEvent(
            tool_use_id="tu_xyz",
            tool_name="Read",
            action="allow",
            evaluator="DenylistGate",
        )
        await plugin.handle(event)
        records = _read_records(log_file)
        assert records[0]["action"] == "allow"

    @pytest.mark.anyio
    async def test_console_log_format(self, logger_plugin, caplog):
        plugin, _ = logger_plugin
        event = GateDecisionEvent(
            tool_use_id="tu_1",
            tool_name="Bash",
            action="block",
            evaluator="DenylistGate",
        )
        with caplog.at_level(logging.INFO, logger="cross.plugins.logger"):
            await plugin.handle(event)
        log_msgs = [m for m in caplog.messages if "gate:" in m]
        assert len(log_msgs) == 1
        assert "Bash" in log_msgs[0]
        assert "block" in log_msgs[0]
        assert "DenylistGate" in log_msgs[0]


class TestSentinelReviewEvent:
    @pytest.mark.anyio
    async def test_logs_sentinel_review(self, logger_plugin):
        plugin, log_file = logger_plugin
        event = SentinelReviewEvent(
            action="alert",
            summary="Activity looks suspicious",
            concerns="Multiple credential reads followed by network calls",
            event_count=12,
            evaluator="LLMSentinel",
        )
        await plugin.handle(event)
        records = _read_records(log_file)
        assert len(records) == 1
        r = records[0]
        assert r["type"] == "sentinel_review"
        assert r["action"] == "alert"
        assert r["summary"] == "Activity looks suspicious"
        assert r["concerns"] == "Multiple credential reads followed by network calls"
        assert r["event_count"] == 12
        assert r["evaluator"] == "LLMSentinel"

    @pytest.mark.anyio
    async def test_alert_logged_as_warning(self, logger_plugin, caplog):
        plugin, _ = logger_plugin
        event = SentinelReviewEvent(
            action="alert",
            concerns="Suspicious pattern detected",
            summary="",
            evaluator="LLMSentinel",
        )
        with caplog.at_level(logging.WARNING, logger="cross.plugins.logger"):
            await plugin.handle(event)
        assert any("sentinel alert" in m for m in caplog.messages)

    @pytest.mark.anyio
    async def test_escalate_logged_as_warning(self, logger_plugin, caplog):
        plugin, _ = logger_plugin
        event = SentinelReviewEvent(
            action="escalate",
            concerns="Needs human review",
            summary="",
            evaluator="LLMSentinel",
        )
        with caplog.at_level(logging.WARNING, logger="cross.plugins.logger"):
            await plugin.handle(event)
        assert any("sentinel escalate" in m for m in caplog.messages)

    @pytest.mark.anyio
    async def test_halt_session_logged_as_warning(self, logger_plugin, caplog):
        plugin, _ = logger_plugin
        event = SentinelReviewEvent(
            action="halt_session",
            concerns="Critical threat",
            summary="",
            evaluator="LLMSentinel",
        )
        with caplog.at_level(logging.WARNING, logger="cross.plugins.logger"):
            await plugin.handle(event)
        assert any("sentinel halt_session" in m for m in caplog.messages)

    @pytest.mark.anyio
    async def test_allow_logged_as_info(self, logger_plugin, caplog):
        plugin, _ = logger_plugin
        event = SentinelReviewEvent(
            action="allow",
            summary="Everything looks normal",
            concerns="",
            evaluator="LLMSentinel",
        )
        with caplog.at_level(logging.INFO, logger="cross.plugins.logger"):
            await plugin.handle(event)
        log_msgs = [m for m in caplog.messages if "sentinel allow" in m]
        assert len(log_msgs) == 1
        assert "Everything looks normal" in log_msgs[0]


class TestMultipleEvents:
    @pytest.mark.anyio
    async def test_mixed_events_produce_correct_records(self, logger_plugin):
        plugin, log_file = logger_plugin
        await plugin.handle(RequestEvent(method="POST", path="/v1/messages"))
        await plugin.handle(MessageStartEvent(message_id="msg_1", model="test"))
        await plugin.handle(ToolUseEvent(name="Bash", tool_use_id="tu_1", input={"command": "ls"}))
        await plugin.handle(TextEvent(text="Done"))
        await plugin.handle(MessageDeltaEvent(stop_reason="end_turn", output_tokens=10))

        records = _read_records(log_file)
        assert len(records) == 5
        types = [r["type"] for r in records]
        assert types == ["request", "message_start", "tool_use", "text", "message_delta"]

    @pytest.mark.anyio
    async def test_timestamps_are_monotonic(self, logger_plugin):
        plugin, log_file = logger_plugin
        await plugin.handle(TextEvent(text="first"))
        await plugin.handle(TextEvent(text="second"))
        records = _read_records(log_file)
        assert records[0]["ts"] <= records[1]["ts"]


class TestShutdown:
    def test_file_can_be_closed(self, logger_plugin):
        plugin, log_file = logger_plugin
        plugin._file.close()
        assert plugin._file.closed

    @pytest.mark.anyio
    async def test_write_before_close_persists(self, logger_plugin):
        plugin, log_file = logger_plugin
        await plugin.handle(TextEvent(text="final"))
        plugin._file.close()
        records = _read_records(log_file)
        assert len(records) == 1
        assert records[0]["text"] == "final"
