"""Tests for proxy-level gating (blocked tool feedback injection)."""

import json
import time

from cross.proxy import (
    _blocked_tool_ids,
    _blocked_tool_info,
    _blocked_tool_timestamps,
    _inject_blocked_tool_feedback,
)


class TestBlockedToolFeedbackInjection:
    def setup_method(self):
        _blocked_tool_ids.clear()
        _blocked_tool_info.clear()
        _blocked_tool_timestamps.clear()

    def test_no_blocked_ids_passthrough(self):
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
        result = _inject_blocked_tool_feedback(body)
        assert result == body

    def test_injects_blocked_tool_feedback(self):
        _blocked_tool_ids["toolu_123"] = "Blocked by denylist"
        _blocked_tool_info["toolu_123"] = {"name": "Bash", "input": {"command": "rm -rf /"}}
        _blocked_tool_timestamps["toolu_123"] = time.time()

        body = json.dumps(
            {
                "messages": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "I will run this"}],
                    },
                    {
                        "role": "user",
                        "content": "next message",
                    },
                ]
            }
        ).encode()

        result = _inject_blocked_tool_feedback(body)
        data = json.loads(result)

        # Tool_use appended to assistant message
        assistant_content = data["messages"][0]["content"]
        assert len(assistant_content) == 2
        assert assistant_content[1]["type"] == "tool_use"
        assert assistant_content[1]["id"] == "toolu_123"
        assert assistant_content[1]["name"] == "Bash"

        # Error tool_result prepended to user message
        user_content = data["messages"][1]["content"]
        assert user_content[0]["type"] == "tool_result"
        assert user_content[0]["is_error"] is True
        assert "Cross blocked" in user_content[0]["content"]
        assert user_content[0]["tool_use_id"] == "toolu_123"

        # Should be consumed from _blocked_tool_ids
        assert "toolu_123" not in _blocked_tool_ids

    def test_leaves_unblocked_conversation_alone(self):
        """When no blocked tools, messages are untouched."""
        body = json.dumps(
            {
                "messages": [
                    {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                    {"role": "user", "content": "thanks"},
                ]
            }
        ).encode()

        result = _inject_blocked_tool_feedback(body)
        assert result == body

    def test_handles_non_json_body(self):
        _blocked_tool_ids["t1"] = "blocked"
        _blocked_tool_info["t1"] = {"name": "Bash", "input": {}}
        body = b"not json"
        result = _inject_blocked_tool_feedback(body)
        assert result == body

    def test_handles_multiple_blocked_tools(self):
        _blocked_tool_ids["toolu_a"] = "blocked a"
        _blocked_tool_info["toolu_a"] = {"name": "Bash", "input": {"command": "rm"}}
        _blocked_tool_timestamps["toolu_a"] = time.time()
        _blocked_tool_ids["toolu_b"] = "blocked b"
        _blocked_tool_info["toolu_b"] = {"name": "Write", "input": {"file_path": "/etc/x"}}
        _blocked_tool_timestamps["toolu_b"] = time.time()

        body = json.dumps(
            {
                "messages": [
                    {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                    {"role": "user", "content": "next"},
                ]
            }
        ).encode()

        result = _inject_blocked_tool_feedback(body)
        data = json.loads(result)

        # Both tool_use blocks appended to assistant
        assistant_content = data["messages"][0]["content"]
        tool_uses = [b for b in assistant_content if b["type"] == "tool_use"]
        assert len(tool_uses) == 2

        # Both error tool_results prepended to user
        user_content = data["messages"][1]["content"]
        tool_results = [b for b in user_content if b["type"] == "tool_result"]
        assert len(tool_results) == 2
        assert all(tr["is_error"] for tr in tool_results)
