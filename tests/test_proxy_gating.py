"""Tests for proxy-level gating (blocked tool interception)."""

import json

from cross.proxy import _intercept_blocked_tool_results, _blocked_tool_ids, _blocked_tool_timestamps


class TestBlockedToolInterception:
    def setup_method(self):
        _blocked_tool_ids.clear()
        _blocked_tool_timestamps.clear()

    def test_no_blocked_ids_passthrough(self):
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode()
        result = _intercept_blocked_tool_results(body)
        assert result == body

    def test_replaces_blocked_tool_result(self):
        _blocked_tool_ids["toolu_123"] = "Blocked by denylist"

        body = json.dumps({
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_123",
                    "content": "file deleted successfully",
                }]
            }]
        }).encode()

        result = _intercept_blocked_tool_results(body)
        data = json.loads(result)

        block = data["messages"][0]["content"][0]
        assert block["is_error"] is True
        assert "Cross blocked" in block["content"]
        assert block["tool_use_id"] == "toolu_123"

        # Should be removed from _blocked_tool_ids after interception
        assert "toolu_123" not in _blocked_tool_ids

    def test_leaves_unblocked_tool_results_alone(self):
        _blocked_tool_ids["toolu_999"] = "blocked"

        body = json.dumps({
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "toolu_456",
                    "content": "ok",
                }]
            }]
        }).encode()

        result = _intercept_blocked_tool_results(body)
        data = json.loads(result)

        block = data["messages"][0]["content"][0]
        assert block["content"] == "ok"
        assert "is_error" not in block

    def test_handles_non_json_body(self):
        body = b"not json"
        result = _intercept_blocked_tool_results(body)
        assert result == body

    def test_handles_multiple_tool_results(self):
        _blocked_tool_ids["toolu_a"] = "blocked a"

        body = json.dumps({
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_a", "content": "bad"},
                    {"type": "tool_result", "tool_use_id": "toolu_b", "content": "good"},
                ]
            }]
        }).encode()

        result = _intercept_blocked_tool_results(body)
        data = json.loads(result)
        blocks = data["messages"][0]["content"]

        assert blocks[0]["is_error"] is True
        assert blocks[1]["content"] == "good"
