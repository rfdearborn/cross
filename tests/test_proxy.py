"""Unit tests for cross.proxy — core reverse proxy module.

Tests cover:
- _extract_user_intent: message parsing, system-reminder skipping, content types
- _extract_request_event: request parsing, edge cases, body variations
- _inject_blocked_tool_feedback: stale cleanup, message injection, edge cases
- _build_retry_request_body: retry request body construction
- _rewrite_content_block_index: SSE content block index rewriting
- get_client: singleton behavior
- handle_proxy_request: streaming vs non-streaming routing
- _proxy_simple: non-streaming with/without gate chain, error responses
- _gate_non_streaming_response: tool_use evaluation, cascade blocking/halting
- _is_tool_use_block_start: SSE line classification
- _proxy_streaming: SSE streaming with/without gate chain, buffering, blocking, retries, halting, alerts
- shutdown: client cleanup
"""

import asyncio
import json
import time
import unittest.mock
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from starlette.responses import StreamingResponse

import cross.proxy as proxy_module
from cross.chain import GateChain
from cross.config import settings
from cross.evaluator import Action, EvaluationResponse, Gate, GateRequest
from cross.events import (
    ErrorEvent,
    EventBus,
    GateDecisionEvent,
    GateRetryEvent,
    RequestEvent,
)
from cross.proxy import (
    _BLOCKED_TOOL_TTL,
    _MAX_BUFFER_LINES,
    _blocked_tool_ids,
    _blocked_tool_info,
    _blocked_tool_timestamps,
    _build_retry_request_body,
    _extract_request_event,
    _extract_user_intent,
    _gate_non_streaming_response,
    _inject_blocked_tool_feedback,
    _is_tool_use_block_start,
    _proxy_streaming,
    _recent_tools,
    _rewrite_content_block_index,
    get_client,
    handle_proxy_request,
    resolve_gate_approval,
    shutdown,
)

# --- Fixtures ---


@pytest.fixture(autouse=True)
def _reset_proxy_globals():
    """Reset module-level mutable state before each test."""
    _blocked_tool_ids.clear()
    _blocked_tool_info.clear()
    _blocked_tool_timestamps.clear()
    _recent_tools.clear()
    # Reset sentinel halt state
    proxy_module._sentinel_halted = False
    proxy_module._sentinel_halt_reason = ""
    # Reset the singleton client
    old_client = proxy_module._client
    proxy_module._client = None
    yield
    # Restore after test
    proxy_module._sentinel_halted = False
    proxy_module._sentinel_halt_reason = ""
    proxy_module._client = old_client


# --- Test gates ---


class AllowGate(Gate):
    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        return EvaluationResponse(action=Action.ALLOW, evaluator=self.name)


class BlockGate(Gate):
    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        return EvaluationResponse(
            action=Action.BLOCK,
            reason="blocked by gate",
            rule_id="test-rule",
            evaluator=self.name,
        )


class EscalateGate(Gate):
    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        return EvaluationResponse(
            action=Action.ESCALATE,
            reason="escalated by gate",
            evaluator=self.name,
        )


class AlertGate(Gate):
    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        return EvaluationResponse(
            action=Action.ALERT,
            reason="alert from gate",
            evaluator=self.name,
        )


class HaltSessionGate(Gate):
    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        return EvaluationResponse(
            action=Action.HALT_SESSION,
            reason="halted by gate",
            rule_id="test-halt-rule",
            evaluator=self.name,
        )


class SelectiveBlockGate(Gate):
    """Blocks only tools named 'Bash'."""

    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        if request.tool_name == "Bash":
            return EvaluationResponse(
                action=Action.BLOCK,
                reason="Bash is blocked",
                evaluator=self.name,
            )
        return EvaluationResponse(action=Action.ALLOW, evaluator=self.name)


class CapturingGate(Gate):
    """Records GateRequests for inspection."""

    def __init__(self):
        super().__init__(name="capturing")
        self.requests: list[GateRequest] = []

    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        self.requests.append(request)
        return EvaluationResponse(action=Action.ALLOW, evaluator=self.name)


# ============================================================
# _extract_user_intent
# ============================================================


class TestExtractUserIntent:
    def test_empty_messages(self):
        assert _extract_user_intent({}) == ""
        assert _extract_user_intent({"messages": []}) == ""

    def test_string_content(self):
        data = {"messages": [{"role": "user", "content": "Hello world"}]}
        assert _extract_user_intent(data) == "Hello world"

    def test_string_content_truncated_at_500(self):
        long_text = "x" * 1000
        data = {"messages": [{"role": "user", "content": long_text}]}
        result = _extract_user_intent(data)
        assert len(result) == 500
        assert result == "x" * 500

    def test_list_content_with_text_block(self):
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Do the thing"},
                    ],
                }
            ]
        }
        assert _extract_user_intent(data) == "Do the thing"

    def test_list_content_skips_system_reminder(self):
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<system-reminder>You are a helpful assistant</system-reminder>"},
                        {"type": "text", "text": "Actual user message"},
                    ],
                }
            ]
        }
        assert _extract_user_intent(data) == "Actual user message"

    def test_list_content_all_system_reminders_returns_empty(self):
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "<system-reminder>reminder 1</system-reminder>"},
                        {"type": "text", "text": "<system-reminder>reminder 2</system-reminder>"},
                    ],
                }
            ]
        }
        assert _extract_user_intent(data) == ""

    def test_list_content_skips_non_text_blocks(self):
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "t1", "content": "result text"},
                        {"type": "text", "text": "User question"},
                    ],
                }
            ]
        }
        assert _extract_user_intent(data) == "User question"

    def test_list_content_truncated_at_500(self):
        long_text = "a" * 800
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": long_text}],
                }
            ]
        }
        result = _extract_user_intent(data)
        assert len(result) == 500

    def test_uses_last_message(self):
        data = {
            "messages": [
                {"role": "user", "content": "First message"},
                {"role": "assistant", "content": "Response"},
                {"role": "user", "content": "Second message"},
            ]
        }
        assert _extract_user_intent(data) == "Second message"

    def test_empty_content_string(self):
        data = {"messages": [{"role": "user", "content": ""}]}
        assert _extract_user_intent(data) == ""

    def test_empty_text_block_skipped(self):
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": ""},
                        {"type": "text", "text": "Real content"},
                    ],
                }
            ]
        }
        assert _extract_user_intent(data) == "Real content"

    def test_missing_content_key(self):
        data = {"messages": [{"role": "user"}]}
        assert _extract_user_intent(data) == ""

    def test_content_is_other_type(self):
        """Content is neither str nor list (e.g., int) -- should return empty."""
        data = {"messages": [{"role": "user", "content": 42}]}
        assert _extract_user_intent(data) == ""

    def test_list_content_missing_text_key(self):
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text"}],
                }
            ]
        }
        # text defaults to "" which is falsy, so gets skipped
        assert _extract_user_intent(data) == ""


# ============================================================
# _extract_request_event
# ============================================================


class TestExtractRequestEvent:
    def test_basic_request(self):
        body = json.dumps(
            {
                "model": "claude-3-opus",
                "stream": True,
                "messages": [{"role": "user", "content": "Hello"}],
                "tools": [{"name": "Bash"}, {"name": "Read"}],
            }
        ).encode()

        event = _extract_request_event("POST", "/v1/messages", body)

        assert event.method == "POST"
        assert event.path == "/v1/messages"
        assert event.model == "claude-3-opus"
        assert event.stream is True
        assert event.messages_count == 1
        assert event.tool_names == ["Bash", "Read"]
        assert event.last_message_role == "user"
        assert event.last_message_preview == "Hello"
        assert event.raw_body is not None

    def test_no_body(self):
        event = _extract_request_event("GET", "/v1/models", None)
        assert event.method == "GET"
        assert event.path == "/v1/models"
        assert event.model is None
        assert event.stream is False
        assert event.messages_count == 0

    def test_empty_body(self):
        event = _extract_request_event("POST", "/v1/messages", b"")
        assert event.model is None
        assert event.messages_count == 0

    def test_invalid_json_body(self):
        event = _extract_request_event("POST", "/v1/messages", b"not json!")
        assert event.model is None
        assert event.messages_count == 0
        assert event.raw_body is None

    def test_no_messages(self):
        body = json.dumps({"model": "claude-3"}).encode()
        event = _extract_request_event("POST", "/v1/messages", body)
        assert event.messages_count == 0
        assert event.last_message_role is None
        assert event.last_message_preview is None

    def test_preview_truncated_to_200(self):
        long_msg = "z" * 600
        body = json.dumps(
            {
                "messages": [{"role": "user", "content": long_msg}],
            }
        ).encode()
        event = _extract_request_event("POST", "/v1/messages", body)
        assert len(event.last_message_preview) == 200

    def test_tool_result_only_no_preview(self):
        """When last user message has only tool results, preview is None."""
        body = json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": "t1"},
                            {"type": "image", "source": {}},
                        ],
                    }
                ],
            }
        ).encode()
        event = _extract_request_event("POST", "/v1/messages", body)
        assert event.last_message_preview is None

    def test_stream_defaults_to_false(self):
        body = json.dumps({"model": "claude-3", "messages": []}).encode()
        event = _extract_request_event("POST", "/v1/messages", body)
        assert event.stream is False

    def test_tools_missing_name(self):
        body = json.dumps(
            {
                "messages": [],
                "tools": [{"description": "a tool without a name"}, {"name": "Read"}],
            }
        ).encode()
        event = _extract_request_event("POST", "/v1/messages", body)
        assert event.tool_names == ["?", "Read"]

    def test_assistant_message_last(self):
        body = json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ],
            }
        ).encode()
        event = _extract_request_event("POST", "/v1/messages", body)
        assert event.last_message_role == "assistant"
        # _extract_user_intent searches backwards for the last user message
        assert event.last_message_preview == "Hello"

    def test_system_reminder_only_no_preview(self):
        """When all text blocks are system-reminders, preview is None."""
        body = json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "<system-reminder>hidden</system-reminder>"},
                        ],
                    }
                ],
            }
        ).encode()
        event = _extract_request_event("POST", "/v1/messages", body)
        assert event.last_message_preview is None


# ============================================================
# _inject_blocked_tool_feedback
# ============================================================


class TestInjectBlockedToolFeedback:
    """Tests for injecting blocked-tool feedback into the next request."""

    def test_stale_entries_cleaned_up(self):
        _blocked_tool_ids["old_tool"] = "old reason"
        _blocked_tool_info["old_tool"] = {"name": "Bash", "input": {}}
        _blocked_tool_timestamps["old_tool"] = time.time() - _BLOCKED_TOOL_TTL - 10

        body = json.dumps({"messages": []}).encode()
        _inject_blocked_tool_feedback(body)

        assert "old_tool" not in _blocked_tool_ids
        assert "old_tool" not in _blocked_tool_info
        assert "old_tool" not in _blocked_tool_timestamps

    def test_fresh_entries_not_cleaned(self):
        _blocked_tool_ids["fresh_tool"] = "fresh reason"
        _blocked_tool_info["fresh_tool"] = {"name": "Bash", "input": {}}
        _blocked_tool_timestamps["fresh_tool"] = time.time()

        body = json.dumps({"messages": []}).encode()
        _inject_blocked_tool_feedback(body)

        # Empty messages → nothing to inject into, entry preserved
        assert "fresh_tool" in _blocked_tool_ids

    def test_stale_and_fresh_mixed(self):
        _blocked_tool_ids["old"] = "old"
        _blocked_tool_info["old"] = {"name": "Bash", "input": {}}
        _blocked_tool_timestamps["old"] = time.time() - _BLOCKED_TOOL_TTL - 10
        _blocked_tool_ids["new"] = "new"
        _blocked_tool_info["new"] = {"name": "Bash", "input": {}}
        _blocked_tool_timestamps["new"] = time.time()

        body = json.dumps(
            {
                "messages": [
                    {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                    {"role": "user", "content": "next message"},
                ]
            }
        ).encode()
        _inject_blocked_tool_feedback(body)

        # Old should be cleaned, new should be consumed
        assert "old" not in _blocked_tool_ids
        assert "new" not in _blocked_tool_ids

    def test_no_messages_returns_unchanged(self):
        _blocked_tool_ids["t1"] = "reason"
        _blocked_tool_info["t1"] = {"name": "Bash", "input": {}}
        body = json.dumps({"model": "claude-3"}).encode()
        result = _inject_blocked_tool_feedback(body)
        data = json.loads(result)
        assert "messages" not in data

    def test_injects_tool_use_and_error_tool_result(self):
        """Core test: blocked tool gets injected as tool_use in assistant + error tool_result in user."""
        _blocked_tool_ids["t1"] = "dangerous command"
        _blocked_tool_info["t1"] = {"name": "Bash", "input": {"command": "rm -rf /"}}
        _blocked_tool_timestamps["t1"] = time.time()

        body = json.dumps(
            {
                "messages": [
                    {"role": "assistant", "content": [{"type": "text", "text": "I will run this"}]},
                    {"role": "user", "content": "do it"},
                ]
            }
        ).encode()
        result = _inject_blocked_tool_feedback(body)
        data = json.loads(result)

        # Assistant message should now have tool_use appended
        assistant_content = data["messages"][0]["content"]
        assert len(assistant_content) == 2
        assert assistant_content[0]["type"] == "text"
        assert assistant_content[1]["type"] == "tool_use"
        assert assistant_content[1]["id"] == "t1"
        assert assistant_content[1]["name"] == "Bash"
        assert assistant_content[1]["input"] == {"command": "rm -rf /"}

        # User message should have tool_result prepended
        user_content = data["messages"][1]["content"]
        assert len(user_content) == 2
        assert user_content[0]["type"] == "tool_result"
        assert user_content[0]["tool_use_id"] == "t1"
        assert user_content[0]["is_error"] is True
        assert "Cross blocked" in user_content[0]["content"]
        assert user_content[1]["type"] == "text"
        assert user_content[1]["text"] == "do it"

        assert "t1" not in _blocked_tool_ids

    def test_multiple_blocked_tools_injected(self):
        _blocked_tool_ids["t1"] = "reason1"
        _blocked_tool_info["t1"] = {"name": "Bash", "input": {"command": "rm"}}
        _blocked_tool_timestamps["t1"] = time.time()
        _blocked_tool_ids["t2"] = "reason2"
        _blocked_tool_info["t2"] = {"name": "Write", "input": {"file_path": "/etc/passwd"}}
        _blocked_tool_timestamps["t2"] = time.time()

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

    def test_string_assistant_content_converted_to_list(self):
        """Assistant content that is a string should be converted to list before appending."""
        _blocked_tool_ids["t1"] = "blocked"
        _blocked_tool_info["t1"] = {"name": "Bash", "input": {}}
        _blocked_tool_timestamps["t1"] = time.time()

        body = json.dumps(
            {
                "messages": [
                    {"role": "assistant", "content": "I will run it"},
                    {"role": "user", "content": "ok"},
                ]
            }
        ).encode()
        result = _inject_blocked_tool_feedback(body)
        data = json.loads(result)

        assistant_content = data["messages"][0]["content"]
        assert isinstance(assistant_content, list)
        assert assistant_content[0]["type"] == "text"
        assert assistant_content[0]["text"] == "I will run it"
        assert assistant_content[1]["type"] == "tool_use"

    def test_string_user_content_converted_to_list(self):
        """User content that is a string should be converted to list before prepending."""
        _blocked_tool_ids["t1"] = "blocked"
        _blocked_tool_info["t1"] = {"name": "Bash", "input": {}}
        _blocked_tool_timestamps["t1"] = time.time()

        body = json.dumps(
            {
                "messages": [
                    {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                    {"role": "user", "content": "next message"},
                ]
            }
        ).encode()
        result = _inject_blocked_tool_feedback(body)
        data = json.loads(result)

        user_content = data["messages"][1]["content"]
        assert isinstance(user_content, list)
        assert user_content[0]["type"] == "tool_result"
        assert user_content[1]["type"] == "text"
        assert user_content[1]["text"] == "next message"

    def test_no_assistant_message_inserts_both(self):
        """When no assistant message exists, both assistant+user messages are inserted."""
        _blocked_tool_ids["t1"] = "blocked"
        _blocked_tool_info["t1"] = {"name": "Bash", "input": {}}
        _blocked_tool_timestamps["t1"] = time.time()

        body = json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "do something"},
                ]
            }
        ).encode()
        result = _inject_blocked_tool_feedback(body)
        data = json.loads(result)

        assert len(data["messages"]) == 3
        assert data["messages"][0]["role"] == "assistant"
        assert data["messages"][0]["content"][0]["type"] == "tool_use"
        assert data["messages"][1]["role"] == "user"
        assert data["messages"][1]["content"][0]["type"] == "tool_result"
        assert data["messages"][2]["role"] == "user"
        assert data["messages"][2]["content"] == "do something"

    def test_no_following_user_message_appends(self):
        """When no user message follows the assistant, a new user message is appended."""
        _blocked_tool_ids["t1"] = "blocked"
        _blocked_tool_info["t1"] = {"name": "Bash", "input": {}}
        _blocked_tool_timestamps["t1"] = time.time()

        body = json.dumps(
            {
                "messages": [
                    {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                ]
            }
        ).encode()
        result = _inject_blocked_tool_feedback(body)
        data = json.loads(result)

        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "assistant"
        assert data["messages"][1]["role"] == "user"
        assert data["messages"][1]["content"][0]["type"] == "tool_result"

    def test_blocked_without_info_skipped(self):
        """Blocked tool with no info in _blocked_tool_info is skipped (no crash)."""
        _blocked_tool_ids["orphan"] = "reason"
        # No entry in _blocked_tool_info

        body = json.dumps(
            {
                "messages": [
                    {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                    {"role": "user", "content": "next"},
                ]
            }
        ).encode()
        result = _inject_blocked_tool_feedback(body)
        # No info → nothing to inject, body unchanged
        assert result == body
        assert "orphan" not in _blocked_tool_ids  # consumed

    def test_bad_json_with_blocked_ids_returns_unchanged(self):
        """When _blocked_tool_ids is non-empty but body is invalid JSON, return body unchanged."""
        _blocked_tool_ids["t1"] = "some reason"
        _blocked_tool_info["t1"] = {"name": "Bash", "input": {}}
        _blocked_tool_timestamps["t1"] = time.time()

        body = b"this is not valid json at all"
        result = _inject_blocked_tool_feedback(body)
        assert result == body
        assert "t1" in _blocked_tool_ids

    def test_empty_blocked_ids_returns_unchanged(self):
        """When _blocked_tool_ids is empty, return body unchanged."""
        body = json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                ]
            }
        ).encode()
        result = _inject_blocked_tool_feedback(body)
        assert result == body


# ============================================================
# get_client
# ============================================================


class TestGetClient:
    def test_returns_async_client(self):
        client = get_client()
        assert isinstance(client, httpx.AsyncClient)

    def test_singleton_returns_same_instance(self):
        c1 = get_client()
        c2 = get_client()
        assert c1 is c2

    def test_creates_new_after_reset(self):
        c1 = get_client()
        proxy_module._client = None
        c2 = get_client()
        assert c1 is not c2


# ============================================================
# _is_tool_use_block_start (additional cases)
# ============================================================


class TestIsToolUseBlockStart:
    def test_valid_tool_use_start(self):
        data = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Bash"},
        }
        line = f"data: {json.dumps(data)}"
        assert _is_tool_use_block_start(line) is True

    def test_text_block_start(self):
        data = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
        line = f"data: {json.dumps(data)}"
        assert _is_tool_use_block_start(line) is False

    def test_not_data_prefix(self):
        assert _is_tool_use_block_start("event: content_block_start") is False

    def test_empty_string(self):
        assert _is_tool_use_block_start("") is False

    def test_malformed_json(self):
        assert _is_tool_use_block_start("data: {not valid}") is False

    def test_content_block_delta(self):
        data = {"type": "content_block_delta", "index": 0}
        line = f"data: {json.dumps(data)}"
        assert _is_tool_use_block_start(line) is False

    def test_message_start(self):
        data = {"type": "message_start", "message": {"id": "m1", "model": "claude-3"}}
        line = f"data: {json.dumps(data)}"
        assert _is_tool_use_block_start(line) is False

    def test_content_block_stop(self):
        data = {"type": "content_block_stop", "index": 0}
        line = f"data: {json.dumps(data)}"
        assert _is_tool_use_block_start(line) is False

    def test_missing_content_block_key(self):
        data = {"type": "content_block_start", "index": 0}
        line = f"data: {json.dumps(data)}"
        assert _is_tool_use_block_start(line) is False

    def test_content_block_not_dict(self):
        """content_block is a string -- .get() would fail on str, should return False."""
        data = {"type": "content_block_start", "index": 0, "content_block": "not_a_dict"}
        line = f"data: {json.dumps(data)}"
        assert _is_tool_use_block_start(line) is False

    def test_data_done_marker(self):
        assert _is_tool_use_block_start("data: [DONE]") is False

    def test_comment_line(self):
        assert _is_tool_use_block_start(": this is a comment") is False


# ============================================================
# handle_proxy_request
# ============================================================


def _make_mock_request(body_dict: dict, method: str = "POST", path: str = "/v1/messages", query: str = "") -> MagicMock:
    """Build a Starlette-like Request mock."""
    body_bytes = json.dumps(body_dict).encode()
    req = MagicMock()
    req.method = method
    req.url = MagicMock()
    req.url.path = path
    req.url.query = query
    req.headers = {"host": "localhost:8080", "content-type": "application/json", "x-api-key": "sk-test"}
    req.body = AsyncMock(return_value=body_bytes)
    return req


class TestHandleProxyRequest:
    @pytest.mark.anyio
    async def test_sentinel_halt_blocks_requests(self):
        """When sentinel has halted, proxy should reject requests with 403."""
        body = {"model": "claude-3", "stream": False, "messages": [{"role": "user", "content": "hi"}]}
        req = _make_mock_request(body)
        event_bus = EventBus()

        proxy_module._sentinel_halted = True
        proxy_module._sentinel_halt_reason = "Agent exfiltrating credentials"

        resp = await handle_proxy_request(req, event_bus)

        assert resp.status_code == 403
        resp_data = json.loads(resp.body)
        assert "sentinel" in resp_data["error"]["message"].lower()
        assert "exfiltrating credentials" in resp_data["error"]["message"]

    @pytest.mark.anyio
    async def test_routes_streaming_request(self):
        body = {"model": "claude-3", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
        req = _make_mock_request(body)
        event_bus = EventBus()

        with (
            patch("cross.proxy._proxy_streaming", new_callable=AsyncMock) as mock_streaming,
            patch("cross.proxy._proxy_simple", new_callable=AsyncMock) as mock_simple,
        ):
            mock_streaming.return_value = MagicMock()
            await handle_proxy_request(req, event_bus)

            mock_streaming.assert_called_once()
            mock_simple.assert_not_called()

    @pytest.mark.anyio
    async def test_routes_non_streaming_request(self):
        body = {"model": "claude-3", "stream": False, "messages": [{"role": "user", "content": "hi"}]}
        req = _make_mock_request(body)
        event_bus = EventBus()

        with (
            patch("cross.proxy._proxy_streaming", new_callable=AsyncMock) as mock_streaming,
            patch("cross.proxy._proxy_simple", new_callable=AsyncMock) as mock_simple,
        ):
            mock_simple.return_value = MagicMock()
            await handle_proxy_request(req, event_bus)

            mock_simple.assert_called_once()
            mock_streaming.assert_not_called()

    @pytest.mark.anyio
    async def test_defaults_to_non_streaming(self):
        """When 'stream' key is absent, should route to _proxy_simple."""
        body = {"model": "claude-3", "messages": [{"role": "user", "content": "hi"}]}
        req = _make_mock_request(body)
        event_bus = EventBus()

        with (
            patch("cross.proxy._proxy_streaming", new_callable=AsyncMock) as mock_streaming,
            patch("cross.proxy._proxy_simple", new_callable=AsyncMock) as mock_simple,
        ):
            mock_simple.return_value = MagicMock()
            await handle_proxy_request(req, event_bus)

            mock_simple.assert_called_once()
            mock_streaming.assert_not_called()

    @pytest.mark.anyio
    async def test_publishes_request_event(self):
        body = {"model": "claude-3", "stream": False, "messages": [{"role": "user", "content": "testing"}]}
        req = _make_mock_request(body)
        event_bus = EventBus()
        events_received = []
        event_bus.subscribe(AsyncMock(side_effect=lambda e: events_received.append(e)))

        with patch("cross.proxy._proxy_simple", new_callable=AsyncMock) as mock_simple:
            mock_simple.return_value = MagicMock()
            await handle_proxy_request(req, event_bus)

        req_events = [e for e in events_received if isinstance(e, RequestEvent)]
        assert len(req_events) == 1
        assert req_events[0].model == "claude-3"

    @pytest.mark.anyio
    async def test_injects_blocked_tool_feedback(self):
        """Blocked tool feedback should be injected into the conversation before forwarding."""
        _blocked_tool_ids["toolu_blocked"] = "test reason"
        _blocked_tool_info["toolu_blocked"] = {"name": "Bash", "input": {"command": "bad"}}
        _blocked_tool_timestamps["toolu_blocked"] = time.time()

        body = {
            "model": "claude-3",
            "stream": False,
            "messages": [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "I will run this"}],
                },
                {
                    "role": "user",
                    "content": "ok go ahead",
                },
            ],
        }
        req = _make_mock_request(body)
        event_bus = EventBus()

        with patch("cross.proxy._proxy_simple", new_callable=AsyncMock) as mock_simple:
            mock_simple.return_value = MagicMock()
            await handle_proxy_request(req, event_bus)

            # Args: (client, event_bus, method, path, headers, body, gate_chain)
            call_args = mock_simple.call_args
            forwarded_body = call_args[0][5]  # body is the 6th positional arg (index 5)
            data = json.loads(forwarded_body)
            # Tool_use should be appended to assistant message
            assert data["messages"][0]["content"][-1]["type"] == "tool_use"
            assert data["messages"][0]["content"][-1]["id"] == "toolu_blocked"
            # Error tool_result should be prepended to user message
            user_content = data["messages"][1]["content"]
            assert user_content[0]["type"] == "tool_result"
            assert user_content[0]["is_error"] is True
            assert "Cross blocked" in user_content[0]["content"]

    @pytest.mark.anyio
    async def test_query_string_appended_to_path(self):
        body = {"model": "claude-3", "stream": False, "messages": []}
        req = _make_mock_request(body, query="beta=true")
        event_bus = EventBus()

        with patch("cross.proxy._proxy_simple", new_callable=AsyncMock) as mock_simple:
            mock_simple.return_value = MagicMock()
            await handle_proxy_request(req, event_bus)

            # Args: (client, event_bus, method, path, headers, body, gate_chain)
            call_args = mock_simple.call_args
            path_arg = call_args[0][3]  # path is the 4th positional arg (index 3)
            assert path_arg == "/v1/messages?beta=true"

    @pytest.mark.anyio
    async def test_passes_gate_chain_through(self):
        body = {"model": "claude-3", "stream": False, "messages": []}
        req = _make_mock_request(body)
        event_bus = EventBus()
        chain = GateChain(gates=[AllowGate(name="a")])

        with patch("cross.proxy._proxy_simple", new_callable=AsyncMock) as mock_simple:
            mock_simple.return_value = MagicMock()
            await handle_proxy_request(req, event_bus, gate_chain=chain)

            # Args: (client, event_bus, method, path, headers, body, gate_chain)
            call_args = mock_simple.call_args
            chain_arg = call_args[0][6]  # gate_chain is the 7th positional arg (index 6)
            assert chain_arg is chain

    @pytest.mark.anyio
    async def test_invalid_json_body_routes_to_non_streaming(self):
        """When body is not valid JSON, stream check fails silently and defaults to non-streaming."""
        req = MagicMock()
        req.method = "POST"
        req.url = MagicMock()
        req.url.path = "/v1/messages"
        req.url.query = ""
        req.headers = {"host": "localhost:8080", "content-type": "application/json"}
        req.body = AsyncMock(return_value=b"not json at all")
        event_bus = EventBus()

        with (
            patch("cross.proxy._proxy_streaming", new_callable=AsyncMock) as mock_streaming,
            patch("cross.proxy._proxy_simple", new_callable=AsyncMock) as mock_simple,
        ):
            mock_simple.return_value = MagicMock()
            await handle_proxy_request(req, event_bus)

            mock_simple.assert_called_once()
            mock_streaming.assert_not_called()

    @pytest.mark.anyio
    async def test_host_header_replaced(self):
        """The original host header should be replaced with api.anthropic.com."""
        body = {"model": "claude-3", "stream": False, "messages": []}
        req = _make_mock_request(body)
        event_bus = EventBus()

        with patch("cross.proxy._proxy_simple", new_callable=AsyncMock) as mock_simple:
            mock_simple.return_value = MagicMock()
            await handle_proxy_request(req, event_bus)

            call_args = mock_simple.call_args
            headers_arg = call_args[0][4]  # headers is the 5th positional arg (index 4)
            assert headers_arg["host"] == "api.anthropic.com"

    @pytest.mark.anyio
    async def test_content_length_updated_after_interception(self):
        """Content-length should match the body after blocked tool interception."""
        _blocked_tool_ids["toolu_x"] = "reason"
        _blocked_tool_timestamps["toolu_x"] = time.time()

        body = {
            "model": "claude-3",
            "stream": False,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_x", "content": "original"},
                    ],
                }
            ],
        }
        req = _make_mock_request(body)
        event_bus = EventBus()

        with patch("cross.proxy._proxy_simple", new_callable=AsyncMock) as mock_simple:
            mock_simple.return_value = MagicMock()
            await handle_proxy_request(req, event_bus)

            call_args = mock_simple.call_args
            headers_arg = call_args[0][4]
            forwarded_body = call_args[0][5]
            assert headers_arg["content-length"] == str(len(forwarded_body))

    @pytest.mark.anyio
    async def test_no_query_string(self):
        """When query string is empty, path should not have '?' appended."""
        body = {"model": "claude-3", "stream": False, "messages": []}
        req = _make_mock_request(body, query="")
        event_bus = EventBus()

        with patch("cross.proxy._proxy_simple", new_callable=AsyncMock) as mock_simple:
            mock_simple.return_value = MagicMock()
            await handle_proxy_request(req, event_bus)

            call_args = mock_simple.call_args
            path_arg = call_args[0][3]
            assert path_arg == "/v1/messages"
            assert "?" not in path_arg


# ============================================================
# _proxy_simple
# ============================================================


def _make_httpx_response(
    status_code: int = 200,
    content: bytes = b"{}",
    headers: dict | None = None,
) -> MagicMock:
    """Build a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.content = content
    resp.text = content.decode("utf-8", errors="replace")
    resp.headers = headers or {"content-type": "application/json"}
    return resp


class TestProxySimple:
    @pytest.mark.anyio
    async def test_basic_passthrough(self):
        from cross.proxy import _proxy_simple

        mock_client = AsyncMock()
        response_content = json.dumps({"content": [{"type": "text", "text": "Hello"}]}).encode()
        mock_client.request.return_value = _make_httpx_response(content=response_content)

        event_bus = EventBus()
        result = await _proxy_simple(
            mock_client,
            event_bus,
            "POST",
            "/v1/messages",
            {},
            b'{"messages": []}',
        )

        assert result.status_code == 200
        assert result.body == response_content

    @pytest.mark.anyio
    async def test_error_response_publishes_error_event(self):
        from cross.proxy import _proxy_simple

        mock_client = AsyncMock()
        mock_client.request.return_value = _make_httpx_response(
            status_code=429,
            content=b'{"error": "rate limited"}',
        )

        event_bus = EventBus()
        events_received = []
        event_bus.subscribe(AsyncMock(side_effect=lambda e: events_received.append(e)))

        result = await _proxy_simple(
            mock_client,
            event_bus,
            "POST",
            "/v1/messages",
            {},
            b'{"messages": []}',
        )

        assert result.status_code == 429
        error_events = [e for e in events_received if isinstance(e, ErrorEvent)]
        assert len(error_events) == 1
        assert error_events[0].status_code == 429

    @pytest.mark.anyio
    async def test_no_error_event_for_success(self):
        from cross.proxy import _proxy_simple

        mock_client = AsyncMock()
        mock_client.request.return_value = _make_httpx_response(status_code=200, content=b"{}")

        event_bus = EventBus()
        events_received = []
        event_bus.subscribe(AsyncMock(side_effect=lambda e: events_received.append(e)))

        await _proxy_simple(mock_client, event_bus, "POST", "/v1/messages", {}, b"{}")

        error_events = [e for e in events_received if isinstance(e, ErrorEvent)]
        assert len(error_events) == 0

    @pytest.mark.anyio
    async def test_gate_chain_applied_on_200(self):
        from cross.proxy import _proxy_simple

        mock_client = AsyncMock()
        response_body = {
            "content": [
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
            ]
        }
        mock_client.request.return_value = _make_httpx_response(
            content=json.dumps(response_body).encode(),
        )

        event_bus = EventBus()
        chain = GateChain(gates=[BlockGate(name="blocker")])

        result = await _proxy_simple(
            mock_client,
            event_bus,
            "POST",
            "/v1/messages",
            {},
            json.dumps({"messages": [{"role": "user", "content": "test"}], "model": "claude-3"}).encode(),
            gate_chain=chain,
        )

        # The tool should be removed from the response
        data = json.loads(result.body)
        assert len(data["content"]) == 0
        # And registered as blocked
        assert "t1" in _blocked_tool_ids

    @pytest.mark.anyio
    async def test_gate_chain_not_applied_on_error(self):
        from cross.proxy import _proxy_simple

        mock_client = AsyncMock()
        mock_client.request.return_value = _make_httpx_response(
            status_code=500,
            content=b'{"error": "internal"}',
        )

        event_bus = EventBus()
        chain = GateChain(gates=[BlockGate(name="blocker")])

        result = await _proxy_simple(
            mock_client,
            event_bus,
            "POST",
            "/v1/messages",
            {},
            b"{}",
            gate_chain=chain,
        )

        # Should return the error response unmodified
        assert result.status_code == 500
        assert json.loads(result.body)["error"] == "internal"

    @pytest.mark.anyio
    async def test_no_gate_chain_passes_content_through(self):
        from cross.proxy import _proxy_simple

        mock_client = AsyncMock()
        response_body = {
            "content": [
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "rm -rf /"}},
            ]
        }
        content = json.dumps(response_body).encode()
        mock_client.request.return_value = _make_httpx_response(content=content)

        event_bus = EventBus()

        result = await _proxy_simple(
            mock_client,
            event_bus,
            "POST",
            "/v1/messages",
            {},
            b"{}",
            gate_chain=None,
        )

        # Without a gate chain, content passes through unchanged
        assert result.body == content

    @pytest.mark.anyio
    async def test_strips_transfer_encoding(self):
        from cross.proxy import _proxy_simple

        mock_client = AsyncMock()
        mock_client.request.return_value = _make_httpx_response(
            headers={"content-type": "application/json", "transfer-encoding": "chunked", "content-encoding": "gzip"},
        )

        event_bus = EventBus()
        result = await _proxy_simple(
            mock_client,
            event_bus,
            "POST",
            "/v1/messages",
            {},
            b"{}",
        )

        # transfer-encoding and content-encoding should be stripped
        assert "transfer-encoding" not in result.headers
        assert "content-encoding" not in result.headers

    @pytest.mark.anyio
    async def test_gate_exception_does_not_crash(self):
        """If _gate_non_streaming_response raises, the response should still be returned."""
        from cross.proxy import _proxy_simple

        mock_client = AsyncMock()
        mock_client.request.return_value = _make_httpx_response(
            content=b"not valid json for gating",
        )

        event_bus = EventBus()
        chain = GateChain(gates=[AllowGate(name="a")])

        # _gate_non_streaming_response will raise json.JSONDecodeError on "not valid json"
        result = await _proxy_simple(
            mock_client,
            event_bus,
            "POST",
            "/v1/messages",
            {},
            b"{}",
            gate_chain=chain,
        )

        # Should still return the original content despite the gate error
        assert result.status_code == 200
        assert result.body == b"not valid json for gating"

    @pytest.mark.anyio
    async def test_forwards_method_and_path(self):
        """Verify the correct method and path are passed to httpx."""
        from cross.proxy import _proxy_simple

        mock_client = AsyncMock()
        mock_client.request.return_value = _make_httpx_response(content=b"{}")

        event_bus = EventBus()
        await _proxy_simple(
            mock_client,
            event_bus,
            "POST",
            "/v1/messages?beta=true",
            {"host": "api.anthropic.com"},
            b'{"messages": []}',
        )

        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args[0][0] == "POST"  # method
        assert call_args[0][1] == "/v1/messages?beta=true"  # path

    @pytest.mark.anyio
    async def test_error_body_truncated_to_500(self):
        """Error event body should be truncated to 500 chars."""
        from cross.proxy import _proxy_simple

        mock_client = AsyncMock()
        long_error = "x" * 1000
        mock_client.request.return_value = _make_httpx_response(
            status_code=400,
            content=long_error.encode(),
        )

        event_bus = EventBus()
        events_received = []
        event_bus.subscribe(AsyncMock(side_effect=lambda e: events_received.append(e)))

        await _proxy_simple(
            mock_client,
            event_bus,
            "POST",
            "/v1/messages",
            {},
            b"{}",
        )

        error_events = [e for e in events_received if isinstance(e, ErrorEvent)]
        assert len(error_events) == 1
        assert len(error_events[0].body) == 500

    @pytest.mark.anyio
    async def test_alert_action_passes_content_through(self):
        """ALERT action should allow the tool through (not block it)."""
        from cross.proxy import _proxy_simple

        mock_client = AsyncMock()
        response_body = {
            "content": [
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
            ]
        }
        mock_client.request.return_value = _make_httpx_response(
            content=json.dumps(response_body).encode(),
        )

        event_bus = EventBus()
        chain = GateChain(gates=[AlertGate(name="alerter")])

        result = await _proxy_simple(
            mock_client,
            event_bus,
            "POST",
            "/v1/messages",
            {},
            json.dumps({"messages": [{"role": "user", "content": "test"}]}).encode(),
            gate_chain=chain,
        )

        # Alert allows execution, so tool should remain
        data = json.loads(result.body)
        assert len(data["content"]) == 1
        assert data["content"][0]["type"] == "tool_use"
        assert "t1" not in _blocked_tool_ids


# ============================================================
# _proxy_streaming
# ============================================================


def _make_mock_upstream(sse_lines, status_code=200, headers=None):
    """Build a mock upstream response that yields SSE lines."""
    mock_upstream = AsyncMock()
    mock_upstream.status_code = status_code
    mock_upstream.headers = headers or {"content-type": "text/event-stream"}

    async def fake_aiter_lines():
        for line in sse_lines:
            yield line

    mock_upstream.aiter_lines = fake_aiter_lines
    mock_upstream.aclose = AsyncMock()
    return mock_upstream


def _make_mock_streaming_client(mock_upstream):
    """Build a mock httpx.AsyncClient that returns the given upstream."""
    mock_client = AsyncMock()
    mock_client.build_request.return_value = MagicMock()
    mock_client.send.return_value = mock_upstream
    return mock_client


def _streaming_request_body(model="claude-3", content="hi"):
    """Build a standard streaming request body."""
    return json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": content}],
            "stream": True,
        }
    ).encode()


async def _collect_streaming_output(result):
    """Consume a StreamingResponse and return all output chunks."""
    output = []
    async for chunk in result.body_iterator:
        output.append(chunk)
    return output


class TestProxyStreaming:
    """Tests for _proxy_streaming through the actual function with mock upstream."""

    @pytest.mark.anyio
    async def test_passthrough_no_gate(self):
        """Without a gate chain, all SSE lines should pass through."""

        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"m1","model":"claude-3","role":"assistant"}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}',
            "",
        ]

        upstream = _make_mock_upstream(sse_lines)
        client = _make_mock_streaming_client(upstream)
        event_bus = EventBus()

        result = await _proxy_streaming(
            client,
            event_bus,
            "POST",
            "/v1/messages",
            {"host": "api.anthropic.com"},
            _streaming_request_body(),
            gate_chain=None,
        )

        assert isinstance(result, StreamingResponse)
        assert result.status_code == 200

        output = await _collect_streaming_output(result)
        # All lines should pass through (each yielded with "\n" appended)
        assert len(output) == len(sse_lines)

    @pytest.mark.anyio
    async def test_text_streams_through_with_gate(self):
        """With a gate chain, text blocks should stream through immediately."""

        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"m1","model":"claude-3","role":"assistant"}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}',
            "",
        ]

        upstream = _make_mock_upstream(sse_lines)
        client = _make_mock_streaming_client(upstream)
        event_bus = EventBus()
        chain = GateChain(gates=[AllowGate(name="a")])

        result = await _proxy_streaming(
            client,
            event_bus,
            "POST",
            "/v1/messages",
            {"host": "api.anthropic.com"},
            _streaming_request_body(),
            gate_chain=chain,
        )

        output = await _collect_streaming_output(result)
        # Text blocks should pass through (not buffered), all lines present
        assert len(output) == len(sse_lines)

    @pytest.mark.anyio
    async def test_halted_tool_suppressed(self):
        """With a HALT_SESSION gate, tool_use SSE lines should be suppressed and response terminated."""

        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"m1","model":"claude-3","role":"assistant"}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"Bash"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"command\\": \\"rm -rf /\\"}"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":10}}',
            "",
        ]

        upstream = _make_mock_upstream(sse_lines)
        client = _make_mock_streaming_client(upstream)
        event_bus = EventBus()
        chain = GateChain(gates=[HaltSessionGate(name="halter")])

        result = await _proxy_streaming(
            client,
            event_bus,
            "POST",
            "/v1/messages",
            {"host": "api.anthropic.com"},
            _streaming_request_body(),
            gate_chain=chain,
        )

        output = await _collect_streaming_output(result)
        output_text = "".join(output)

        # Tool_use lines should NOT appear (suppressed)
        assert "rm -rf" not in output_text
        assert "content_block_start" not in output_text
        # Synthetic message_stop should appear (clean termination)
        assert "message_stop" in output_text
        assert "end_turn" in output_text
        # Tool should be in the blocked list for next-request feedback injection
        assert "toolu_1" in _blocked_tool_ids
        assert "toolu_1" in _blocked_tool_info

    @pytest.mark.anyio
    async def test_allowed_tool_flushed(self):
        """With an allowing gate, tool_use SSE lines should be flushed through."""

        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"m1","model":"claude-3","role":"assistant"}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"Read"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"file_path\\": \\"/tmp/x\\"}"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":10}}',
            "",
        ]

        upstream = _make_mock_upstream(sse_lines)
        client = _make_mock_streaming_client(upstream)
        event_bus = EventBus()
        chain = GateChain(gates=[AllowGate(name="allower")])

        result = await _proxy_streaming(
            client,
            event_bus,
            "POST",
            "/v1/messages",
            {"host": "api.anthropic.com"},
            _streaming_request_body(),
            gate_chain=chain,
        )

        output = await _collect_streaming_output(result)

        # Tool should pass through
        output_text = "".join(output)
        assert "toolu_1" in output_text
        assert "Read" in output_text
        assert "toolu_1" not in _blocked_tool_ids

    @pytest.mark.anyio
    async def test_publishes_gate_decision_event(self):
        """Gate decisions should be published as GateDecisionEvent."""

        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"m1","model":"claude-3","role":"assistant"}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"Bash"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"command\\": \\"ls\\"}"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":10}}',
            "",
        ]

        upstream = _make_mock_upstream(sse_lines)
        client = _make_mock_streaming_client(upstream)
        event_bus = EventBus()
        events_received = []
        event_bus.subscribe(AsyncMock(side_effect=lambda e: events_received.append(e)))
        chain = GateChain(gates=[HaltSessionGate(name="halter")])

        result = await _proxy_streaming(
            client,
            event_bus,
            "POST",
            "/v1/messages",
            {"host": "api.anthropic.com"},
            _streaming_request_body(),
            gate_chain=chain,
        )

        await _collect_streaming_output(result)

        gate_events = [e for e in events_received if isinstance(e, GateDecisionEvent)]
        assert len(gate_events) == 1
        assert gate_events[0].tool_use_id == "toolu_1"
        assert gate_events[0].tool_name == "Bash"
        assert gate_events[0].action == "halt_session"

    @pytest.mark.anyio
    async def test_strips_transfer_and_content_encoding(self):
        """Transfer-encoding and content-encoding should be stripped from streaming headers."""

        upstream = _make_mock_upstream(
            ["data: [DONE]"],
            headers={
                "content-type": "text/event-stream",
                "transfer-encoding": "chunked",
                "content-encoding": "gzip",
            },
        )
        client = _make_mock_streaming_client(upstream)
        event_bus = EventBus()

        result = await _proxy_streaming(
            client,
            event_bus,
            "POST",
            "/v1/messages",
            {"host": "api.anthropic.com"},
            _streaming_request_body(),
        )

        assert "transfer-encoding" not in result.headers
        assert "content-encoding" not in result.headers

    @pytest.mark.anyio
    async def test_closes_upstream(self):
        """The upstream connection should be closed after streaming completes."""

        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"m1","model":"claude-3"}}',
            "",
        ]

        upstream = _make_mock_upstream(sse_lines)
        client = _make_mock_streaming_client(upstream)
        event_bus = EventBus()

        result = await _proxy_streaming(
            client,
            event_bus,
            "POST",
            "/v1/messages",
            {"host": "api.anthropic.com"},
            _streaming_request_body(),
        )

        await _collect_streaming_output(result)
        upstream.aclose.assert_called_once()

    @pytest.mark.anyio
    async def test_cascade_halt(self):
        """When first tool is halted, response is terminated — second tool never reaches client."""

        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"m1","model":"claude-3","role":"assistant"}}',
            "",
            # First tool -- halted by HaltSessionGate
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_bash","name":"Bash"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"command\\": \\"rm -rf /\\"}"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            # Second tool -- never processed because response terminates after first halt
            "event: content_block_start",
            'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_read","name":"Read"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"file_path\\": \\"/tmp/safe\\"}"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":1}',
            "",
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":20}}',
            "",
        ]

        upstream = _make_mock_upstream(sse_lines)
        client = _make_mock_streaming_client(upstream)
        event_bus = EventBus()
        chain = GateChain(gates=[HaltSessionGate(name="halter")])

        result = await _proxy_streaming(
            client,
            event_bus,
            "POST",
            "/v1/messages",
            {"host": "api.anthropic.com"},
            _streaming_request_body(),
            gate_chain=chain,
        )

        output = await _collect_streaming_output(result)
        output_text = "".join(output)

        # First tool halted and suppressed
        assert "toolu_bash" in _blocked_tool_ids
        assert "rm -rf" not in output_text
        # Second tool never reached client (response terminated after first halt)
        assert "toolu_read" not in output_text
        # Synthetic message_stop ends the response
        assert "message_stop" in output_text

    @pytest.mark.anyio
    async def test_user_intent_passed_to_gate(self):
        """User intent from request body should be passed to gate requests."""

        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"m1","model":"claude-3-opus","role":"assistant"}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"Read"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"file_path\\": \\"/tmp/x\\"}"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}',
            "",
        ]

        upstream = _make_mock_upstream(sse_lines)
        client = _make_mock_streaming_client(upstream)
        event_bus = EventBus()
        capturing = CapturingGate()
        chain = GateChain(gates=[capturing])

        result = await _proxy_streaming(
            client,
            event_bus,
            "POST",
            "/v1/messages",
            {"host": "api.anthropic.com"},
            _streaming_request_body(model="claude-3-opus", content="Please read the file"),
            gate_chain=chain,
        )

        await _collect_streaming_output(result)

        assert len(capturing.requests) == 1
        assert capturing.requests[0].user_intent == "Please read the file"
        assert capturing.requests[0].agent == "claude-3-opus"
        assert capturing.requests[0].tool_name == "Read"

    @pytest.mark.anyio
    async def test_recent_tools_updated_during_streaming(self):
        """Recent tools deque should be updated as tools are processed."""

        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"m1","model":"claude-3","role":"assistant"}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"Read"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"file_path\\": \\"/a\\"}"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_2","name":"Write"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"file_path\\": \\"/b\\"}"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":1}',
            "",
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":10}}',
            "",
        ]

        upstream = _make_mock_upstream(sse_lines)
        client = _make_mock_streaming_client(upstream)
        event_bus = EventBus()
        chain = GateChain(gates=[AllowGate(name="a")])

        result = await _proxy_streaming(
            client,
            event_bus,
            "POST",
            "/v1/messages",
            {"host": "api.anthropic.com"},
            _streaming_request_body(),
            gate_chain=chain,
        )

        await _collect_streaming_output(result)

        assert len(_recent_tools) == 2
        assert _recent_tools[0]["name"] == "Read"
        assert _recent_tools[1]["name"] == "Write"

    @pytest.mark.anyio
    async def test_alert_tool_flushed(self):
        """ALERT action should allow the tool through (flush buffer)."""

        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"m1","model":"claude-3","role":"assistant"}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"Bash"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"command\\": \\"ls\\"}"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}',
            "",
        ]

        upstream = _make_mock_upstream(sse_lines)
        client = _make_mock_streaming_client(upstream)
        event_bus = EventBus()
        chain = GateChain(gates=[AlertGate(name="alerter")])

        result = await _proxy_streaming(
            client,
            event_bus,
            "POST",
            "/v1/messages",
            {"host": "api.anthropic.com"},
            _streaming_request_body(),
            gate_chain=chain,
        )

        output = await _collect_streaming_output(result)

        # Alert allows execution, tool lines should be in output
        output_text = "".join(output)
        assert "toolu_1" in output_text
        assert "toolu_1" not in _blocked_tool_ids

    @pytest.mark.anyio
    async def test_bad_request_body_doesnt_crash(self):
        """Invalid JSON in request body should not crash streaming."""

        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"m1","model":"claude-3"}}',
            "",
        ]

        upstream = _make_mock_upstream(sse_lines)
        client = _make_mock_streaming_client(upstream)
        event_bus = EventBus()

        result = await _proxy_streaming(
            client,
            event_bus,
            "POST",
            "/v1/messages",
            {"host": "api.anthropic.com"},
            b"not valid json",
            gate_chain=None,
        )

        output = await _collect_streaming_output(result)
        # Should still work, just with empty user_intent/model
        assert len(output) == len(sse_lines)

    @pytest.mark.anyio
    async def test_buffer_overflow_flushes_without_gate(self):
        """When buffer exceeds _MAX_BUFFER_LINES, it should flush without gate evaluation."""

        # Build a tool_use block with enough delta lines to exceed the buffer limit
        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"m1","model":"claude-3","role":"assistant"}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_big","name":"Bash"}}',
            "",
        ]
        # Add many delta lines to exceed buffer limit
        for i in range(_MAX_BUFFER_LINES + 10):
            sse_lines.extend(
                [
                    "event: content_block_delta",
                    'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"x"}}',
                    "",
                ]
            )
        sse_lines.extend(
            [
                "event: content_block_stop",
                'data: {"type":"content_block_stop","index":0}',
                "",
                "event: message_delta",
                'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}',
                "",
            ]
        )

        upstream = _make_mock_upstream(sse_lines)
        client = _make_mock_streaming_client(upstream)
        event_bus = EventBus()
        chain = GateChain(gates=[BlockGate(name="blocker")])

        result = await _proxy_streaming(
            client,
            event_bus,
            "POST",
            "/v1/messages",
            {"host": "api.anthropic.com"},
            _streaming_request_body(),
            gate_chain=chain,
        )

        output = await _collect_streaming_output(result)

        # The buffer overflow should have flushed the tool lines through
        # without gate evaluation, so the tool should NOT be blocked
        output_text = "".join(output)
        assert "toolu_big" in output_text

    @pytest.mark.anyio
    async def test_trailing_pending_line_flushed(self):
        """A pending event: line at end of stream should be flushed."""

        # Stream ends with an event: line that never gets a following data: line
        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"m1","model":"claude-3","role":"assistant"}}',
            "",
            "event: orphaned_event",
        ]

        upstream = _make_mock_upstream(sse_lines)
        client = _make_mock_streaming_client(upstream)
        event_bus = EventBus()
        chain = GateChain(gates=[AllowGate(name="a")])

        result = await _proxy_streaming(
            client,
            event_bus,
            "POST",
            "/v1/messages",
            {"host": "api.anthropic.com"},
            _streaming_request_body(),
            gate_chain=chain,
        )

        output = await _collect_streaming_output(result)
        output_text = "".join(output)

        # The orphaned event: line should still appear in output
        assert "orphaned_event" in output_text


# ============================================================
# _gate_non_streaming_response
# ============================================================


class TestGateNonStreamingResponse:
    @pytest.mark.anyio
    async def test_no_tool_use_blocks(self):
        content = json.dumps({"content": [{"type": "text", "text": "Hello"}]}).encode()
        event_bus = EventBus()
        chain = GateChain(gates=[BlockGate(name="b")])

        result = await _gate_non_streaming_response(content, b"{}", chain, event_bus)
        assert result == content  # unchanged

    @pytest.mark.anyio
    async def test_single_tool_blocked(self):
        content = json.dumps(
            {
                "stop_reason": "tool_use",
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
                ],
            }
        ).encode()
        event_bus = EventBus()
        chain = GateChain(gates=[BlockGate(name="b")])

        result = await _gate_non_streaming_response(content, b"{}", chain, event_bus)
        data = json.loads(result)
        assert len(data["content"]) == 0
        assert data["stop_reason"] == "end_turn"
        assert "t1" in _blocked_tool_ids
        assert "t1" in _blocked_tool_info
        assert "t1" in _blocked_tool_timestamps

    @pytest.mark.anyio
    async def test_single_tool_allowed(self):
        content = json.dumps(
            {
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/tmp/x"}},
                ]
            }
        ).encode()
        event_bus = EventBus()
        chain = GateChain(gates=[AllowGate(name="a")])

        result = await _gate_non_streaming_response(content, b"{}", chain, event_bus)
        assert result == content  # unchanged, tool allowed
        assert "t1" not in _blocked_tool_ids

    @pytest.mark.anyio
    async def test_cascade_blocking(self):
        """When first tool is blocked, subsequent tools are cascade-blocked."""
        content = json.dumps(
            {
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "rm -rf /"}},
                    {"type": "tool_use", "id": "t2", "name": "Read", "input": {"file_path": "/safe"}},
                ]
            }
        ).encode()
        event_bus = EventBus()
        chain = GateChain(gates=[SelectiveBlockGate(name="selective")])

        result = await _gate_non_streaming_response(content, b"{}", chain, event_bus)
        data = json.loads(result)
        assert len(data["content"]) == 0

        assert "t1" in _blocked_tool_ids
        assert "Bash is blocked" in _blocked_tool_ids["t1"]
        assert "t2" in _blocked_tool_ids
        assert "Preceding tool" in _blocked_tool_ids["t2"]

    @pytest.mark.anyio
    async def test_mixed_text_and_tool_blocks(self):
        """Text blocks are preserved when a tool_use is blocked."""
        content = json.dumps(
            {
                "content": [
                    {"type": "text", "text": "I will run a command"},
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
                ]
            }
        ).encode()
        event_bus = EventBus()
        chain = GateChain(gates=[BlockGate(name="b")])

        result = await _gate_non_streaming_response(content, b"{}", chain, event_bus)
        data = json.loads(result)
        # Text block should remain, tool_use should be removed
        assert len(data["content"]) == 1
        assert data["content"][0]["type"] == "text"

    @pytest.mark.anyio
    async def test_escalate_waits_for_approval_then_times_out(self):
        content = json.dumps(
            {
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}},
                ]
            }
        ).encode()
        event_bus = EventBus()
        chain = GateChain(gates=[EscalateGate(name="e")])

        # Use a tiny timeout so the test doesn't hang
        with unittest.mock.patch.object(settings, "gate_approval_timeout", 0.01):
            _blocked_tool_ids.pop("t1", None)
            result = await _gate_non_streaming_response(content, b"{}", chain, event_bus)
        data = json.loads(result)
        assert len(data["content"]) == 0
        assert "t1" in _blocked_tool_ids

    @pytest.mark.anyio
    async def test_escalate_approved_allows_tool(self):
        content = json.dumps(
            {
                "content": [
                    {"type": "tool_use", "id": "t2", "name": "Bash", "input": {}},
                ]
            }
        ).encode()
        event_bus = EventBus()
        chain = GateChain(gates=[EscalateGate(name="e")])

        # Resolve approval shortly after escalation
        async def approve_soon():
            await asyncio.sleep(0.01)
            resolve_gate_approval("t2", approved=True, username="testuser")

        _blocked_tool_ids.pop("t2", None)
        asyncio.get_event_loop().create_task(approve_soon())
        result = await _gate_non_streaming_response(content, b"{}", chain, event_bus)
        data = json.loads(result)
        # Tool should still be in the response (not blocked)
        assert len(data["content"]) == 1
        assert "t2" not in _blocked_tool_ids

    @pytest.mark.anyio
    async def test_user_intent_extracted_from_request(self):
        content = json.dumps(
            {
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/tmp/x"}},
                ]
            }
        ).encode()
        request_body = json.dumps(
            {
                "model": "claude-3-opus",
                "messages": [{"role": "user", "content": "Please read the file"}],
            }
        ).encode()

        event_bus = EventBus()
        capturing = CapturingGate()
        chain = GateChain(gates=[capturing])

        await _gate_non_streaming_response(content, request_body, chain, event_bus)

        assert len(capturing.requests) == 1
        assert capturing.requests[0].user_intent == "Please read the file"
        assert capturing.requests[0].agent == "claude-3-opus"

    @pytest.mark.anyio
    async def test_bad_request_body_doesnt_crash(self):
        content = json.dumps(
            {
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
                ]
            }
        ).encode()
        event_bus = EventBus()
        chain = GateChain(gates=[AllowGate(name="a")])

        # Bad JSON in request body
        result = await _gate_non_streaming_response(content, b"not json", chain, event_bus)
        # Should still process the response content
        assert result == content

    @pytest.mark.anyio
    async def test_recent_tools_populated(self):
        content = json.dumps(
            {
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/a"}},
                    {"type": "tool_use", "id": "t2", "name": "Write", "input": {"file_path": "/b"}},
                ]
            }
        ).encode()
        event_bus = EventBus()
        chain = GateChain(gates=[AllowGate(name="a")])

        await _gate_non_streaming_response(content, b"{}", chain, event_bus)

        assert len(_recent_tools) == 2
        assert _recent_tools[0]["name"] == "Read"
        assert _recent_tools[1]["name"] == "Write"

    @pytest.mark.anyio
    async def test_recent_tools_provided_to_gate(self):
        """Second tool's GateRequest should include the first tool in recent_tools."""
        # Pre-populate recent_tools to verify they are passed
        _recent_tools.append({"name": "Glob", "input": {"pattern": "*.py"}})

        content = json.dumps(
            {
                "content": [
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file_path": "/a"}},
                    {"type": "tool_use", "id": "t2", "name": "Write", "input": {"file_path": "/b"}},
                ]
            }
        ).encode()
        event_bus = EventBus()
        capturing = CapturingGate()
        chain = GateChain(gates=[capturing])

        await _gate_non_streaming_response(content, b"{}", chain, event_bus)

        # First tool should see the pre-populated recent tool
        assert len(capturing.requests[0].recent_tools) == 1
        assert capturing.requests[0].recent_tools[0]["name"] == "Glob"

        # Second tool should see Glob + Read
        assert len(capturing.requests[1].recent_tools) == 2
        assert capturing.requests[1].recent_tools[0]["name"] == "Glob"
        assert capturing.requests[1].recent_tools[1]["name"] == "Read"

    @pytest.mark.anyio
    async def test_empty_content_list(self):
        content = json.dumps({"content": []}).encode()
        event_bus = EventBus()
        chain = GateChain(gates=[BlockGate(name="b")])

        result = await _gate_non_streaming_response(content, b"{}", chain, event_bus)
        assert result == content

    @pytest.mark.anyio
    async def test_tool_index_in_message_set(self):
        """tool_index_in_message should reflect position within content blocks."""
        content = json.dumps(
            {
                "content": [
                    {"type": "text", "text": "some text"},
                    {"type": "tool_use", "id": "t1", "name": "Read", "input": {}},
                    {"type": "tool_use", "id": "t2", "name": "Write", "input": {}},
                ]
            }
        ).encode()
        event_bus = EventBus()
        capturing = CapturingGate()
        chain = GateChain(gates=[capturing])

        await _gate_non_streaming_response(content, b"{}", chain, event_bus)

        # tool_index_in_message uses the block index from the content list (i variable)
        assert capturing.requests[0].tool_index_in_message == 1  # index 1 in blocks
        assert capturing.requests[1].tool_index_in_message == 2  # index 2 in blocks


# ============================================================
# shutdown
# ============================================================


class TestShutdown:
    @pytest.mark.anyio
    async def test_shutdown_closes_client(self):
        client = get_client()
        with patch.object(client, "aclose", new_callable=AsyncMock) as mock_close:
            await shutdown()
            mock_close.assert_called_once()

    @pytest.mark.anyio
    async def test_shutdown_with_no_client(self):
        # _client is already None from fixture
        # Should not raise
        await shutdown()


# ============================================================
# _proxy_streaming — additional edge-case coverage
# ============================================================


class TestProxyStreamingNonToolEventDuringBuffering:
    """Cover line 391: non-ToolUseEvent published while buffering a tool_use block.

    When the parser emits a non-ToolUseEvent from a line processed during buffering
    (e.g., a message_delta interleaved mid-tool), that event should be published
    through the else branch at line 391.
    """

    @pytest.mark.anyio
    async def test_non_tool_event_published_during_tool_buffering(self):
        """Inject a message_delta event mid-tool-buffer so parser emits a
        MessageDeltaEvent while buffering is True. This hits the else branch (line 391)."""

        # Construct a stream where a message_delta event occurs mid-tool-buffer.
        # After the tool_use content_block_start, we inject a full message_delta event
        # (event: + data: + blank) before the tool_use content_block_stop.
        # The blank line after the message_delta data will cause the parser to emit
        # a MessageDeltaEvent while the code is in the buffering state.
        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"m1","model":"claude-3","role":"assistant"}}',
            "",
            # Start tool_use block (triggers buffering)
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"Read"}}',
            "",
            # Tool input delta (normal, during buffering)
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"file_path\\":"}}',
            "",
            # Inject a message_delta mid-buffer -- the blank line will cause SSEParser
            # to emit a MessageDeltaEvent while we are still in the buffering state.
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":null},"usage":{"output_tokens":10}}',
            "",
            # More tool input
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":" \\"/tmp/x\\"}"}}',
            "",
            # End tool_use block
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
        ]

        upstream = _make_mock_upstream(sse_lines)
        client = _make_mock_streaming_client(upstream)
        event_bus = EventBus()
        events_received = []
        event_bus.subscribe(AsyncMock(side_effect=lambda e: events_received.append(e)))
        chain = GateChain(gates=[AllowGate(name="a")])

        result = await _proxy_streaming(
            client,
            event_bus,
            "POST",
            "/v1/messages",
            {"host": "api.anthropic.com"},
            _streaming_request_body(),
            gate_chain=chain,
        )

        await _collect_streaming_output(result)

        # The MessageDeltaEvent should have been published (via line 391)
        from cross.events import MessageDeltaEvent

        delta_events = [e for e in events_received if isinstance(e, MessageDeltaEvent)]
        assert len(delta_events) >= 1
        assert delta_events[0].output_tokens == 10


class TestProxyStreamingParserFlushAtEnd:
    """Cover line 470: parser.feed_line('') at end of stream produces events.

    When the stream ends without a trailing blank line after data: lines,
    the parser has accumulated data that hasn't been processed. The final
    parser.feed_line('') call should flush those events.
    """

    @pytest.mark.anyio
    async def test_parser_flush_emits_events_at_stream_end(self):
        """Stream ends with data: line but no trailing blank -- parser flush should emit."""

        # The stream ends with "data: ..." line for a message_delta, but no blank line.
        # When the async for loop ends, the code calls parser.feed_line("") which
        # processes the accumulated event+data and returns a MessageDeltaEvent.
        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"m1","model":"claude-3","role":"assistant"}}',
            "",
            # message_delta event WITHOUT trailing blank line
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":42}}',
            # No trailing "" -- parser won't emit until feed_line("") is called
        ]

        upstream = _make_mock_upstream(sse_lines)
        client = _make_mock_streaming_client(upstream)
        event_bus = EventBus()
        events_received = []
        event_bus.subscribe(AsyncMock(side_effect=lambda e: events_received.append(e)))

        # Use no gate_chain to test the post-loop flush path (line 467-470)
        result = await _proxy_streaming(
            client,
            event_bus,
            "POST",
            "/v1/messages",
            {"host": "api.anthropic.com"},
            _streaming_request_body(),
            gate_chain=None,
        )

        await _collect_streaming_output(result)

        # The MessageDeltaEvent should have been emitted by the parser flush
        from cross.events import MessageDeltaEvent

        delta_events = [e for e in events_received if isinstance(e, MessageDeltaEvent)]
        assert len(delta_events) == 1
        assert delta_events[0].stop_reason == "end_turn"
        assert delta_events[0].output_tokens == 42

    @pytest.mark.anyio
    async def test_parser_flush_with_gate_chain(self):
        """Same as above but with gate_chain active."""

        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"m1","model":"claude-3","role":"assistant"}}',
            "",
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":99}}',
            # No trailing blank
        ]

        upstream = _make_mock_upstream(sse_lines)
        client = _make_mock_streaming_client(upstream)
        event_bus = EventBus()
        events_received = []
        event_bus.subscribe(AsyncMock(side_effect=lambda e: events_received.append(e)))
        chain = GateChain(gates=[AllowGate(name="a")])

        result = await _proxy_streaming(
            client,
            event_bus,
            "POST",
            "/v1/messages",
            {"host": "api.anthropic.com"},
            _streaming_request_body(),
            gate_chain=chain,
        )

        await _collect_streaming_output(result)

        from cross.events import MessageDeltaEvent

        delta_events = [e for e in events_received if isinstance(e, MessageDeltaEvent)]
        assert len(delta_events) == 1
        assert delta_events[0].output_tokens == 99


class TestProxyStreamingBufferOverflowWithEvents:
    """Cover line 378: events published during buffer overflow.

    When the buffer overflows on a blank line that causes the parser to emit events,
    those events should be published (line 377-378).
    """

    @pytest.mark.anyio
    async def test_buffer_overflow_publishes_parser_events(self):
        """Buffer overflow triggered by a blank line that produces parser events."""

        # Build a tool_use block. We need the buffer to exceed _MAX_BUFFER_LINES
        # on a blank line that also causes the parser to emit events.
        #
        # Strategy: fill the buffer with content_block_delta lines (each group of 3
        # adds 3 lines to buffer: event, data, blank). Then add a message_delta event
        # whose blank line triggers both the overflow AND a parser event emission.
        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"m1","model":"claude-3","role":"assistant"}}',
            "",
            # Start tool_use block
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_of","name":"Read"}}',
            "",
        ]

        # Fill buffer to just under the limit.
        # After tool_start, the buffer has: [event:content_block_start, data:..., ""]
        # Wait, actually: the event: line is held as pending_line, then pulled into buffer.
        # The data: line triggers _is_tool_use_block_start, so buffer starts with
        # [pending_line="event: content_block_start", "data: ..."].
        # Then the blank line is added to buffer: ["event:..", "data:..", ""]
        # So after tool_start SSE event, buffer has 3 lines.
        #
        # Each delta group adds 3 lines (event:, data:, blank).
        # But wait, in the gated path:
        #   - "event:" line: pending_line is set, line is NOT added to buffer (line 365-369)
        #     But we ARE in buffering mode, so the code hits line 371 first.
        #     Wait no -- the event: check is at line 365, BEFORE the buffering check at 371.
        #     Let me re-read the code flow:
        #
        # if not buffering and _is_tool_use_block_start(line):  # line 348
        #     buffering = True; buffer.append...; continue
        # if pending_line is not None:                          # line 360
        #     yield pending_line; pending_line = None
        # if line.startswith("event: "):                        # line 365
        #     pending_line = line; continue
        # if buffering:                                         # line 371
        #     buffer.append(line)
        #
        # So when buffering=True and line starts with "event: ":
        # - Line 348: buffering is True, so `not buffering` is False, skip
        # - Line 360: pending_line might be set from previous event: line, flush it
        # - Line 365: line starts with "event: ", set pending_line, continue
        #
        # So event: lines during buffering are NOT added to buffer! They become pending.
        # Then the next data: line: not buffering check fails, pending_line is flushed
        # (yield pending_line), then the line goes to `if buffering:` and is added to buffer.
        # Wait, but pending_line is flushed via yield, which means it goes to OUTPUT not buffer.
        # That seems wrong for tool buffering. Let me re-read...
        #
        # Actually, line 360: `yield pending_line + "\n"` -- this yields the line to the
        # output stream, NOT to the buffer. So during buffering, event: lines leak to output.
        # And the data: line after it goes into the buffer.
        # And the blank line goes into the buffer.
        #
        # So for each delta (event:, data:, blank) during buffering:
        # - event: -> becomes pending_line (not in buffer)
        # - data: -> pending_line is yielded to output, data: goes into buffer
        # - blank: -> goes into buffer
        #
        # So each delta adds 2 lines to buffer (data:, blank), not 3.
        #
        # After tool_start, buffer has:
        #   [event: content_block_start (from pending), data: content_block_start, ""]
        # Wait, the blank line after tool_start: by the time we process the blank line,
        # buffering=True. The blank line doesn't start with "event:", so it goes to
        # `if buffering:` and is added to buffer.
        #
        # Let me trace carefully for _tool_start which produces:
        #   "event: content_block_start"
        #   "data: {tool_use...}"
        #   ""
        #
        # Line "event: content_block_start":
        #   - not buffering=True, skip 348
        #   - pending_line is None, skip 360
        #   - starts with "event:", set pending_line="event: content_block_start", continue
        #
        # Line "data: {tool_use...}":
        #   - not buffering=True, skip 348... wait, buffering is currently False!
        #     _is_tool_use_block_start("data: {tool_use...}") = True!
        #   - So we enter: buffering=True, buffer=[]
        #     pending_line is "event: content_block_start" (not None)
        #     buffer.append("event: content_block_start")
        #     pending_line = None
        #     buffer.append("data: {tool_use...}")
        #   - continue
        #   Buffer now has 2 items.
        #
        # Line "":
        #   - not buffering is False, skip 348
        #   - pending_line is None, skip 360
        #   - doesn't start with "event:", skip 365
        #   - buffering is True: buffer.append("")
        #   Buffer now has 3 items.
        #   - len(buffer)=3, not > 500, skip overflow
        #   - check events: parser emits [] for content_block_start (no CrossEvent)
        #   - tool_event = None, not tool_event, skip to continue
        #
        # Now for each _tool_delta: "event: content_block_delta", "data: ...", ""
        #
        # Line "event: content_block_delta":
        #   - not buffering=False, skip 348
        #   - pending_line is None, skip 360
        #   - starts with "event:", set pending_line, continue
        #
        # Line "data: {content_block_delta...}":
        #   - not buffering=False, skip 348
        #   - pending_line is "event:...", yield it to output, pending_line=None
        #   - doesn't start with "event:", skip 365
        #   - buffering is True: buffer.append("data:...")
        #   Buffer grows by 1.
        #
        # Line "":
        #   - not buffering=False, skip 348
        #   - pending_line is None, skip 360
        #   - doesn't start with "event:", skip 365
        #   - buffering is True: buffer.append("")
        #   Buffer grows by 1.
        #
        # So each delta adds 2 lines to buffer. We start with 3.
        # To exceed 500: need (500 - 3) / 2 = 248.5, so 249 deltas.
        # The 249th delta's blank line would put buffer at 3 + 249*2 = 501 > 500.
        #
        # But the overflow check (line 375) happens AFTER the append.
        # So the overflow triggers on the blank line after the 249th delta.
        # That blank line causes the parser to emit events for content_block_delta,
        # which for input_json_delta just accumulates state and returns [].
        #
        # To make the parser emit events on the overflow line, I need to end with
        # a different kind of event. Let me use content_block_stop (which emits ToolUseEvent).
        # But the ToolUseEvent would be handled by the tool_event check, not the overflow.
        # The overflow check is BEFORE the tool_event check.
        #
        # Actually, the order in the code is:
        #   buffer.append(line)        # line 372
        #   if overflow: ...           # line 375
        #   tool_event = None          # line 386
        #   for ev in events: ...      # line 387
        #
        # So if the overflow triggers, we flush and continue BEFORE checking tool events.
        # The overflow branch publishes events from line 377-378.
        #
        # To get events on the overflow line, I need the parser to emit CrossEvents
        # on that particular line. The overflow triggers when buffer exceeds 500.
        #
        # Idea: Use exactly enough deltas to bring buffer to 499, then add a
        # message_delta event. The message_delta's data: line adds 1 (buffer=500),
        # then the blank line adds 1 (buffer=501 > 500), AND the parser emits
        # MessageDeltaEvent on that blank line.
        #
        # Let me calculate: 3 (from tool_start) + N*2 (from deltas) = 499
        # N = (499-3)/2 = 248. So 248 deltas bring buffer to 3+496=499.
        # Then message_delta's data: adds 1 (500), blank adds 1 (501 > 500).
        # The parser processes the message_delta and emits MessageDeltaEvent!

        # 248 deltas to get buffer to 499
        num_deltas = (_MAX_BUFFER_LINES - 3) // 2  # = 248 for _MAX_BUFFER_LINES=500
        for i in range(num_deltas):
            sse_lines.extend(
                [
                    "event: content_block_delta",
                    'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"x"}}',
                    "",
                ]
            )

        # Now buffer should be at 3 + 248*2 = 499.
        # Add a message_delta event: its data: line brings buffer to 500,
        # its blank line brings buffer to 501 > 500, triggering overflow
        # AND the parser emits MessageDeltaEvent.
        sse_lines.extend(
            [
                "event: message_delta",
                'data: {"type":"message_delta","delta":{"stop_reason":null},"usage":{"output_tokens":77}}',
                "",
            ]
        )

        # Add remaining lines to complete the stream
        sse_lines.extend(
            [
                "event: content_block_stop",
                'data: {"type":"content_block_stop","index":0}',
                "",
                "event: message_delta",
                'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}',
                "",
            ]
        )

        upstream = _make_mock_upstream(sse_lines)
        client = _make_mock_streaming_client(upstream)
        event_bus = EventBus()
        events_received = []
        event_bus.subscribe(AsyncMock(side_effect=lambda e: events_received.append(e)))
        chain = GateChain(gates=[AllowGate(name="a")])

        result = await _proxy_streaming(
            client,
            event_bus,
            "POST",
            "/v1/messages",
            {"host": "api.anthropic.com"},
            _streaming_request_body(),
            gate_chain=chain,
        )

        output = await _collect_streaming_output(result)

        # The buffer should have overflowed, flushing all lines
        output_text = "".join(output)
        assert "toolu_of" in output_text

        # The MessageDeltaEvent from the overflow line should have been published (line 378)
        from cross.events import MessageDeltaEvent

        delta_events = [e for e in events_received if isinstance(e, MessageDeltaEvent)]
        # At least one delta event should be the one from the overflow path
        assert any(e.output_tokens == 77 for e in delta_events)


# ============================================================
# _build_retry_request_body
# ============================================================


class TestBuildRetryRequestBody:
    def test_basic_retry_body(self):
        original = json.dumps(
            {
                "model": "claude-3",
                "messages": [{"role": "user", "content": "do something"}],
                "stream": True,
            }
        ).encode()
        blocked_tools = [
            {
                "tool_use_id": "t1",
                "name": "Bash",
                "input": {"command": "rm -rf /"},
                "reason": "dangerous command",
            }
        ]
        result = _build_retry_request_body(original, blocked_tools)
        data = json.loads(result)

        assert len(data["messages"]) == 3
        # Original user message
        assert data["messages"][0]["role"] == "user"
        # Appended assistant message with tool_use
        assert data["messages"][1]["role"] == "assistant"
        assert data["messages"][1]["content"][0]["type"] == "tool_use"
        assert data["messages"][1]["content"][0]["id"] == "t1"
        assert data["messages"][1]["content"][0]["name"] == "Bash"
        # Appended user message with error tool_result
        assert data["messages"][2]["role"] == "user"
        assert data["messages"][2]["content"][0]["type"] == "tool_result"
        assert data["messages"][2]["content"][0]["tool_use_id"] == "t1"
        assert data["messages"][2]["content"][0]["is_error"] is True
        assert "Cross blocked" in data["messages"][2]["content"][0]["content"]

    def test_multiple_blocked_tools(self):
        original = json.dumps(
            {
                "model": "claude-3",
                "messages": [{"role": "user", "content": "test"}],
                "stream": True,
            }
        ).encode()
        blocked_tools = [
            {"tool_use_id": "t1", "name": "Bash", "input": {"command": "rm"}, "reason": "r1"},
            {"tool_use_id": "t2", "name": "Write", "input": {"file_path": "/etc/passwd"}, "reason": "r2"},
        ]
        result = _build_retry_request_body(original, blocked_tools)
        data = json.loads(result)

        # Two tool_use blocks in assistant message
        assert len(data["messages"][1]["content"]) == 2
        assert data["messages"][1]["content"][0]["id"] == "t1"
        assert data["messages"][1]["content"][1]["id"] == "t2"
        # Two tool_result blocks in user message
        assert len(data["messages"][2]["content"]) == 2
        assert data["messages"][2]["content"][0]["tool_use_id"] == "t1"
        assert data["messages"][2]["content"][1]["tool_use_id"] == "t2"

    def test_preserves_model_and_stream(self):
        original = json.dumps(
            {
                "model": "claude-3-opus",
                "stream": True,
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"name": "Bash"}],
            }
        ).encode()
        blocked_tools = [
            {"tool_use_id": "t1", "name": "Bash", "input": {}, "reason": "blocked"},
        ]
        result = _build_retry_request_body(original, blocked_tools)
        data = json.loads(result)

        assert data["model"] == "claude-3-opus"
        assert data["stream"] is True
        assert data["tools"] == [{"name": "Bash"}]


# ============================================================
# _rewrite_content_block_index
# ============================================================


class TestRewriteContentBlockIndex:
    def test_rewrites_content_block_start(self):
        data = {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}
        line = f"data: {json.dumps(data)}"
        result = _rewrite_content_block_index(line, 2)
        result_data = json.loads(result[6:])
        assert result_data["index"] == 2

    def test_rewrites_content_block_delta(self):
        data = {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "hi"}}
        line = f"data: {json.dumps(data)}"
        result = _rewrite_content_block_index(line, 3)
        result_data = json.loads(result[6:])
        assert result_data["index"] == 4

    def test_rewrites_content_block_stop(self):
        data = {"type": "content_block_stop", "index": 0}
        line = f"data: {json.dumps(data)}"
        result = _rewrite_content_block_index(line, 5)
        result_data = json.loads(result[6:])
        assert result_data["index"] == 5

    def test_no_rewrite_for_message_start(self):
        data = {"type": "message_start", "message": {"id": "m1"}}
        line = f"data: {json.dumps(data)}"
        result = _rewrite_content_block_index(line, 3)
        assert result == line  # unchanged

    def test_no_rewrite_for_zero_offset(self):
        data = {"type": "content_block_start", "index": 0}
        line = f"data: {json.dumps(data)}"
        result = _rewrite_content_block_index(line, 0)
        assert result == line  # unchanged

    def test_no_rewrite_for_non_data_line(self):
        line = "event: content_block_start"
        result = _rewrite_content_block_index(line, 5)
        assert result == line  # unchanged

    def test_no_rewrite_for_malformed_json(self):
        line = "data: not valid json"
        result = _rewrite_content_block_index(line, 5)
        assert result == line  # unchanged

    def test_no_rewrite_without_index_key(self):
        data = {"type": "content_block_start", "content_block": {"type": "text"}}
        line = f"data: {json.dumps(data)}"
        result = _rewrite_content_block_index(line, 3)
        assert result == line  # no index key, unchanged


# ============================================================
# BLOCK retry behavior (streaming)
# ============================================================


def _make_allow_sse_lines():
    """Build SSE lines for a simple allowed text response."""
    return [
        "event: message_start",
        'data: {"type":"message_start","message":{"id":"m2","model":"claude-3","role":"assistant"}}',
        "",
        "event: content_block_start",
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
        "",
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"OK I will try something else"}}',
        "",
        "event: content_block_stop",
        'data: {"type":"content_block_stop","index":0}',
        "",
        "event: message_delta",
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":8}}',
        "",
        "event: message_stop",
        'data: {"type":"message_stop"}',
        "",
    ]


class TestBlockRetryStreaming:
    """Tests for BLOCK action triggering retry behavior in streaming proxy."""

    @pytest.mark.anyio
    async def test_block_triggers_retry(self):
        """BLOCK should suppress tool_use and make a new API call with error feedback."""
        # First upstream: model tries a blocked tool
        first_sse = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"m1","model":"claude-3","role":"assistant"}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"Bash"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"command\\": \\"rm -rf /\\"}"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":10}}',
            "",
        ]
        # Second upstream (retry): model self-corrects
        second_sse = _make_allow_sse_lines()

        first_upstream = _make_mock_upstream(first_sse)
        second_upstream = _make_mock_upstream(second_sse)

        mock_client = AsyncMock()
        mock_client.build_request.return_value = MagicMock()
        mock_client.send = AsyncMock(side_effect=[first_upstream, second_upstream])

        event_bus = EventBus()
        events_received = []
        event_bus.subscribe(AsyncMock(side_effect=lambda e: events_received.append(e)))
        chain = GateChain(gates=[BlockGate(name="blocker")])

        with unittest.mock.patch.object(settings, "gate_max_retries", 3):
            result = await _proxy_streaming(
                mock_client,
                event_bus,
                "POST",
                "/v1/messages",
                {"host": "api.anthropic.com"},
                _streaming_request_body(),
                gate_chain=chain,
            )
            output = await _collect_streaming_output(result)

        output_text = "".join(output)

        # The blocked tool should NOT appear in output
        assert "rm -rf" not in output_text
        # The retry response text should appear
        assert "try something else" in output_text
        # Tool should NOT be in _blocked_tool_ids (retry handled it)
        assert "toolu_1" not in _blocked_tool_ids
        # A GateRetryEvent should have been published
        retry_events = [e for e in events_received if isinstance(e, GateRetryEvent)]
        assert len(retry_events) == 1
        assert retry_events[0].tool_name == "Bash"
        assert retry_events[0].retry_number == 1
        # The client should have been called twice (original + retry)
        assert mock_client.send.call_count == 2

    @pytest.mark.anyio
    async def test_block_retry_budget_exhausted(self):
        """When max retries are exhausted, BLOCK falls back to HALT_SESSION behavior."""

        # All upstreams return a blocked tool
        def make_blocked_sse():
            return [
                "event: message_start",
                'data: {"type":"message_start","message":{"id":"m1","model":"claude-3","role":"assistant"}}',
                "",
                "event: content_block_start",
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"Bash"}}',
                "",
                "event: content_block_delta",
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"command\\": \\"rm -rf /\\"}"}}',
                "",
                "event: content_block_stop",
                'data: {"type":"content_block_stop","index":0}',
                "",
            ]

        # Create upstreams: 1 original + 1 retry (max_retries=1)
        upstreams = [_make_mock_upstream(make_blocked_sse()) for _ in range(2)]

        mock_client = AsyncMock()
        mock_client.build_request.return_value = MagicMock()
        mock_client.send = AsyncMock(side_effect=upstreams)

        event_bus = EventBus()
        chain = GateChain(gates=[BlockGate(name="blocker")])

        with unittest.mock.patch.object(settings, "gate_max_retries", 1):
            result = await _proxy_streaming(
                mock_client,
                event_bus,
                "POST",
                "/v1/messages",
                {"host": "api.anthropic.com"},
                _streaming_request_body(),
                gate_chain=chain,
            )
            output = await _collect_streaming_output(result)

        output_text = "".join(output)

        # After retry budget exhausted, should fall back to HALT_SESSION behavior
        assert "message_stop" in output_text
        assert "end_turn" in output_text
        # Tool should be in _blocked_tool_ids (halted after exhaustion)
        assert "toolu_1" in _blocked_tool_ids
        # Client called twice: original + 1 retry
        assert mock_client.send.call_count == 2

    @pytest.mark.anyio
    async def test_block_retry_skips_message_start(self):
        """Retry response's message_start should be skipped (client already received one)."""
        first_sse = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"m1","model":"claude-3","role":"assistant"}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"Bash"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"command\\": \\"ls\\"}"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
        ]
        second_sse = _make_allow_sse_lines()

        first_upstream = _make_mock_upstream(first_sse)
        second_upstream = _make_mock_upstream(second_sse)

        mock_client = AsyncMock()
        mock_client.build_request.return_value = MagicMock()
        mock_client.send = AsyncMock(side_effect=[first_upstream, second_upstream])

        event_bus = EventBus()
        chain = GateChain(gates=[BlockGate(name="blocker")])

        with unittest.mock.patch.object(settings, "gate_max_retries", 3):
            result = await _proxy_streaming(
                mock_client,
                event_bus,
                "POST",
                "/v1/messages",
                {"host": "api.anthropic.com"},
                _streaming_request_body(),
                gate_chain=chain,
            )
            output = await _collect_streaming_output(result)

        output_text = "".join(output)

        # The first attempt emits message_start (event: + data: lines = 2 occurrences).
        # The retry should skip its message_start lines, so no additional occurrences.
        count = output_text.count('"type":"message_start"')
        assert count == 1  # only from the first attempt's data: line

    @pytest.mark.anyio
    async def test_halt_session_does_not_retry(self):
        """HALT_SESSION should NOT trigger retry — it freezes immediately."""
        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"m1","model":"claude-3","role":"assistant"}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"Bash"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"command\\": \\"bad\\"}"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
        ]

        upstream = _make_mock_upstream(sse_lines)
        mock_client = AsyncMock()
        mock_client.build_request.return_value = MagicMock()
        mock_client.send = AsyncMock(return_value=upstream)

        event_bus = EventBus()
        chain = GateChain(gates=[HaltSessionGate(name="halter")])

        with unittest.mock.patch.object(settings, "gate_max_retries", 3):
            result = await _proxy_streaming(
                mock_client,
                event_bus,
                "POST",
                "/v1/messages",
                {"host": "api.anthropic.com"},
                _streaming_request_body(),
                gate_chain=chain,
            )
            output = await _collect_streaming_output(result)

        output_text = "".join(output)

        # Should freeze: synthetic message_stop, tool in _blocked_tool_ids
        assert "message_stop" in output_text
        assert "toolu_1" in _blocked_tool_ids
        # Client should only be called once (no retry)
        assert mock_client.send.call_count == 1

    @pytest.mark.anyio
    async def test_zero_max_retries_falls_back_to_halt(self):
        """With gate_max_retries=0, BLOCK immediately falls back to HALT_SESSION behavior."""
        sse_lines = [
            "event: message_start",
            'data: {"type":"message_start","message":{"id":"m1","model":"claude-3","role":"assistant"}}',
            "",
            "event: content_block_start",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"Bash"}}',
            "",
            "event: content_block_delta",
            'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"command\\": \\"rm\\"}"}}',
            "",
            "event: content_block_stop",
            'data: {"type":"content_block_stop","index":0}',
            "",
        ]

        upstream = _make_mock_upstream(sse_lines)
        mock_client = AsyncMock()
        mock_client.build_request.return_value = MagicMock()
        mock_client.send = AsyncMock(return_value=upstream)

        event_bus = EventBus()
        chain = GateChain(gates=[BlockGate(name="blocker")])

        with unittest.mock.patch.object(settings, "gate_max_retries", 0):
            result = await _proxy_streaming(
                mock_client,
                event_bus,
                "POST",
                "/v1/messages",
                {"host": "api.anthropic.com"},
                _streaming_request_body(),
                gate_chain=chain,
            )
            output = await _collect_streaming_output(result)

        output_text = "".join(output)

        # Should fall back to HALT_SESSION behavior
        assert "message_stop" in output_text
        assert "toolu_1" in _blocked_tool_ids
        # No retry attempt
        assert mock_client.send.call_count == 1
