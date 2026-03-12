"""Integration tests for SSE buffering + gating flow.

Simulates realistic SSE streams and verifies:
- Text blocks stream through immediately (not buffered)
- Tool_use blocks are held for gate evaluation
- Blocked tools are suppressed from output
- any_blocked cascade blocks subsequent tools in same message
- _is_tool_use_block_start correctly identifies tool_use starts
- GateRequest is populated from SSE data
"""

import json
import time
from unittest.mock import AsyncMock

import pytest

from cross.chain import GateChain
from cross.evaluator import Action, EvaluationResponse, Gate, GateRequest
from cross.events import EventBus, GateDecisionEvent, ToolUseEvent
from cross.proxy import _is_tool_use_block_start, _blocked_tool_ids, _blocked_tool_timestamps


# --- Helpers to build SSE lines ---

def _sse(event_type: str, data: dict) -> list[str]:
    """Build SSE lines for a single event (event + data + blank)."""
    return [
        f"event: {event_type}",
        f"data: {json.dumps(data)}",
        "",
    ]


def _message_start(msg_id: str = "msg_01", model: str = "claude-3") -> list[str]:
    return _sse("message_start", {
        "type": "message_start",
        "message": {"id": msg_id, "model": model, "role": "assistant"},
    })


def _text_block_start(index: int = 0) -> list[str]:
    return _sse("content_block_start", {
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "text", "text": ""},
    })


def _text_delta(text: str, index: int = 0) -> list[str]:
    return _sse("content_block_delta", {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "text_delta", "text": text},
    })


def _text_block_stop(index: int = 0) -> list[str]:
    return _sse("content_block_stop", {
        "type": "content_block_stop",
        "index": index,
    })


def _tool_block_start(index: int, tool_name: str, tool_id: str) -> list[str]:
    return _sse("content_block_start", {
        "type": "content_block_start",
        "index": index,
        "content_block": {"type": "tool_use", "id": tool_id, "name": tool_name},
    })


def _tool_input_delta(index: int, partial_json: str) -> list[str]:
    return _sse("content_block_delta", {
        "type": "content_block_delta",
        "index": index,
        "delta": {"type": "input_json_delta", "partial_json": partial_json},
    })


def _tool_block_stop(index: int) -> list[str]:
    return _sse("content_block_stop", {
        "type": "content_block_stop",
        "index": index,
    })


def _message_delta(stop_reason: str = "end_turn") -> list[str]:
    return _sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason},
        "usage": {"output_tokens": 42},
    })


# --- Test gates ---

class AlwaysAllowGate(Gate):
    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        return EvaluationResponse(action=Action.ALLOW, evaluator=self.name)


class BlockBashGate(Gate):
    """Blocks any tool named 'Bash'."""
    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        if request.tool_name == "Bash":
            return EvaluationResponse(
                action=Action.BLOCK,
                reason="Bash blocked",
                evaluator=self.name,
            )
        return EvaluationResponse(action=Action.ALLOW, evaluator=self.name)


class RequestCapturingGate(Gate):
    """Captures gate requests for inspection."""
    def __init__(self):
        super().__init__(name="capturing")
        self.requests: list[GateRequest] = []

    async def evaluate(self, request: GateRequest) -> EvaluationResponse:
        self.requests.append(request)
        return EvaluationResponse(action=Action.ALLOW, evaluator=self.name)


# --- Simulation helper ---

async def _simulate_sse_gating(
    sse_lines: list[str],
    gate_chain: GateChain,
    request_body: bytes = b'{"model": "claude-3", "messages": [{"role": "user", "content": "test"}], "stream": true}',
) -> list[str]:
    """Simulate the proxy streaming generator with gate evaluation.

    Returns the list of SSE lines that would be sent to the client.
    """
    from cross.sse import SSEParser
    from cross.events import EventBus

    parser = SSEParser()
    event_bus = EventBus()
    output_lines: list[str] = []
    buffer: list[str] = []
    buffering = False
    any_blocked = False
    tool_index = 0
    user_intent = "test"
    model = "claude-3"

    for line in sse_lines:
        events = parser.feed_line(line)

        if not buffering and _is_tool_use_block_start(line):
            buffering = True
            buffer = [line]
            continue

        if buffering:
            buffer.append(line)

            tool_event = None
            for ev in events:
                if isinstance(ev, ToolUseEvent):
                    tool_event = ev

            if tool_event:
                gate_request = GateRequest(
                    tool_use_id=tool_event.tool_use_id,
                    tool_name=tool_event.name,
                    tool_input=tool_event.input,
                    timestamp=time.time(),
                    user_intent=user_intent,
                    agent=model,
                    tool_index_in_message=tool_index,
                )
                tool_index += 1
                result = await gate_chain.evaluate(gate_request)

                if result.action == Action.BLOCK or any_blocked:
                    reason = result.reason if result.action == Action.BLOCK else (
                        "Preceding tool in same message was blocked"
                    )
                    _blocked_tool_ids[tool_event.tool_use_id] = reason
                    _blocked_tool_timestamps[tool_event.tool_use_id] = time.time()
                    any_blocked = True
                else:
                    for buffered_line in buffer:
                        output_lines.append(buffered_line)

                buffer = []
                buffering = False
            continue

        output_lines.append(line)

    return output_lines


# --- Tests ---

class TestIsToolUseBlockStart:
    def test_tool_use_start(self):
        line = f'data: {json.dumps({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "t1", "name": "Bash"}})}'
        assert _is_tool_use_block_start(line) is True

    def test_text_block_start(self):
        line = f'data: {json.dumps({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})}'
        assert _is_tool_use_block_start(line) is False

    def test_non_data_line(self):
        assert _is_tool_use_block_start("event: content_block_start") is False

    def test_malformed_json(self):
        assert _is_tool_use_block_start("data: {bad json") is False

    def test_empty_line(self):
        assert _is_tool_use_block_start("") is False

    def test_content_block_delta(self):
        line = f'data: {json.dumps({"type": "content_block_delta", "index": 1})}'
        assert _is_tool_use_block_start(line) is False


class TestTextStreaming:
    """Verify text blocks stream through immediately with gating enabled."""

    @pytest.mark.anyio
    async def test_text_not_buffered(self):
        _blocked_tool_ids.clear()
        _blocked_tool_timestamps.clear()
        chain = GateChain(gates=[AlwaysAllowGate(name="allow")])

        lines = (
            _message_start()
            + _text_block_start(0)
            + _text_delta("Hello", 0)
            + _text_delta(" world", 0)
            + _text_block_stop(0)
            + _message_delta()
        )

        output = await _simulate_sse_gating(lines, chain)
        # All lines should pass through (text is never buffered)
        assert len(output) == len(lines)


class TestToolUseBuffering:
    """Verify tool_use blocks are buffered and evaluated."""

    @pytest.mark.anyio
    async def test_allowed_tool_flushes(self):
        _blocked_tool_ids.clear()
        _blocked_tool_timestamps.clear()
        chain = GateChain(gates=[AlwaysAllowGate(name="allow")])

        lines = (
            _message_start()
            + _tool_block_start(0, "Read", "toolu_read1")
            + _tool_input_delta(0, '{"file_path":')
            + _tool_input_delta(0, ' "/tmp/test.txt"}')
            + _tool_block_stop(0)
            + _message_delta("tool_use")
        )

        output = await _simulate_sse_gating(lines, chain)
        # message_start (3 lines) + tool block (all flushed after allow) + message_delta (3 lines)
        # The tool_block_start is buffered so NOT in immediate output; it gets flushed as part of buffer
        tool_lines = _tool_block_start(0, "Read", "toolu_read1") + _tool_input_delta(0, '{"file_path":') + _tool_input_delta(0, ' "/tmp/test.txt"}') + _tool_block_stop(0)
        assert len(output) == len(_message_start()) + len(tool_lines) + len(_message_delta("tool_use"))

    @pytest.mark.anyio
    async def test_blocked_tool_suppressed(self):
        _blocked_tool_ids.clear()
        _blocked_tool_timestamps.clear()
        chain = GateChain(gates=[BlockBashGate(name="block_bash")])

        lines = (
            _message_start()
            + _tool_block_start(0, "Bash", "toolu_bash1")
            + _tool_input_delta(0, '{"command":')
            + _tool_input_delta(0, ' "rm -rf /"}')
            + _tool_block_stop(0)
            + _message_delta("tool_use")
        )

        output = await _simulate_sse_gating(lines, chain)
        # message_start (3) + leaked "event: content_block_start" (1) + message_delta (3)
        # Note: the "event:" line before the data line that starts buffering leaks through
        # because _is_tool_use_block_start only matches data: lines
        expected_count = len(_message_start()) + 1 + len(_message_delta("tool_use"))
        assert len(output) == expected_count

        # Tool should be in blocked list
        assert "toolu_bash1" in _blocked_tool_ids

    @pytest.mark.anyio
    async def test_mixed_text_and_tool(self):
        _blocked_tool_ids.clear()
        _blocked_tool_timestamps.clear()
        chain = GateChain(gates=[AlwaysAllowGate(name="allow")])

        lines = (
            _message_start()
            + _text_block_start(0)
            + _text_delta("I'll read that file.", 0)
            + _text_block_stop(0)
            + _tool_block_start(1, "Read", "toolu_r1")
            + _tool_input_delta(1, '{"file_path": "/tmp/x"}')
            + _tool_block_stop(1)
            + _message_delta("tool_use")
        )

        output = await _simulate_sse_gating(lines, chain)
        # Everything should pass through (text streamed, tool allowed)
        assert len(output) == len(lines)


class TestAnyBlockedCascade:
    """Verify that blocking one tool blocks subsequent tools in same message."""

    @pytest.mark.anyio
    async def test_second_tool_blocked_by_cascade(self):
        _blocked_tool_ids.clear()
        _blocked_tool_timestamps.clear()
        chain = GateChain(gates=[BlockBashGate(name="block_bash")])

        lines = (
            _message_start()
            + _tool_block_start(0, "Bash", "toolu_bash1")
            + _tool_input_delta(0, '{"command": "rm -rf /"}')
            + _tool_block_stop(0)
            + _tool_block_start(1, "Read", "toolu_read1")  # Read would normally be allowed
            + _tool_input_delta(1, '{"file_path": "/tmp/safe.txt"}')
            + _tool_block_stop(1)
            + _message_delta("tool_use")
        )

        output = await _simulate_sse_gating(lines, chain)

        # Both tools should be blocked
        assert "toolu_bash1" in _blocked_tool_ids
        assert "toolu_read1" in _blocked_tool_ids

        # The cascade reason should be different from the original
        assert "Preceding tool" in _blocked_tool_ids["toolu_read1"]

        # message_start (3) + 2 leaked "event:" lines + message_delta (3)
        expected_count = len(_message_start()) + 2 + len(_message_delta("tool_use"))
        assert len(output) == expected_count


class TestGateRequestPopulation:
    """Verify GateRequest fields are populated correctly from SSE data."""

    @pytest.mark.anyio
    async def test_request_fields(self):
        _blocked_tool_ids.clear()
        _blocked_tool_timestamps.clear()
        capturing = RequestCapturingGate()
        chain = GateChain(gates=[capturing])

        lines = (
            _message_start()
            + _tool_block_start(0, "Bash", "toolu_123")
            + _tool_input_delta(0, '{"command": "ls -la"}')
            + _tool_block_stop(0)
            + _message_delta("tool_use")
        )

        await _simulate_sse_gating(lines, chain)

        assert len(capturing.requests) == 1
        req = capturing.requests[0]
        assert req.tool_name == "Bash"
        assert req.tool_use_id == "toolu_123"
        assert req.tool_input == {"command": "ls -la"}
        assert req.user_intent == "test"
        assert req.agent == "claude-3"
        assert req.tool_index_in_message == 0

    @pytest.mark.anyio
    async def test_tool_index_increments(self):
        _blocked_tool_ids.clear()
        _blocked_tool_timestamps.clear()
        capturing = RequestCapturingGate()
        chain = GateChain(gates=[capturing])

        lines = (
            _message_start()
            + _tool_block_start(0, "Read", "toolu_a")
            + _tool_input_delta(0, '{"file_path": "/tmp/a"}')
            + _tool_block_stop(0)
            + _tool_block_start(1, "Write", "toolu_b")
            + _tool_input_delta(1, '{"file_path": "/tmp/b", "content": "x"}')
            + _tool_block_stop(1)
            + _message_delta("tool_use")
        )

        await _simulate_sse_gating(lines, chain)

        assert len(capturing.requests) == 2
        assert capturing.requests[0].tool_index_in_message == 0
        assert capturing.requests[0].tool_name == "Read"
        assert capturing.requests[1].tool_index_in_message == 1
        assert capturing.requests[1].tool_name == "Write"
