"""Unit tests for SSE parser edge cases.

Covers the lines missed by test_sse_gating.py:
- data: [DONE] sentinel (line 54)
- Comment lines and non-SSE lines (lines 65-68)
- Empty event (no accumulated data lines) on blank line (line 65)
- Malformed JSON in SSE data (lines 74-75)
- Malformed JSON in tool_use input assembly (lines 118-119)
- message_delta parsing
- content_block_stop without prior content_block_start
- content_block_delta for unknown index
"""

import json

from cross.events import (
    MessageDeltaEvent,
    MessageStartEvent,
    TextEvent,
    ToolUseEvent,
)
from cross.sse import SSEParser

# --- Helpers ---


def _feed_event(parser: SSEParser, event_type: str, data: dict) -> list:
    """Feed a complete SSE event (event line, data line, blank line) and return emitted events."""
    events = []
    events.extend(parser.feed_line(f"event: {event_type}"))
    events.extend(parser.feed_line(f"data: {json.dumps(data)}"))
    events.extend(parser.feed_line(""))
    return events


# --- Tests ---


class TestDoneSentinel:
    """Verify data: [DONE] is consumed without error."""

    def test_done_returns_no_events(self):
        parser = SSEParser()
        result = parser.feed_line("data: [DONE]")
        assert result == []

    def test_done_after_normal_stream(self):
        parser = SSEParser()
        _feed_event(
            parser,
            "message_start",
            {"type": "message_start", "message": {"id": "msg_1", "model": "claude-3"}},
        )
        result = parser.feed_line("data: [DONE]")
        assert result == []


class TestCommentAndNonSSELines:
    """Verify comment lines and other non-SSE lines are ignored."""

    def test_comment_line_ignored(self):
        parser = SSEParser()
        result = parser.feed_line(": this is a comment")
        assert result == []

    def test_colon_only_comment(self):
        parser = SSEParser()
        result = parser.feed_line(":")
        assert result == []

    def test_arbitrary_text_ignored(self):
        parser = SSEParser()
        result = parser.feed_line("some random text")
        assert result == []

    def test_ping_line_ignored(self):
        parser = SSEParser()
        result = parser.feed_line(": ping")
        assert result == []

    def test_consecutive_blank_lines_no_crash(self):
        """Multiple blank lines in a row should not produce events."""
        parser = SSEParser()
        assert parser.feed_line("") == []
        assert parser.feed_line("") == []
        assert parser.feed_line("") == []


class TestMalformedJSON:
    """Verify malformed JSON in SSE data is handled gracefully."""

    def test_malformed_json_returns_empty(self):
        parser = SSEParser()
        parser.feed_line("event: message_start")
        parser.feed_line("data: {this is not valid json")
        result = parser.feed_line("")
        assert result == []

    def test_truncated_json(self):
        parser = SSEParser()
        parser.feed_line("event: content_block_start")
        parser.feed_line('data: {"type": "content_block_start", "index":')
        result = parser.feed_line("")
        assert result == []

    def test_empty_data_value(self):
        """data: with empty string after prefix."""
        parser = SSEParser()
        parser.feed_line("event: message_start")
        parser.feed_line("data: ")
        # Empty string is not valid JSON, so should be handled
        result = parser.feed_line("")
        assert result == []


class TestMalformedToolInput:
    """Verify malformed JSON in tool_use input assembly produces _raw fallback."""

    def test_invalid_json_in_tool_input(self):
        parser = SSEParser()
        # Start a tool_use content block
        _feed_event(
            parser,
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_1", "name": "Bash"},
            },
        )
        # Feed partial JSON that won't parse
        _feed_event(
            parser,
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"command": '},
            },
        )
        _feed_event(
            parser,
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": "BROKEN"},
            },
        )
        # Stop the block -- should produce ToolUseEvent with _raw key
        events = _feed_event(
            parser,
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        )
        assert len(events) == 1
        assert isinstance(events[0], ToolUseEvent)
        assert events[0].name == "Bash"
        assert events[0].tool_use_id == "toolu_1"
        assert "_raw" in events[0].input
        assert events[0].input["_raw"] == '{"command": BROKEN'

    def test_empty_tool_input_produces_empty_dict(self):
        parser = SSEParser()
        _feed_event(
            parser,
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_2", "name": "Read"},
            },
        )
        # Stop immediately without any input_json_delta
        events = _feed_event(
            parser,
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        )
        assert len(events) == 1
        assert isinstance(events[0], ToolUseEvent)
        assert events[0].input == {}


class TestContentBlockStopWithoutStart:
    """Verify content_block_stop for an unknown index is a no-op."""

    def test_stop_unknown_index(self):
        parser = SSEParser()
        events = _feed_event(
            parser,
            "content_block_stop",
            {"type": "content_block_stop", "index": 99},
        )
        assert events == []


class TestContentBlockDeltaWithoutStart:
    """Verify content_block_delta for an unknown index is a no-op."""

    def test_delta_unknown_index(self):
        parser = SSEParser()
        events = _feed_event(
            parser,
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 42,
                "delta": {"type": "text_delta", "text": "orphan"},
            },
        )
        assert events == []


class TestMessageDeltaParsing:
    """Verify message_delta events are parsed correctly."""

    def test_message_delta_with_stop_reason(self):
        parser = SSEParser()
        events = _feed_event(
            parser,
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 150},
            },
        )
        assert len(events) == 1
        assert isinstance(events[0], MessageDeltaEvent)
        assert events[0].stop_reason == "end_turn"
        assert events[0].output_tokens == 150

    def test_message_delta_tool_use_stop(self):
        parser = SSEParser()
        events = _feed_event(
            parser,
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use"},
                "usage": {"output_tokens": 42},
            },
        )
        assert len(events) == 1
        assert isinstance(events[0], MessageDeltaEvent)
        assert events[0].stop_reason == "tool_use"

    def test_message_delta_missing_usage(self):
        parser = SSEParser()
        events = _feed_event(
            parser,
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
            },
        )
        assert len(events) == 1
        assert isinstance(events[0], MessageDeltaEvent)
        assert events[0].output_tokens == 0


class TestMessageStartParsing:
    """Verify message_start events are parsed correctly."""

    def test_message_start(self):
        parser = SSEParser()
        events = _feed_event(
            parser,
            "message_start",
            {
                "type": "message_start",
                "message": {"id": "msg_abc", "model": "claude-3-opus"},
            },
        )
        assert len(events) == 1
        assert isinstance(events[0], MessageStartEvent)
        assert events[0].message_id == "msg_abc"
        assert events[0].model == "claude-3-opus"

    def test_message_start_missing_fields(self):
        parser = SSEParser()
        events = _feed_event(
            parser,
            "message_start",
            {"type": "message_start", "message": {}},
        )
        assert len(events) == 1
        assert isinstance(events[0], MessageStartEvent)
        assert events[0].message_id == ""
        assert events[0].model == ""


class TestTextBlockAssembly:
    """Verify text blocks are assembled correctly."""

    def test_text_block_with_multiple_deltas(self):
        parser = SSEParser()
        _feed_event(
            parser,
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
        _feed_event(
            parser,
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hello "},
            },
        )
        _feed_event(
            parser,
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "world!"},
            },
        )
        events = _feed_event(
            parser,
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        )
        assert len(events) == 1
        assert isinstance(events[0], TextEvent)
        assert events[0].text == "Hello world!"


class TestToolUseFullAssembly:
    """Verify full tool_use block assembly with multi-part JSON input."""

    def test_tool_use_multi_chunk_json(self):
        parser = SSEParser()
        _feed_event(
            parser,
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "tool_use", "id": "toolu_abc", "name": "Write"},
            },
        )
        # Send JSON in small chunks
        chunks = ['{"file', '_path":', ' "/tmp/', 'out.txt",', ' "content":', ' "hello"}']
        for chunk in chunks:
            _feed_event(
                parser,
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": chunk},
                },
            )
        events = _feed_event(
            parser,
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        )
        assert len(events) == 1
        assert isinstance(events[0], ToolUseEvent)
        assert events[0].name == "Write"
        assert events[0].tool_use_id == "toolu_abc"
        assert events[0].input == {"file_path": "/tmp/out.txt", "content": "hello"}


class TestMultipleContentBlocks:
    """Verify multiple content blocks at different indices are tracked independently."""

    def test_interleaved_blocks(self):
        parser = SSEParser()
        # Start two blocks
        _feed_event(
            parser,
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
        )
        _feed_event(
            parser,
            "content_block_start",
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "id": "toolu_x", "name": "Bash"},
            },
        )
        # Deltas for both
        _feed_event(
            parser,
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Running command..."},
            },
        )
        _feed_event(
            parser,
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "input_json_delta", "partial_json": '{"command": "ls"}'},
            },
        )
        # Stop both
        text_events = _feed_event(
            parser,
            "content_block_stop",
            {"type": "content_block_stop", "index": 0},
        )
        tool_events = _feed_event(
            parser,
            "content_block_stop",
            {"type": "content_block_stop", "index": 1},
        )
        assert len(text_events) == 1
        assert isinstance(text_events[0], TextEvent)
        assert text_events[0].text == "Running command..."

        assert len(tool_events) == 1
        assert isinstance(tool_events[0], ToolUseEvent)
        assert tool_events[0].input == {"command": "ls"}


class TestUnknownEventTypes:
    """Verify unknown SSE event types are silently ignored."""

    def test_unknown_type_in_data(self):
        parser = SSEParser()
        events = _feed_event(
            parser,
            "unknown_event",
            {"type": "some_future_event", "data": "whatever"},
        )
        assert events == []

    def test_error_type_ignored(self):
        parser = SSEParser()
        events = _feed_event(
            parser,
            "error",
            {"type": "error", "error": {"message": "overloaded"}},
        )
        assert events == []


class TestMultiLineData:
    """Verify multi-line data fields are joined before parsing."""

    def test_data_split_across_lines(self):
        parser = SSEParser()
        parser.feed_line("event: message_start")
        # Simulate data split across multiple data: lines
        parser.feed_line('data: {"type": "message_start",')
        parser.feed_line('data: "message": {"id": "msg_split", "model": "claude-3"}}')
        events = parser.feed_line("")
        assert len(events) == 1
        assert isinstance(events[0], MessageStartEvent)
        assert events[0].message_id == "msg_split"


class TestCarriageReturnStripping:
    """Verify \\r\\n line endings are handled."""

    def test_crlf_event_line(self):
        parser = SSEParser()
        result = parser.feed_line("event: message_start\r\n")
        assert result == []
        # The event type should be set correctly despite CRLF
        parser.feed_line('data: {"type": "message_start", "message": {"id": "m1", "model": "c3"}}')
        events = parser.feed_line("")
        assert len(events) == 1
        assert isinstance(events[0], MessageStartEvent)
