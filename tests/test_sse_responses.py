"""Unit tests for the OpenAI Responses API SSE parser.

Tests parsing of OpenAI Responses API streaming format into CrossEvents.
"""

import json

from cross.events import (
    MessageDeltaEvent,
    MessageStartEvent,
    TextEvent,
    ToolUseEvent,
)
from cross.sse import ResponsesSSEParser


def _feed_event(parser: ResponsesSSEParser, event_type: str, data: dict) -> list:
    """Feed a complete Responses API SSE event (event + data + blank line)."""
    events = []
    events.extend(parser.feed_line(f"event: {event_type}"))
    events.extend(parser.feed_line(f"data: {json.dumps(data)}"))
    events.extend(parser.feed_line(""))
    return events


class TestResponsesCreated:
    def test_emits_message_start(self):
        parser = ResponsesSSEParser()
        events = _feed_event(
            parser,
            "response.created",
            {
                "type": "response.created",
                "response": {
                    "id": "resp_abc123",
                    "model": "gpt-5.4",
                    "status": "in_progress",
                },
            },
        )
        assert len(events) == 1
        assert isinstance(events[0], MessageStartEvent)
        assert events[0].message_id == "resp_abc123"
        assert events[0].model == "gpt-5.4"

    def test_missing_response_fields(self):
        parser = ResponsesSSEParser()
        events = _feed_event(
            parser,
            "response.created",
            {"type": "response.created", "response": {}},
        )
        assert len(events) == 1
        assert isinstance(events[0], MessageStartEvent)
        assert events[0].message_id == ""
        assert events[0].model == ""


class TestResponsesOutputItemAdded:
    def test_tracks_function_call(self):
        """output_item.added should track the tool call but not emit events yet."""
        parser = ResponsesSSEParser()
        events = _feed_event(
            parser,
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_001",
                    "call_id": "call_abc",
                    "name": "shell",
                    "arguments": "",
                },
            },
        )
        assert events == []
        # Verify internal tracking
        assert 0 in parser._tool_calls
        assert parser._tool_calls[0].call_id == "call_abc"
        assert parser._tool_calls[0].function_name == "shell"

    def test_ignores_non_function_call_items(self):
        parser = ResponsesSSEParser()
        events = _feed_event(
            parser,
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "message", "id": "msg_001"},
            },
        )
        assert events == []
        assert 0 not in parser._tool_calls


class TestResponsesFunctionCallArgumentsDelta:
    def test_accumulates_argument_fragments(self):
        parser = ResponsesSSEParser()
        # Start tracking a tool call
        _feed_event(
            parser,
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_001",
                    "call_id": "call_abc",
                    "name": "shell",
                    "arguments": "",
                },
            },
        )

        # Feed argument deltas
        _feed_event(
            parser,
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_001",
                "output_index": 0,
                "delta": '{"com',
            },
        )
        _feed_event(
            parser,
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_001",
                "output_index": 0,
                "delta": 'mand":',
            },
        )
        _feed_event(
            parser,
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_001",
                "output_index": 0,
                "delta": ' "ls"}',
            },
        )

        # Verify accumulation
        tc = parser._tool_calls[0]
        assert "".join(tc.argument_parts) == '{"command": "ls"}'

    def test_delta_for_unknown_index_ignored(self):
        parser = ResponsesSSEParser()
        events = _feed_event(
            parser,
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_999",
                "output_index": 99,
                "delta": "ignored",
            },
        )
        assert events == []


class TestResponsesOutputItemDone:
    def test_function_call_emits_tool_use_event(self):
        parser = ResponsesSSEParser()
        # Start and accumulate
        _feed_event(
            parser,
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_001",
                    "call_id": "call_abc",
                    "name": "shell",
                    "arguments": "",
                },
            },
        )
        _feed_event(
            parser,
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_001",
                "output_index": 0,
                "delta": '{"command": "ls"}',
            },
        )

        # Complete the tool call
        events = _feed_event(
            parser,
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_001",
                    "call_id": "call_abc",
                    "name": "shell",
                    "arguments": '{"command": "ls"}',
                },
            },
        )

        assert len(events) == 1
        assert isinstance(events[0], ToolUseEvent)
        assert events[0].name == "shell"
        assert events[0].tool_use_id == "call_abc"
        assert events[0].input == {"command": "ls"}

    def test_non_function_call_item_done_ignored(self):
        parser = ResponsesSSEParser()
        events = _feed_event(
            parser,
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {"type": "message", "id": "msg_001"},
            },
        )
        assert events == []

    def test_malformed_arguments_produces_raw(self):
        """Malformed JSON arguments should produce ToolUseEvent with _raw key."""
        parser = ResponsesSSEParser()
        _feed_event(
            parser,
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_bad",
                    "call_id": "call_bad",
                    "name": "shell",
                    "arguments": "",
                },
            },
        )
        _feed_event(
            parser,
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_bad",
                "output_index": 0,
                "delta": "{broken json",
            },
        )
        events = _feed_event(
            parser,
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_bad",
                    "call_id": "call_bad",
                    "name": "shell",
                    "arguments": "{broken json",
                },
            },
        )
        assert len(events) == 1
        assert isinstance(events[0], ToolUseEvent)
        assert events[0].input == {"_raw": "{broken json"}

    def test_empty_arguments_produce_empty_dict(self):
        parser = ResponsesSSEParser()
        _feed_event(
            parser,
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_empty",
                    "call_id": "call_empty",
                    "name": "no_args_tool",
                    "arguments": "",
                },
            },
        )
        events = _feed_event(
            parser,
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_empty",
                    "call_id": "call_empty",
                    "name": "no_args_tool",
                    "arguments": "",
                },
            },
        )
        assert len(events) == 1
        assert events[0].input == {}


class TestResponsesTextDone:
    def test_emits_text_event(self):
        parser = ResponsesSSEParser()
        events = _feed_event(
            parser,
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "text": "Hello world",
                "output_index": 0,
                "content_index": 0,
            },
        )
        assert len(events) == 1
        assert isinstance(events[0], TextEvent)
        assert events[0].text == "Hello world"

    def test_empty_text_not_emitted(self):
        parser = ResponsesSSEParser()
        events = _feed_event(
            parser,
            "response.output_text.done",
            {"type": "response.output_text.done", "text": ""},
        )
        assert events == []


class TestResponsesCompleted:
    def test_emits_message_delta_end_turn(self):
        parser = ResponsesSSEParser()
        events = _feed_event(
            parser,
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_abc123",
                    "status": "completed",
                    "output": [{"type": "message", "content": [{"type": "output_text", "text": "Hi"}]}],
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                },
            },
        )
        assert len(events) == 1
        assert isinstance(events[0], MessageDeltaEvent)
        assert events[0].stop_reason == "end_turn"
        assert events[0].output_tokens == 50

    def test_emits_tool_use_stop_reason_when_has_function_calls(self):
        parser = ResponsesSSEParser()
        events = _feed_event(
            parser,
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_abc123",
                    "status": "completed",
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "call_abc",
                            "name": "shell",
                            "arguments": '{"command": "ls"}',
                        }
                    ],
                    "usage": {"input_tokens": 100, "output_tokens": 50},
                },
            },
        )
        assert len(events) == 1
        assert isinstance(events[0], MessageDeltaEvent)
        assert events[0].stop_reason == "tool_use"
        assert events[0].output_tokens == 50


class TestResponsesMultipleParallelToolCalls:
    def test_two_parallel_tool_calls(self):
        parser = ResponsesSSEParser()

        # First tool call
        _feed_event(
            parser,
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_001",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": "",
                },
            },
        )
        _feed_event(
            parser,
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_001",
                "output_index": 0,
                "delta": '{"path": "a.txt"}',
            },
        )

        # Second tool call
        _feed_event(
            parser,
            "response.output_item.added",
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {
                    "type": "function_call",
                    "id": "fc_002",
                    "call_id": "call_2",
                    "name": "write_file",
                    "arguments": "",
                },
            },
        )
        _feed_event(
            parser,
            "response.function_call_arguments.delta",
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_002",
                "output_index": 1,
                "delta": '{"path": "b.txt"}',
            },
        )

        # Complete first tool call
        events1 = _feed_event(
            parser,
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_001",
                    "call_id": "call_1",
                    "name": "read_file",
                    "arguments": '{"path": "a.txt"}',
                },
            },
        )
        assert len(events1) == 1
        assert isinstance(events1[0], ToolUseEvent)
        assert events1[0].name == "read_file"
        assert events1[0].tool_use_id == "call_1"

        # Complete second tool call
        events2 = _feed_event(
            parser,
            "response.output_item.done",
            {
                "type": "response.output_item.done",
                "output_index": 1,
                "item": {
                    "type": "function_call",
                    "id": "fc_002",
                    "call_id": "call_2",
                    "name": "write_file",
                    "arguments": '{"path": "b.txt"}',
                },
            },
        )
        assert len(events2) == 1
        assert isinstance(events2[0], ToolUseEvent)
        assert events2[0].name == "write_file"
        assert events2[0].tool_use_id == "call_2"


class TestResponsesEdgeCases:
    def test_done_line_ignored(self):
        parser = ResponsesSSEParser()
        events = parser.feed_line("data: [DONE]")
        assert events == []

    def test_malformed_json_ignored(self):
        parser = ResponsesSSEParser()
        events = []
        events.extend(parser.feed_line("event: response.created"))
        events.extend(parser.feed_line("data: {bad json"))
        events.extend(parser.feed_line(""))
        assert events == []

    def test_comment_lines_ignored(self):
        parser = ResponsesSSEParser()
        events = parser.feed_line(": keep-alive")
        assert events == []

    def test_unknown_event_types_ignored(self):
        parser = ResponsesSSEParser()
        events = _feed_event(
            parser,
            "response.some_unknown_event",
            {"type": "response.some_unknown_event", "data": "whatever"},
        )
        assert events == []

    def test_full_streaming_sequence(self):
        """Test a complete streaming sequence: created -> text -> completed."""
        parser = ResponsesSSEParser()

        # response.created
        events = _feed_event(
            parser,
            "response.created",
            {
                "type": "response.created",
                "response": {"id": "resp_001", "model": "gpt-5.4"},
            },
        )
        assert len(events) == 1
        assert isinstance(events[0], MessageStartEvent)

        # response.output_text.done
        events = _feed_event(
            parser,
            "response.output_text.done",
            {"type": "response.output_text.done", "text": "Hello world"},
        )
        assert len(events) == 1
        assert isinstance(events[0], TextEvent)

        # response.completed
        events = _feed_event(
            parser,
            "response.completed",
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_001",
                    "status": "completed",
                    "output": [],
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                },
            },
        )
        assert len(events) == 1
        assert isinstance(events[0], MessageDeltaEvent)
        assert events[0].stop_reason == "end_turn"
