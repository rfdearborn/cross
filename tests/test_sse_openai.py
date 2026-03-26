"""Unit tests for the OpenAI SSE parser.

Tests parsing of OpenAI Chat Completions streaming format into CrossEvents.
"""

import json

from cross.events import (
    MessageDeltaEvent,
    MessageStartEvent,
    TextEvent,
    ToolUseEvent,
)
from cross.sse import OpenAISSEParser


def _feed_chunk(parser: OpenAISSEParser, chunk: dict) -> list:
    """Feed a single OpenAI SSE chunk (data line + blank line)."""
    events = []
    events.extend(parser.feed_line(f"data: {json.dumps(chunk)}"))
    events.extend(parser.feed_line(""))
    return events


def _make_chunk(
    chunk_id="chatcmpl-test",
    model="gpt-4o",
    delta=None,
    finish_reason=None,
    usage=None,
):
    """Build an OpenAI streaming chunk."""
    choice = {"index": 0, "delta": delta or {}}
    if finish_reason:
        choice["finish_reason"] = finish_reason
    chunk = {"id": chunk_id, "object": "chat.completion.chunk", "model": model, "choices": [choice]}
    if usage:
        chunk["usage"] = usage
    return chunk


class TestOpenAIMessageStart:
    def test_emits_message_start_on_first_chunk(self):
        parser = OpenAISSEParser()
        events = _feed_chunk(parser, _make_chunk(delta={"role": "assistant", "content": ""}))
        assert len(events) == 1
        assert isinstance(events[0], MessageStartEvent)
        assert events[0].message_id == "chatcmpl-test"
        assert events[0].model == "gpt-4o"

    def test_message_start_only_once(self):
        parser = OpenAISSEParser()
        _feed_chunk(parser, _make_chunk(delta={"role": "assistant", "content": ""}))
        events = _feed_chunk(parser, _make_chunk(delta={"content": "Hello"}))
        # No second MessageStartEvent
        assert not any(isinstance(e, MessageStartEvent) for e in events)


class TestOpenAITextStreaming:
    def test_text_emitted_on_stop(self):
        parser = OpenAISSEParser()
        _feed_chunk(parser, _make_chunk(delta={"role": "assistant", "content": ""}))
        _feed_chunk(parser, _make_chunk(delta={"content": "Hello"}))
        _feed_chunk(parser, _make_chunk(delta={"content": " world"}))
        events = _feed_chunk(parser, _make_chunk(delta={}, finish_reason="stop"))

        text_events = [e for e in events if isinstance(e, TextEvent)]
        assert len(text_events) == 1
        assert text_events[0].text == "Hello world"

    def test_text_emitted_on_done(self):
        parser = OpenAISSEParser()
        _feed_chunk(parser, _make_chunk(delta={"role": "assistant", "content": ""}))
        _feed_chunk(parser, _make_chunk(delta={"content": "Hi"}))
        events = parser.feed_line("data: [DONE]")
        assert len(events) == 1
        assert isinstance(events[0], TextEvent)
        assert events[0].text == "Hi"


class TestOpenAIToolCalls:
    def test_single_tool_call(self):
        parser = OpenAISSEParser()
        _feed_chunk(parser, _make_chunk(delta={"role": "assistant", "content": None}))

        # First tool_call delta — has id, type, function.name
        _feed_chunk(
            parser,
            _make_chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_abc",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": ""},
                        }
                    ]
                }
            ),
        )

        # Argument fragments
        _feed_chunk(
            parser,
            _make_chunk(delta={"tool_calls": [{"index": 0, "function": {"arguments": '{"lo'}}]}),
        )
        _feed_chunk(
            parser,
            _make_chunk(delta={"tool_calls": [{"index": 0, "function": {"arguments": 'cation":'}}]}),
        )
        _feed_chunk(
            parser,
            _make_chunk(delta={"tool_calls": [{"index": 0, "function": {"arguments": '"NYC"}'}}]}),
        )

        # Finish
        events = _feed_chunk(parser, _make_chunk(delta={}, finish_reason="tool_calls"))

        tool_events = [e for e in events if isinstance(e, ToolUseEvent)]
        assert len(tool_events) == 1
        assert tool_events[0].name == "get_weather"
        assert tool_events[0].tool_use_id == "call_abc"
        assert tool_events[0].input == {"location": "NYC"}

        # Should also emit MessageDeltaEvent with normalized stop_reason
        delta_events = [e for e in events if isinstance(e, MessageDeltaEvent)]
        assert len(delta_events) == 1
        assert delta_events[0].stop_reason == "tool_use"

    def test_multiple_parallel_tool_calls(self):
        parser = OpenAISSEParser()
        _feed_chunk(parser, _make_chunk(delta={"role": "assistant"}))

        # Two tool calls started in same delta
        _feed_chunk(
            parser,
            _make_chunk(
                delta={
                    "tool_calls": [
                        {"index": 0, "id": "call_1", "type": "function", "function": {"name": "read", "arguments": ""}},
                        {
                            "index": 1,
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "write", "arguments": ""},
                        },
                    ]
                }
            ),
        )

        # Arguments for each
        _feed_chunk(
            parser,
            _make_chunk(
                delta={
                    "tool_calls": [
                        {"index": 0, "function": {"arguments": '{"path":"a.txt"}'}},
                        {"index": 1, "function": {"arguments": '{"path":"b.txt"}'}},
                    ]
                }
            ),
        )

        events = _feed_chunk(parser, _make_chunk(delta={}, finish_reason="tool_calls"))

        tool_events = [e for e in events if isinstance(e, ToolUseEvent)]
        assert len(tool_events) == 2
        assert tool_events[0].name == "read"
        assert tool_events[0].tool_use_id == "call_1"
        assert tool_events[1].name == "write"
        assert tool_events[1].tool_use_id == "call_2"

    def test_malformed_arguments(self):
        """Malformed JSON in arguments should still produce a ToolUseEvent with _raw."""
        parser = OpenAISSEParser()
        _feed_chunk(parser, _make_chunk(delta={"role": "assistant"}))
        _feed_chunk(
            parser,
            _make_chunk(
                delta={
                    "tool_calls": [
                        {"index": 0, "id": "call_bad", "type": "function", "function": {"name": "foo", "arguments": ""}}
                    ]
                }
            ),
        )
        _feed_chunk(
            parser,
            _make_chunk(delta={"tool_calls": [{"index": 0, "function": {"arguments": "{broken"}}]}),
        )
        events = _feed_chunk(parser, _make_chunk(delta={}, finish_reason="tool_calls"))

        tool_events = [e for e in events if isinstance(e, ToolUseEvent)]
        assert len(tool_events) == 1
        assert tool_events[0].input == {"_raw": "{broken"}


class TestOpenAIStopReasonNormalization:
    def test_stop_normalizes_to_end_turn(self):
        parser = OpenAISSEParser()
        _feed_chunk(parser, _make_chunk(delta={"role": "assistant", "content": "done"}))
        events = _feed_chunk(parser, _make_chunk(delta={}, finish_reason="stop"))

        delta_events = [e for e in events if isinstance(e, MessageDeltaEvent)]
        assert len(delta_events) == 1
        assert delta_events[0].stop_reason == "end_turn"

    def test_tool_calls_normalizes_to_tool_use(self):
        parser = OpenAISSEParser()
        _feed_chunk(parser, _make_chunk(delta={"role": "assistant"}))
        _feed_chunk(
            parser,
            _make_chunk(
                delta={
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_x",
                            "type": "function",
                            "function": {"name": "bash", "arguments": "{}"},
                        },
                    ]
                }
            ),
        )
        events = _feed_chunk(parser, _make_chunk(delta={}, finish_reason="tool_calls"))

        delta_events = [e for e in events if isinstance(e, MessageDeltaEvent)]
        assert delta_events[0].stop_reason == "tool_use"


class TestOpenAIEdgeCases:
    def test_empty_choices(self):
        parser = OpenAISSEParser()
        events = _feed_chunk(parser, {"id": "x", "model": "m", "choices": []})
        # Just MessageStartEvent, nothing else
        assert len(events) == 1
        assert isinstance(events[0], MessageStartEvent)

    def test_done_line(self):
        parser = OpenAISSEParser()
        events = parser.feed_line("data: [DONE]")
        assert events == []

    def test_malformed_json(self):
        parser = OpenAISSEParser()
        events = parser.feed_line("data: {bad json")
        assert events == []

    def test_comment_lines_ignored(self):
        parser = OpenAISSEParser()
        events = parser.feed_line(": keep-alive")
        assert events == []

    def test_event_lines_ignored(self):
        parser = OpenAISSEParser()
        events = parser.feed_line("event: ping")
        assert events == []
