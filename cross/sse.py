"""SSE stream parsers for Anthropic, OpenAI Chat Completions, and OpenAI Responses APIs.

Parses SSE lines and emits CrossEvent objects as complete events are assembled.
Handles accumulation of input_json_delta / function.arguments chunks into full tool inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from cross.events import (
    CrossEvent,
    MessageDeltaEvent,
    MessageStartEvent,
    TextEvent,
    ToolUseEvent,
)


@dataclass
class _ContentBlock:
    """Tracks state for a content block being streamed."""

    index: int
    block_type: str  # "text" or "tool_use"
    # For tool_use blocks
    tool_name: str = ""
    tool_use_id: str = ""
    input_json_parts: list[str] = field(default_factory=list)
    # For text blocks
    text_parts: list[str] = field(default_factory=list)


class SSEParser:
    """Stateful parser that processes Anthropic SSE lines and yields CrossEvent objects."""

    def __init__(self):
        self._current_event_type: str = ""
        self._data_lines: list[str] = []
        self._blocks: dict[int, _ContentBlock] = {}

    def feed_line(self, line: str) -> list[CrossEvent]:
        """Feed a single SSE line. Returns any events that are now complete."""
        line = line.rstrip("\r\n")

        if line.startswith("event: "):
            self._current_event_type = line[7:]
            return []

        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return []
            self._data_lines.append(data)
            return []

        if line == "":
            # Empty line = end of SSE event, process accumulated data
            if self._data_lines:
                events = self._process_event()
                self._data_lines = []
                self._current_event_type = ""
                return events
            return []

        # Ignore other lines (comments starting with :, etc.)
        return []

    def _process_event(self) -> list[CrossEvent]:
        raw = "\n".join(self._data_lines)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []

        event_type = data.get("type", "")
        events: list[CrossEvent] = []

        if event_type == "message_start":
            msg = data.get("message", {})
            events.append(
                MessageStartEvent(
                    message_id=msg.get("id", ""),
                    model=msg.get("model", ""),
                )
            )

        elif event_type == "content_block_start":
            index = data.get("index", 0)
            block = data.get("content_block", {})
            btype = block.get("type", "")
            cb = _ContentBlock(index=index, block_type=btype)
            if btype == "tool_use":
                cb.tool_name = block.get("name", "")
                cb.tool_use_id = block.get("id", "")
            self._blocks[index] = cb

        elif event_type == "content_block_delta":
            index = data.get("index", 0)
            delta = data.get("delta", {})
            dtype = delta.get("type", "")
            cb = self._blocks.get(index)
            if cb:
                if dtype == "text_delta":
                    cb.text_parts.append(delta.get("text", ""))
                elif dtype == "input_json_delta":
                    cb.input_json_parts.append(delta.get("partial_json", ""))

        elif event_type == "content_block_stop":
            index = data.get("index", 0)
            cb = self._blocks.pop(index, None)
            if cb:
                if cb.block_type == "tool_use":
                    input_json = "".join(cb.input_json_parts)
                    try:
                        tool_input = json.loads(input_json) if input_json else {}
                    except json.JSONDecodeError:
                        tool_input = {"_raw": input_json}
                    events.append(
                        ToolUseEvent(
                            name=cb.tool_name,
                            tool_use_id=cb.tool_use_id,
                            input=tool_input,
                        )
                    )
                elif cb.block_type == "text":
                    events.append(TextEvent(text="".join(cb.text_parts)))

        elif event_type == "message_delta":
            delta = data.get("delta", {})
            usage = data.get("usage", {})
            events.append(
                MessageDeltaEvent(
                    stop_reason=delta.get("stop_reason"),
                    output_tokens=usage.get("output_tokens", 0),
                )
            )

        return events


@dataclass
class _OpenAIToolCall:
    """Tracks state for an OpenAI tool_call being streamed."""

    index: int
    tool_call_id: str = ""
    function_name: str = ""
    argument_parts: list[str] = field(default_factory=list)


class OpenAISSEParser:
    """Stateful parser for OpenAI Chat Completions streaming SSE format.

    OpenAI streams chunks as:
        data: {"id":"chatcmpl-...","choices":[{"index":0,"delta":{...},"finish_reason":null}]}

    Tool calls arrive as delta.tool_calls[] with index-based tracking.
    Arguments accumulate as string fragments in function.arguments.
    Text arrives as delta.content strings.
    """

    def __init__(self):
        self._tool_calls: dict[int, _OpenAIToolCall] = {}
        self._text_parts: list[str] = []
        self._message_id: str = ""
        self._model: str = ""
        self._sent_message_start: bool = False

    def feed_line(self, line: str) -> list[CrossEvent]:
        """Feed a single SSE line. Returns any events that are now complete."""
        line = line.rstrip("\r\n")

        if line.startswith("data: "):
            data_str = line[6:]
            if data_str == "[DONE]":
                return self._flush_text()
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                return []
            return self._process_chunk(data)

        # Empty line or event: line — OpenAI doesn't use event: for chat completions
        return []

    def _process_chunk(self, data: dict) -> list[CrossEvent]:
        events: list[CrossEvent] = []

        # Emit MessageStartEvent on first chunk
        if not self._sent_message_start:
            self._message_id = data.get("id", "")
            self._model = data.get("model", "")
            events.append(MessageStartEvent(message_id=self._message_id, model=self._model))
            self._sent_message_start = True

        choices = data.get("choices", [])
        if not choices:
            return events

        choice = choices[0]
        delta = choice.get("delta", {})
        finish_reason = choice.get("finish_reason")

        # Accumulate text content
        content = delta.get("content")
        if content:
            self._text_parts.append(content)

        # Accumulate tool calls
        tool_calls = delta.get("tool_calls", [])
        for tc_delta in tool_calls:
            tc_index = tc_delta.get("index", 0)
            tc = self._tool_calls.get(tc_index)

            if tc is None:
                # New tool call — first delta has id, type, function.name
                tc = _OpenAIToolCall(index=tc_index)
                self._tool_calls[tc_index] = tc

            # Update fields present in this delta
            if "id" in tc_delta:
                tc.tool_call_id = tc_delta["id"]
            func = tc_delta.get("function", {})
            if "name" in func:
                tc.function_name = func["name"]
            if "arguments" in func:
                tc.argument_parts.append(func["arguments"])

        # Check for finish
        if finish_reason == "tool_calls":
            # Flush any pending text first
            events.extend(self._flush_text())
            # Emit ToolUseEvents for all accumulated tool calls
            events.extend(self._flush_tool_calls())
            events.append(
                MessageDeltaEvent(
                    stop_reason="tool_use",  # Normalize to Anthropic's stop reason
                    output_tokens=data.get("usage", {}).get("completion_tokens", 0),
                )
            )
        elif finish_reason == "stop":
            events.extend(self._flush_text())
            events.append(
                MessageDeltaEvent(
                    stop_reason="end_turn",  # Normalize to Anthropic's stop reason
                    output_tokens=data.get("usage", {}).get("completion_tokens", 0),
                )
            )

        return events

    def _flush_text(self) -> list[CrossEvent]:
        """Emit a TextEvent if any text has accumulated."""
        if self._text_parts:
            text = "".join(self._text_parts)
            self._text_parts = []
            return [TextEvent(text=text)]
        return []

    def _flush_tool_calls(self) -> list[CrossEvent]:
        """Emit ToolUseEvents for all accumulated tool calls."""
        events: list[CrossEvent] = []
        for tc in sorted(self._tool_calls.values(), key=lambda t: t.index):
            args_json = "".join(tc.argument_parts)
            try:
                tool_input = json.loads(args_json) if args_json else {}
            except json.JSONDecodeError:
                tool_input = {"_raw": args_json}
            events.append(
                ToolUseEvent(
                    name=tc.function_name,
                    tool_use_id=tc.tool_call_id,
                    input=tool_input,
                )
            )
        self._tool_calls = {}
        return events


@dataclass
class _ResponsesToolCall:
    """Tracks state for a Responses API function_call being streamed."""

    output_index: int
    call_id: str = ""
    function_name: str = ""
    argument_parts: list[str] = field(default_factory=list)


class ResponsesSSEParser:
    """Stateful parser for OpenAI Responses API streaming SSE format.

    The Responses API uses event: lines (like Anthropic) with explicit types.
    Tool calls arrive as per-item start/delta/done events:
      - response.output_item.added  (function_call item)
      - response.function_call_arguments.delta
      - response.function_call_arguments.done
      - response.output_item.done

    Text arrives as:
      - response.output_text.done

    Message lifecycle:
      - response.created  → MessageStartEvent
      - response.completed → MessageDeltaEvent
    """

    def __init__(self):
        self._current_event_type: str = ""
        self._data_lines: list[str] = []
        self._tool_calls: dict[int, _ResponsesToolCall] = {}

    def feed_line(self, line: str) -> list[CrossEvent]:
        """Feed a single SSE line. Returns any events that are now complete."""
        line = line.rstrip("\r\n")

        if line.startswith("event: "):
            self._current_event_type = line[7:]
            return []

        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                return []
            self._data_lines.append(data)
            return []

        if line == "":
            # Empty line = end of SSE event, process accumulated data
            if self._data_lines:
                events = self._process_event()
                self._data_lines = []
                self._current_event_type = ""
                return events
            return []

        # Ignore other lines (comments starting with :, etc.)
        return []

    def _process_event(self) -> list[CrossEvent]:
        raw = "\n".join(self._data_lines)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []

        event_type = data.get("type", "")
        events: list[CrossEvent] = []

        if event_type == "response.created":
            resp = data.get("response", {})
            events.append(
                MessageStartEvent(
                    message_id=resp.get("id", ""),
                    model=resp.get("model", ""),
                )
            )

        elif event_type == "response.output_item.added":
            item = data.get("item", {})
            if item.get("type") == "function_call":
                output_index = data.get("output_index", 0)
                tc = _ResponsesToolCall(
                    output_index=output_index,
                    call_id=item.get("call_id", ""),
                    function_name=item.get("name", ""),
                )
                self._tool_calls[output_index] = tc

        elif event_type == "response.function_call_arguments.delta":
            output_index = data.get("output_index", 0)
            tc = self._tool_calls.get(output_index)
            if tc:
                tc.argument_parts.append(data.get("delta", ""))

        elif event_type == "response.output_item.done":
            item = data.get("item", {})
            if item.get("type") == "function_call":
                output_index = data.get("output_index", 0)
                tc = self._tool_calls.pop(output_index, None)
                if tc:
                    args_json = "".join(tc.argument_parts)
                    try:
                        tool_input = json.loads(args_json) if args_json else {}
                    except json.JSONDecodeError:
                        tool_input = {"_raw": args_json}
                    events.append(
                        ToolUseEvent(
                            name=tc.function_name,
                            tool_use_id=tc.call_id,
                            input=tool_input,
                        )
                    )

        elif event_type == "response.output_text.done":
            text = data.get("text", "")
            if text:
                events.append(TextEvent(text=text))

        elif event_type == "response.completed":
            resp = data.get("response", {})
            usage = resp.get("usage", {})
            status = resp.get("status", "")
            # Map Responses API status to normalized stop reason
            if status == "completed":
                # Check if there are function_call items in output
                output = resp.get("output", [])
                has_tool_calls = any(item.get("type") == "function_call" for item in output)
                stop_reason = "tool_use" if has_tool_calls else "end_turn"
            else:
                stop_reason = "end_turn"
            events.append(
                MessageDeltaEvent(
                    stop_reason=stop_reason,
                    output_tokens=usage.get("output_tokens", 0),
                )
            )

        return events
