"""SSE stream parser for the Anthropic Messages API.

Parses SSE lines and emits CrossEvent objects as complete events are assembled.
Handles accumulation of input_json_delta chunks into full tool inputs.
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
    """Stateful parser that processes SSE lines and yields CrossEvent objects."""

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
            events.append(MessageStartEvent(
                message_id=msg.get("id", ""),
                model=msg.get("model", ""),
            ))

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
                    events.append(ToolUseEvent(
                        name=cb.tool_name,
                        tool_use_id=cb.tool_use_id,
                        input=tool_input,
                    ))
                elif cb.block_type == "text":
                    events.append(TextEvent(text="".join(cb.text_parts)))

        elif event_type == "message_delta":
            delta = data.get("delta", {})
            usage = data.get("usage", {})
            events.append(MessageDeltaEvent(
                stop_reason=delta.get("stop_reason"),
                output_tokens=usage.get("output_tokens", 0),
            ))

        return events
