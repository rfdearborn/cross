"""StreamFormat classes that encapsulate provider-specific SSE operations.

Each format knows how to:
- Create the right SSE parser
- Detect tool call starts in SSE lines
- Generate synthetic stop events
- Build retry and feedback messages for blocked tools
- Handle retry-specific line rewriting and skipping
- Detect non-tool content block completion
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod

from cross.sse import OpenAISSEParser, ResponsesSSEParser, SSEParser


class StreamFormat(ABC):
    """Abstract base for provider-specific streaming format handlers."""

    @abstractmethod
    def create_parser(self):
        """Create a new SSE parser for this format."""

    @abstractmethod
    def is_tool_start(self, line: str) -> bool:
        """Check if an SSE line marks the beginning of a tool call."""

    @abstractmethod
    def synthetic_stop_sse(self, **kwargs) -> list[str]:
        """Generate SSE lines to cleanly terminate a response."""

    @abstractmethod
    def build_retry_messages(self, blocked_tools: list[dict]) -> list[dict]:
        """Build messages to append for retry after blocked tools.

        Args:
            blocked_tools: List of dicts with keys: tool_use_id, name, input, reason.

        Returns:
            List of message dicts to append to the messages array.
        """

    @abstractmethod
    def build_feedback_messages(self, to_inject: list[tuple[str, str, dict]]) -> list[dict]:
        """Build messages for next-request blocked-tool feedback injection.

        Args:
            to_inject: List of (tool_id, reason, info) tuples where info has
                       keys 'name' and 'input'.

        Returns:
            List of message dicts to append to the messages array.
        """

    @abstractmethod
    def rewrite_line_for_retry(self, line: str, next_content_index: int) -> str:
        """Rewrite an SSE line for a retry response (e.g., offset content block indices).

        Args:
            line: The SSE line to potentially rewrite.
            next_content_index: The content block index offset.

        Returns:
            The (possibly rewritten) line.
        """

    @abstractmethod
    def should_skip_on_retry(self, line: str) -> bool:
        """Check if a line should be skipped entirely during retry responses.

        Args:
            line: The SSE line to check.

        Returns:
            True if this line should be dropped from the retry stream.
        """

    @abstractmethod
    def is_content_block_stop(self, line: str) -> bool:
        """Check if an SSE line is a content_block_stop (non-tool block completion).

        Used to track how many content blocks have been sent to the client,
        for index rewriting in retries.
        """

    @property
    @abstractmethod
    def uses_event_lines(self) -> bool:
        """Whether this format uses 'event: ...' lines that need pending-line handling."""

    @property
    @abstractmethod
    def upstream_host(self) -> str:
        """The default upstream hostname for this format (e.g., 'api.anthropic.com')."""


class AnthropicFormat(StreamFormat):
    """Anthropic Messages API streaming format."""

    @property
    def upstream_host(self) -> str:
        return "api.anthropic.com"

    def create_parser(self):
        return SSEParser()

    def is_tool_start(self, line: str) -> bool:
        if not line.startswith("data: "):
            return False
        try:
            data = json.loads(line[6:])
            return data.get("type") == "content_block_start" and data.get("content_block", {}).get("type") == "tool_use"
        except (json.JSONDecodeError, AttributeError):
            return False

    def synthetic_stop_sse(self, **kwargs) -> list[str]:
        return [
            "event: message_delta",
            'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}',
            "",
            "event: message_stop",
            'data: {"type":"message_stop"}',
            "",
        ]

    def build_retry_messages(self, blocked_tools: list[dict]) -> list[dict]:
        tool_use_blocks = []
        tool_result_blocks = []
        for bt in blocked_tools:
            tool_use_blocks.append(
                {
                    "type": "tool_use",
                    "id": bt["tool_use_id"],
                    "name": bt["name"],
                    "input": bt["input"],
                }
            )
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": bt["tool_use_id"],
                    "content": f"[Cross blocked this tool call: {bt['reason']}]",
                    "is_error": True,
                }
            )
        return [
            {"role": "assistant", "content": tool_use_blocks},
            {"role": "user", "content": tool_result_blocks},
        ]

    def build_feedback_messages(self, to_inject: list[tuple[str, str, dict]]) -> list[dict]:
        tool_use_blocks = []
        tool_result_blocks = []
        for tool_id, reason, info in to_inject:
            tool_use_blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": info["name"],
                    "input": info["input"],
                }
            )
            tool_result_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": f"[Cross blocked this tool call: {reason}]",
                    "is_error": True,
                }
            )
        return [
            {"role": "assistant", "content": tool_use_blocks},
            {"role": "user", "content": tool_result_blocks},
        ]

    def rewrite_line_for_retry(self, line: str, next_content_index: int) -> str:
        if next_content_index == 0 or not line.startswith("data: "):
            return line
        try:
            data = json.loads(line[6:])
            if data.get("type") in (
                "content_block_start",
                "content_block_delta",
                "content_block_stop",
            ):
                if "index" in data:
                    data["index"] = data["index"] + next_content_index
                    return f"data: {json.dumps(data)}"
        except (json.JSONDecodeError, TypeError):
            pass
        return line

    def should_skip_on_retry(self, line: str) -> bool:
        # Skip message_start events in retry responses
        if line.startswith("event: message_start"):
            return True
        if line.startswith("data: "):
            try:
                d = json.loads(line[6:])
                if d.get("type") == "message_start":
                    return True
            except (json.JSONDecodeError, TypeError):
                pass
        return False

    def is_content_block_stop(self, line: str) -> bool:
        if not line.startswith("data: "):
            return False
        try:
            d = json.loads(line[6:])
            return d.get("type") == "content_block_stop"
        except (json.JSONDecodeError, TypeError):
            return False

    @property
    def uses_event_lines(self) -> bool:
        return True


class OpenAIFormat(StreamFormat):
    """OpenAI Chat Completions API streaming format."""

    @property
    def upstream_host(self) -> str:
        return "api.openai.com"

    def create_parser(self):
        return OpenAISSEParser()

    def is_tool_start(self, line: str) -> bool:
        if not line.startswith("data: "):
            return False
        try:
            data = json.loads(line[6:])
            choices = data.get("choices", [])
            if choices:
                tool_calls = choices[0].get("delta", {}).get("tool_calls", [])
                return any("id" in tc for tc in tool_calls)
        except (json.JSONDecodeError, AttributeError):
            pass
        return False

    def synthetic_stop_sse(self, **kwargs) -> list[str]:
        chunk_id = kwargs.get("chunk_id", "chatcmpl-cross-stop")
        model = kwargs.get("model", "")
        chunk = {
            "id": chunk_id or "chatcmpl-cross-stop",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        return [f"data: {json.dumps(chunk)}", "", "data: [DONE]", ""]

    def build_retry_messages(self, blocked_tools: list[dict]) -> list[dict]:
        tool_calls = []
        tool_messages = []
        for bt in blocked_tools:
            tool_calls.append(
                {
                    "id": bt["tool_use_id"],
                    "type": "function",
                    "function": {
                        "name": bt["name"],
                        "arguments": json.dumps(bt["input"]),
                    },
                }
            )
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": bt["tool_use_id"],
                    "content": f"[Cross blocked this tool call: {bt['reason']}]",
                }
            )
        return [{"role": "assistant", "tool_calls": tool_calls}] + tool_messages

    def build_feedback_messages(self, to_inject: list[tuple[str, str, dict]]) -> list[dict]:
        tool_calls = []
        tool_messages = []
        for tool_id, reason, info in to_inject:
            tool_calls.append(
                {
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": info["name"],
                        "arguments": json.dumps(info["input"]),
                    },
                }
            )
            tool_messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": f"[Cross blocked this tool call: {reason}]",
                }
            )
        return [{"role": "assistant", "tool_calls": tool_calls}] + tool_messages

    def rewrite_line_for_retry(self, line: str, next_content_index: int) -> str:
        # OpenAI doesn't use content block indices — no-op
        return line

    def should_skip_on_retry(self, line: str) -> bool:
        # OpenAI doesn't need to skip any lines on retry
        return False

    def is_content_block_stop(self, line: str) -> bool:
        # OpenAI doesn't have content_block_stop events
        return False

    @property
    def uses_event_lines(self) -> bool:
        return False


class ResponsesFormat(StreamFormat):
    """OpenAI Responses API streaming format.

    Uses event: lines (like Anthropic) with per-item start/delta/done events.
    Tool calls are function_call items tracked by output_index.
    """

    @property
    def upstream_host(self) -> str:
        return "api.openai.com"

    def create_parser(self):
        return ResponsesSSEParser()

    def is_tool_start(self, line: str) -> bool:
        if not line.startswith("data: "):
            return False
        try:
            data = json.loads(line[6:])
            return (
                data.get("type") == "response.output_item.added" and data.get("item", {}).get("type") == "function_call"
            )
        except (json.JSONDecodeError, AttributeError):
            return False

    def synthetic_stop_sse(self, **kwargs) -> list[str]:
        resp_id = kwargs.get("response_id", "resp-cross-stop")
        model = kwargs.get("model", "")
        resp = {
            "type": "response.completed",
            "response": {
                "id": resp_id or "resp-cross-stop",
                "model": model,
                "status": "completed",
                "output": [],
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        }
        return [
            "event: response.completed",
            f"data: {json.dumps(resp)}",
            "",
        ]

    def build_retry_messages(self, blocked_tools: list[dict]) -> list[dict]:
        """Build input items to append for retry after blocked tools.

        Returns a list of Responses API input items (function_call + function_call_output).
        """
        items: list[dict] = []
        for bt in blocked_tools:
            items.append(
                {
                    "type": "function_call",
                    "call_id": bt["tool_use_id"],
                    "name": bt["name"],
                    "arguments": json.dumps(bt["input"]),
                }
            )
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": bt["tool_use_id"],
                    "output": f"[Cross blocked this tool call: {bt['reason']}]",
                }
            )
        return items

    def build_feedback_messages(self, to_inject: list[tuple[str, str, dict]]) -> list[dict]:
        """Build input items for next-request blocked-tool feedback injection."""
        items: list[dict] = []
        for tool_id, reason, info in to_inject:
            items.append(
                {
                    "type": "function_call",
                    "call_id": tool_id,
                    "name": info["name"],
                    "arguments": json.dumps(info["input"]),
                }
            )
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": tool_id,
                    "output": f"[Cross blocked this tool call: {reason}]",
                }
            )
        return items

    def rewrite_line_for_retry(self, line: str, next_content_index: int) -> str:
        # Responses API doesn't use content block indices — no rewriting needed
        return line

    def should_skip_on_retry(self, line: str) -> bool:
        # Skip response.created events in retry responses
        if line.startswith("event: response.created"):
            return True
        if line.startswith("data: "):
            try:
                d = json.loads(line[6:])
                if d.get("type") == "response.created":
                    return True
            except (json.JSONDecodeError, TypeError):
                pass
        return False

    def is_content_block_stop(self, line: str) -> bool:
        # Responses API uses response.output_item.done for item completion
        if not line.startswith("data: "):
            return False
        try:
            d = json.loads(line[6:])
            return d.get("type") == "response.output_item.done"
        except (json.JSONDecodeError, TypeError):
            return False

    @property
    def uses_event_lines(self) -> bool:
        return True


class ResponsesChatGPTFormat(ResponsesFormat):
    """OpenAI Responses API via ChatGPT backend (chatgpt.com).

    Inherits all behavior from ResponsesFormat but routes to the ChatGPT
    backend host. ChatGPT OAuth tokens (JWTs) are only valid at chatgpt.com,
    not at api.openai.com.
    """

    @property
    def upstream_host(self) -> str:
        return "chatgpt.com"


# Singleton instances
_ANTHROPIC_FORMAT = AnthropicFormat()
_OPENAI_FORMAT = OpenAIFormat()
_RESPONSES_FORMAT = ResponsesFormat()
_RESPONSES_CHATGPT_FORMAT = ResponsesChatGPTFormat()


def get_format(api_format: str, chatgpt_oauth: bool = False) -> StreamFormat:
    """Get the StreamFormat instance for the given API format string.

    Args:
        api_format: One of "anthropic", "openai", "responses".
        chatgpt_oauth: If True and format is "responses", return
            ResponsesChatGPTFormat for routing to chatgpt.com.
    """
    if api_format == "openai":
        return _OPENAI_FORMAT
    if api_format == "responses":
        if chatgpt_oauth:
            return _RESPONSES_CHATGPT_FORMAT
        return _RESPONSES_FORMAT
    return _ANTHROPIC_FORMAT


def is_chatgpt_oauth(headers: dict) -> bool:
    """Check if the request uses ChatGPT OAuth (JWT token, not API key).

    ChatGPT OAuth tokens are JWTs (start with "Bearer eyJ").
    API keys start with "Bearer sk-".
    """
    auth = headers.get("authorization", "")
    return auth.startswith("Bearer eyJ")
