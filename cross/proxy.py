"""Core reverse proxy — forwards requests to Anthropic API and parses SSE responses.

When a GateChain is configured, tool_use blocks in streaming responses are buffered
until the gate evaluates them. Text blocks stream through with zero added latency.

Gate actions:
  BLOCK: suppress tool_use, make a new API call with error feedback injected, stream
         that retry response back. The agent self-corrects. After max retries, fall
         back to HALT_SESSION behavior.
  HALT_SESSION: suppress tool_use, send synthetic message_stop, store blocked tool info
         for next-request injection. Freeze the agent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from cross.chain import GateChain
from cross.config import settings
from cross.evaluator import Action, GateRequest
from cross.events import ErrorEvent, EventBus, GateDecisionEvent, GateRetryEvent, RequestEvent, ToolUseEvent
from cross.script_resolver import resolve_script_contents
from cross.stream_formats import get_format

logger = logging.getLogger("cross.proxy")

# Tool names that execute bash commands (for script resolution)
_BASH_TOOL_NAMES = {"bash", "exec"}


def _resolve_scripts_for_tool(tool_name: str, tool_input: Any, cwd: str = "") -> dict[str, str]:
    """Resolve script file contents if this is a Bash/exec tool call."""
    if tool_name.lower() not in _BASH_TOOL_NAMES:
        return {}
    command = ""
    if isinstance(tool_input, dict):
        command = tool_input.get("command", "")
    elif isinstance(tool_input, str):
        command = tool_input
    if not command:
        return {}
    return resolve_script_contents(command, cwd=cwd)


_client: httpx.AsyncClient | None = None
_openai_client: httpx.AsyncClient | None = None
_chatgpt_client: httpx.AsyncClient | None = None

# API format detection
API_FORMAT_ANTHROPIC = "anthropic"
API_FORMAT_OPENAI = "openai"
API_FORMAT_RESPONSES = "responses"

# OpenAI API paths
_OPENAI_CHAT_PATHS = ("/v1/chat/completions",)
_OPENAI_RESPONSES_PATHS = ("/v1/responses", "/responses")
_OPENAI_GENERIC_PATHS = ("/v1/models",)  # Non-streaming OpenAI paths that need correct routing

# Blocked tool_use_ids -> reason, info, timestamp for next-request feedback injection
_blocked_tool_ids: dict[str, str] = {}
_blocked_tool_info: dict[str, dict] = {}  # tool_use_id -> {name, input}
_blocked_tool_timestamps: dict[str, float] = {}

# Recent tool calls for LLM gate context (bounded deque)
_recent_tools: deque[dict[str, Any]] = deque(maxlen=max(settings.llm_gate_context_tools, 1))

# Safety limits for SSE buffering
_MAX_BUFFER_LINES = 500  # Flush buffer if it exceeds this many lines
_BLOCKED_TOOL_TTL = 300.0  # Seconds before stale blocked tool entries are cleaned up

# Sentinel halt flag — when set, proxy rejects all requests
_sentinel_halted = False
_sentinel_halt_reason = ""


def set_sentinel_halt(reason: str) -> None:
    """Called by the sentinel when it issues a HALT verdict."""
    global _sentinel_halted, _sentinel_halt_reason
    _sentinel_halted = True
    _sentinel_halt_reason = reason
    logger.critical(f"Proxy halted by sentinel: {reason}")


def is_sentinel_halted() -> bool:
    """Check if the sentinel has halted the proxy."""
    return _sentinel_halted


# Gate approval infrastructure (for ESCALATE → human review)
_pending_approvals: dict[str, asyncio.Event] = {}
_approval_results: dict[str, tuple[bool, str]] = {}  # tool_use_id -> (approved, username)


def resolve_gate_approval(tool_use_id: str, approved: bool, username: str = ""):
    """Called (from any thread) when a human approves/denies a gate escalation."""
    _approval_results[tool_use_id] = (approved, username)
    event = _pending_approvals.get(tool_use_id)
    if event:
        event.set()


def _is_chatgpt_oauth(headers: dict) -> bool:
    """Check if the request uses ChatGPT OAuth (JWT token, not API key)."""
    from cross.stream_formats import is_chatgpt_oauth

    return is_chatgpt_oauth(headers)


@dataclass
class ProxyRequest:
    """Parsed proxy request with all context needed for routing, gating, and forwarding."""

    method: str
    path: str  # After session prefix stripping and /v1 stripping for ChatGPT
    headers: dict
    body: bytes  # Original body for forwarding upstream
    body_parsed: bytes  # Decompressed body for JSON parsing
    api_format: str
    chatgpt_oauth: bool
    session_id: str
    agent_label: str
    # Extracted from parsed body
    model: str = ""
    user_intent: str = ""
    conversation_context: list = field(default_factory=list)
    is_streaming: bool = False


def get_client(api_format: str = API_FORMAT_ANTHROPIC, chatgpt_oauth: bool = False) -> httpx.AsyncClient:
    """Get the httpx client for the given API format and auth type."""
    global _client, _openai_client, _chatgpt_client
    if chatgpt_oauth:
        if _chatgpt_client is None:
            _chatgpt_client = httpx.AsyncClient(
                base_url=settings.chatgpt_base_url,
                timeout=httpx.Timeout(connect=10, read=300, write=30, pool=10),
            )
        return _chatgpt_client
    elif api_format in (API_FORMAT_OPENAI, API_FORMAT_RESPONSES):
        if _openai_client is None:
            _openai_client = httpx.AsyncClient(
                base_url=settings.openai_base_url,
                timeout=httpx.Timeout(connect=10, read=300, write=30, pool=10),
            )
        return _openai_client
    else:
        if _client is None:
            _client = httpx.AsyncClient(
                base_url=settings.anthropic_base_url,
                timeout=httpx.Timeout(connect=10, read=300, write=30, pool=10),
            )
        return _client


def detect_api_format(path: str) -> str:
    """Detect API format from request path: responses, openai, or anthropic."""
    # Strip query string for matching
    clean = path.split("?")[0]
    if any(clean.startswith(p) for p in _OPENAI_RESPONSES_PATHS):
        return API_FORMAT_RESPONSES
    if any(clean.startswith(p) for p in _OPENAI_CHAT_PATHS):
        return API_FORMAT_OPENAI
    if any(clean.startswith(p) for p in _OPENAI_GENERIC_PATHS):
        return API_FORMAT_OPENAI
    return API_FORMAT_ANTHROPIC


_SKIP_PREFIXES = ("<system-reminder>", "[Request interrupted by user]", "Conversation info")


def _extract_user_intent(req_data: dict) -> str:
    """Extract user text from the last user message.

    Supports both Chat Completions (messages array) and Responses API (input array).
    """
    msgs = req_data.get("messages", [])
    # Responses API uses 'input' instead of 'messages'
    if not msgs:
        input_items = req_data.get("input", [])
        if isinstance(input_items, str):
            # Simple string input
            if input_items and not input_items.startswith(_SKIP_PREFIXES):
                return input_items[: settings.llm_gate_context_intent_chars]
            return ""
        if isinstance(input_items, list):
            for item in reversed(input_items):
                if item.get("role") == "user":
                    content = item.get("content", "")
                    if isinstance(content, str):
                        if content and not content.startswith(_SKIP_PREFIXES):
                            return content[: settings.llm_gate_context_intent_chars]
                    elif isinstance(content, list):
                        for b in reversed(content):
                            if b.get("type") in ("input_text", "text"):
                                text = b.get("text", "")
                                if text and not text.startswith(_SKIP_PREFIXES):
                                    return text[: settings.llm_gate_context_intent_chars]
            return ""
    for msg in reversed(msgs):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            if content and not content.startswith(_SKIP_PREFIXES):
                return content[: settings.llm_gate_context_intent_chars]
        elif isinstance(content, list):
            for b in reversed(content):
                if b.get("type") == "text":
                    text = b.get("text", "")
                    if text and not text.startswith(_SKIP_PREFIXES):
                        return text[: settings.llm_gate_context_intent_chars]
        return ""
    return ""


def _extract_conversation_context(
    req_data: dict,
    max_turns: int | None = None,
    max_chars_per_turn: int | None = None,
) -> list[dict[str, str]]:
    """Extract recent user/assistant text exchanges from the messages array.

    Returns a chronological list of {"role": "user"|"assistant", "text": "..."} dicts,
    capped at max_turns, each text truncated to max_chars_per_turn.
    """
    if max_turns is None:
        max_turns = settings.llm_gate_context_turns
    if max_chars_per_turn is None:
        max_chars_per_turn = settings.llm_gate_context_chars_per_turn
    msgs = req_data.get("messages", [])
    # Responses API uses 'input' instead of 'messages'
    if not msgs:
        input_items = req_data.get("input", [])
        if isinstance(input_items, list):
            msgs = [item for item in input_items if isinstance(item, dict) and item.get("role")]
    turns: list[dict[str, str]] = []

    for msg in reversed(msgs):
        if len(turns) >= max_turns:
            break
        role = msg.get("role")
        if role not in ("user", "assistant"):
            continue
        content = msg.get("content", "")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            # Extract text blocks only (skip tool_use, tool_result, images, etc.)
            text_parts = []
            for b in content:
                if b.get("type") in ("text", "input_text"):
                    text_parts.append(b.get("text", ""))
            text = " ".join(text_parts)
        if not text or text.startswith(_SKIP_PREFIXES):
            continue
        turns.append({"role": role, "text": text[:max_chars_per_turn]})

    turns.reverse()  # chronological order
    return turns


def _extract_request_event(method: str, path: str, body: bytes | None, session_id: str = "") -> RequestEvent:
    from cross.daemon import get_agent_label

    event = RequestEvent(method=method, path=path, agent=get_agent_label(session_id), session_id=session_id)
    if body:
        try:
            data = json.loads(body)
            event.model = data.get("model")
            event.stream = data.get("stream", False)
            event.raw_body = data

            # Responses API uses 'input' instead of 'messages'
            msgs = data.get("messages", [])
            input_items = data.get("input", [])
            if msgs:
                event.messages_count = len(msgs)
            elif isinstance(input_items, list):
                event.messages_count = len(input_items)

            tools = data.get("tools", [])
            # OpenAI nests tool names under function.name; Anthropic uses top-level name
            event.tool_names = [t.get("function", {}).get("name", t.get("name", "?")) for t in tools]

            if msgs:
                last = msgs[-1]
                event.last_message_role = last.get("role")
            intent = _extract_user_intent(data)
            if intent:
                event.last_message_preview = intent[:200]
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
            pass
    return event


def _inject_blocked_tool_feedback(body: bytes, api_format: str = API_FORMAT_ANTHROPIC) -> bytes:
    """Inject blocked-tool feedback into the next request's messages.

    Since blocked tool_use blocks are suppressed from SSE responses, the client
    never sees them and won't send tool_results. We reconstruct the exchange by
    appending the tool_use to the last assistant message and prepending an error
    tool_result to the following user message.
    """
    # Clean up stale entries
    if _blocked_tool_timestamps:
        now = time.time()
        stale = [k for k, t in _blocked_tool_timestamps.items() if now - t > _BLOCKED_TOOL_TTL]
        for k in stale:
            _blocked_tool_ids.pop(k, None)
            _blocked_tool_info.pop(k, None)
            _blocked_tool_timestamps.pop(k, None)

    if not _blocked_tool_ids:
        return body

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body

    # Responses API uses 'input' not 'messages'
    if api_format == API_FORMAT_RESPONSES:
        input_items = data.get("input", [])
        if not isinstance(input_items, list):
            return body

        # Collect blocked tools with full info
        to_inject: list[tuple[str, str, dict]] = []
        for tool_id in list(_blocked_tool_ids.keys()):
            reason = _blocked_tool_ids.pop(tool_id)
            info = _blocked_tool_info.pop(tool_id, None)
            _blocked_tool_timestamps.pop(tool_id, None)
            if info:
                to_inject.append((tool_id, reason, info))

        if not to_inject:
            return body

        fmt = get_format(api_format)
        input_items.extend(fmt.build_feedback_messages(to_inject))
        data["input"] = input_items
        logger.info(f"Injected feedback for {len(to_inject)} blocked tool(s) into next request")
        return json.dumps(data).encode()

    messages = data.get("messages", [])
    if not messages:
        return body

    # Collect blocked tools with full info
    to_inject: list[tuple[str, str, dict]] = []  # (tool_id, reason, info)
    for tool_id in list(_blocked_tool_ids.keys()):
        reason = _blocked_tool_ids.pop(tool_id)
        info = _blocked_tool_info.pop(tool_id, None)
        _blocked_tool_timestamps.pop(tool_id, None)
        if info:
            to_inject.append((tool_id, reason, info))

    if not to_inject:
        return body

    if api_format == API_FORMAT_OPENAI:
        # OpenAI format: use StreamFormat to build feedback messages
        fmt = get_format(api_format)
        messages.extend(fmt.build_feedback_messages(to_inject))
    else:
        # Anthropic format: tool_use blocks in assistant message, tool_result blocks in user message
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

        # Find the last assistant message and append tool_use blocks to it
        last_assistant_idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "assistant":
                last_assistant_idx = i
                break

        if last_assistant_idx is None:
            insert_at = len(messages)
            for i in range(len(messages) - 1, -1, -1):
                if messages[i].get("role") == "user":
                    insert_at = i
                    break
            messages.insert(insert_at, {"role": "user", "content": tool_result_blocks})
            messages.insert(insert_at, {"role": "assistant", "content": tool_use_blocks})
        else:
            content = messages[last_assistant_idx].get("content", [])
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            content.extend(tool_use_blocks)
            messages[last_assistant_idx]["content"] = content

            next_user_idx = None
            for i in range(last_assistant_idx + 1, len(messages)):
                if messages[i].get("role") == "user":
                    next_user_idx = i
                    break

            if next_user_idx is not None:
                user_content = messages[next_user_idx].get("content", [])
                if isinstance(user_content, str):
                    user_content = [{"type": "text", "text": user_content}]
                messages[next_user_idx]["content"] = tool_result_blocks + user_content
            else:
                messages.append({"role": "user", "content": tool_result_blocks})

    data["messages"] = messages
    logger.info(f"Injected feedback for {len(to_inject)} blocked tool(s) into next request")
    return json.dumps(data).encode()


def _build_retry_request_body(
    original_body: bytes,
    blocked_tools: list[dict],
    api_format: str = API_FORMAT_ANTHROPIC,
) -> bytes:
    """Build a new request body with blocked tool feedback injected for retry.

    Takes the original request body and appends:
    - Anthropic: assistant message with tool_use blocks + user message with tool_result blocks
    - OpenAI Chat: assistant message with tool_calls + tool messages with error content
    - OpenAI Responses: function_call + function_call_output items in the input array

    Args:
        original_body: The original request body bytes.
        blocked_tools: List of dicts with keys: tool_use_id, name, input, reason.
        api_format: API format (anthropic, openai, or responses).

    Returns:
        New request body bytes with feedback injected.
    """
    fmt = get_format(api_format)
    data = json.loads(original_body)
    if api_format == API_FORMAT_RESPONSES:
        # Responses API: extend the 'input' array with feedback items
        input_items = data.get("input", [])
        if isinstance(input_items, str):
            # Convert simple string input to a list
            input_items = [{"role": "user", "content": input_items}]
        input_items.extend(fmt.build_retry_messages(blocked_tools))
        data["input"] = input_items
    else:
        messages = data.get("messages", [])
        messages.extend(fmt.build_retry_messages(blocked_tools))
        data["messages"] = messages
    return json.dumps(data).encode()


def _rewrite_content_block_index(line: str, offset: int) -> str:
    """Rewrite the index field in content_block_start/delta/stop SSE events.

    Thin wrapper delegating to AnthropicFormat.rewrite_line_for_retry().
    """
    return get_format(API_FORMAT_ANTHROPIC).rewrite_line_for_retry(line, offset)


def _build_proxy_request(
    method: str,
    path: str,
    headers: dict,
    body: bytes,
    body_parsed: bytes,
    session_id: str,
) -> ProxyRequest:
    """Build a ProxyRequest from raw HTTP request data.

    Handles API format detection, ChatGPT OAuth detection, host header
    routing, /v1 path stripping, and request body context extraction.
    """
    from cross.daemon import get_agent_label

    api_format = detect_api_format(path)

    # Detect ChatGPT OAuth (JWT) vs API key auth for OpenAI routing.
    chatgpt_oauth = api_format in (API_FORMAT_OPENAI, API_FORMAT_RESPONSES) and _is_chatgpt_oauth(headers)

    # Use the format to determine upstream host and path adjustments.
    fmt = get_format(api_format, chatgpt_oauth=chatgpt_oauth)

    if chatgpt_oauth and path.startswith("/v1/"):
        # ChatGPT base URL already includes the full path prefix,
        # so strip /v1 from the request path to avoid doubling.
        path = "/" + path[4:]  # /v1/responses -> /responses

    headers["host"] = fmt.upstream_host
    headers["content-length"] = str(len(body))

    agent_label = get_agent_label(session_id)

    # Extract context from parsed body
    model = ""
    user_intent = ""
    conversation_context: list = []
    is_streaming = False
    if body_parsed:
        try:
            req_data = json.loads(body_parsed)
            model = req_data.get("model", "")
            user_intent = _extract_user_intent(req_data)
            conversation_context = _extract_conversation_context(req_data)
            is_streaming = req_data.get("stream", False)
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
            pass

    return ProxyRequest(
        method=method,
        path=path,
        headers=headers,
        body=body,
        body_parsed=body_parsed,
        api_format=api_format,
        chatgpt_oauth=chatgpt_oauth,
        session_id=session_id,
        agent_label=agent_label,
        model=model,
        user_intent=user_intent,
        conversation_context=conversation_context,
        is_streaming=is_streaming,
    )


async def handle_proxy_request(
    request: Request,
    event_bus: EventBus,
    gate_chain: GateChain | None = None,
    session_id: str = "",
) -> Response:
    """Handle a proxy request, publishing events to the given EventBus."""
    # Reject WebSocket upgrade requests — we don't support WebSocket proxying.
    # Returning 426 triggers Codex's fast-path fallback to HTTP SSE.
    if request.headers.get("upgrade", "").lower() == "websocket":
        return Response(
            content=b'{"error": "WebSocket not supported, use HTTP"}',
            status_code=426,
            headers={"content-type": "application/json"},
        )

    if _sentinel_halted:
        return Response(
            content=json.dumps(
                {
                    "type": "error",
                    "error": {
                        "type": "request_blocked",
                        "message": f"Session halted by sentinel: {_sentinel_halt_reason}",
                    },
                }
            ).encode(),
            status_code=403,
            headers={"content-type": "application/json"},
        )

    body = await request.body()

    # Decompress zstd-encoded request bodies for parsing (Codex uses zstd).
    # We keep the original compressed body for forwarding upstream.
    body_for_parsing = body
    if body[:4] == b"\x28\xb5\x2f\xfd":  # zstd magic number
        try:
            import zstandard

            # Use streaming decompression — frames may lack content size
            dctx = zstandard.ZstdDecompressor()
            body_for_parsing = dctx.stream_reader(body).read()
        except ImportError:
            logger.warning("zstd request body but zstandard package not installed")
        except Exception as e:
            logger.warning(f"Failed to decompress zstd request: {e}")

    # Use the path param (already stripped of /s/{session_id} prefix by Starlette)
    path = "/" + request.path_params.get("path", "")
    if request.url.query:
        path = f"{path}?{request.url.query}"

    # Intercept blocked tool_results before forwarding.
    body_before_feedback = body_for_parsing
    body_for_parsing = _inject_blocked_tool_feedback(body_for_parsing, api_format=detect_api_format(path))
    if body_for_parsing is not body_before_feedback:
        # Feedback was injected — forward the modified (decompressed) body
        body = body_for_parsing

    # Build upstream headers — forward everything except host
    headers = dict(request.headers)
    headers.pop("host", None)

    # Build the ProxyRequest with all context
    preq = _build_proxy_request(
        method=request.method,
        path=path,
        headers=headers,
        body=body,
        body_parsed=body_for_parsing,
        session_id=session_id,
    )

    # Record session activity for status tracking
    from cross.daemon import record_session_activity
    record_session_activity(session_id)

    # Publish request event (use decompressed body for parsing)
    req_event = _extract_request_event(preq.method, preq.path, preq.body_parsed, session_id=session_id)
    await event_bus.publish(req_event)

    client = get_client(preq.api_format, chatgpt_oauth=preq.chatgpt_oauth)

    if preq.is_streaming:
        return await _proxy_streaming(
            client,
            event_bus,
            preq.method,
            preq.path,
            preq.headers,
            preq.body,
            gate_chain,
            session_id=preq.session_id,
            agent_label=preq.agent_label,
            api_format=preq.api_format,
            body_for_parsing=preq.body_parsed,
            user_intent=preq.user_intent,
            model=preq.model,
            conversation_context=preq.conversation_context,
        )
    else:
        return await _proxy_simple(
            client,
            event_bus,
            preq.method,
            preq.path,
            preq.headers,
            preq.body,
            gate_chain,
            session_id=preq.session_id,
            agent_label=preq.agent_label,
            api_format=preq.api_format,
            request_body_parsed=preq.body_parsed,
            user_intent=preq.user_intent,
            model=preq.model,
            conversation_context=preq.conversation_context,
        )


async def _proxy_simple(
    client: httpx.AsyncClient,
    event_bus: EventBus,
    method: str,
    path: str,
    headers: dict,
    body: bytes,
    gate_chain: GateChain | None = None,
    session_id: str = "",
    agent_label: str = "",
    api_format: str = API_FORMAT_ANTHROPIC,
    request_body_parsed: bytes | None = None,
    user_intent: str = "",
    model: str = "",
    conversation_context: list | None = None,
) -> Response:
    resp = await client.request(method, path, headers=headers, content=body)

    if resp.status_code >= 400:
        # Suppress non-API error noise:
        # - HEAD 404: Claude Code session registration ping
        # - HTML responses: Starlette/CDN error pages (not API errors)
        is_session_ping = method == "HEAD" and resp.status_code == 404
        is_html_error = resp.text.lstrip().startswith("<")
        if not is_session_ping and not is_html_error:
            await event_bus.publish(
                ErrorEvent(
                    status_code=resp.status_code,
                    body=resp.text[:500],
                )
            )

    content = resp.content

    # Gate tool_use blocks in non-streaming responses
    if gate_chain and resp.status_code == 200:
        try:
            # ChatGPT backend returns SSE even for non-streaming requests.
            # Parse the SSE to extract tool calls for gating.
            parsed_body = request_body_parsed or body
            if content.startswith(b"event: "):
                content = await _gate_sse_response(
                    content,
                    parsed_body,
                    gate_chain,
                    event_bus,
                    session_id=session_id,
                    agent_label=agent_label,
                    api_format=api_format,
                    user_intent=user_intent,
                    model=model,
                    conversation_context=conversation_context,
                )
            else:
                content = await _gate_non_streaming_response(
                    content,
                    parsed_body,
                    gate_chain,
                    event_bus,
                    session_id=session_id,
                    agent_label=agent_label,
                    api_format=api_format,
                    user_intent=user_intent,
                    model=model,
                    conversation_context=conversation_context,
                )
        except Exception as e:
            logger.warning(f"Non-streaming gate check failed: {e}")

    resp_headers = dict(resp.headers)
    resp_headers.pop("transfer-encoding", None)
    resp_headers.pop("content-encoding", None)

    return Response(
        content=content,
        status_code=resp.status_code,
        headers=resp_headers,
    )


async def _process_gate_result(
    result,
    tool_event: ToolUseEvent,
    _publish,
    any_halted: bool,
    cascade_on_block: bool = False,
) -> tuple[bool, bool, str]:
    """Shared gate result processing for SSE, non-streaming, and streaming paths.

    Handles ESCALATE, HALT_SESSION, BLOCK, and ALERT actions uniformly.
    Records blocked tools in module-level state and logs decisions.

    Args:
        result: EvaluationResponse from the gate chain.
        tool_event: The tool event being gated.
        _publish: Async function to publish events.
        any_halted: Whether a preceding tool was halted/blocked.
        cascade_on_block: If True, BLOCK also cascades to subsequent tools
            (sets halted=True). Used in non-streaming/SSE-in-body paths where
            there is no retry mechanism.

    Returns:
        (blocked, halted, reason) where:
        - blocked: True if this tool was blocked
        - halted: Updated any_halted flag
        - reason: Block reason string (empty if not blocked)
    """
    if result.action in (Action.REVIEW, Action.ESCALATE) and not any_halted:
        approved, username, reason = await _handle_escalation(
            tool_event,
            _publish,
            any_halted,
        )
        if approved:
            return False, any_halted, ""
        _blocked_tool_ids[tool_event.tool_use_id] = reason
        _blocked_tool_info[tool_event.tool_use_id] = {
            "name": tool_event.name,
            "input": tool_event.input,
        }
        _blocked_tool_timestamps[tool_event.tool_use_id] = time.time()
        halted = True if cascade_on_block else any_halted
        return True, halted, reason

    if result.action in (Action.HALT_SESSION, Action.BLOCK) or any_halted:
        reason = (
            result.reason
            if result.action in (Action.HALT_SESSION, Action.BLOCK)
            else "Preceding tool in same message was blocked"
        )
        _blocked_tool_ids[tool_event.tool_use_id] = reason
        _blocked_tool_info[tool_event.tool_use_id] = {
            "name": tool_event.name,
            "input": tool_event.input,
        }
        _blocked_tool_timestamps[tool_event.tool_use_id] = time.time()
        if cascade_on_block:
            halted = True
        else:
            halted = any_halted or result.action == Action.HALT_SESSION
        logger.warning(f"BLOCKED tool {tool_event.name} (id={tool_event.tool_use_id}): {reason}")
        return True, halted, reason

    if result.action == Action.ALERT:
        logger.warning(f"ALERT on tool {tool_event.name} (id={tool_event.tool_use_id}): {result.reason}")

    return False, any_halted, ""


async def _gate_tool_events(
    tool_events: list[ToolUseEvent],
    gate_chain: GateChain,
    _publish,
    user_intent: str,
    model: str,
    conversation_context: list,
    cascade_on_block: bool = True,
) -> set[str]:
    """Gate a batch of tool events, returning the set of blocked tool_use_ids.

    Used by both _gate_sse_response and _gate_non_streaming_response to avoid
    duplicating the per-tool evaluation + escalation + blocking logic.

    Args:
        cascade_on_block: If True, BLOCK cascades to subsequent tools.
    """
    blocked_ids: set[str] = set()
    any_halted = False
    for i, tool_event in enumerate(tool_events):
        result, _ = await _evaluate_tool_event(
            tool_event,
            gate_chain,
            _publish,
            user_intent,
            model,
            conversation_context,
            i,
        )
        blocked, any_halted, _reason = await _process_gate_result(
            result, tool_event, _publish, any_halted, cascade_on_block=cascade_on_block
        )
        if blocked:
            blocked_ids.add(tool_event.tool_use_id)
    return blocked_ids


def _filter_blocked_sse_lines(lines: list[str], blocked_ids: set[str]) -> list[str]:
    """Remove SSE lines referencing blocked tool call IDs from Responses API output.

    Filters out:
    - response.output_item.added/done with a blocked call_id
    - response.function_call_arguments.delta/done for blocked items
    - response.completed output entries for blocked items
    """
    filtered_lines: list[str] = []
    for line in lines:
        if line.startswith("event: "):
            filtered_lines.append(line)
            continue

        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                event_type = data.get("type", "")

                # Check if this event references a blocked tool call
                should_skip = False
                if event_type in (
                    "response.output_item.added",
                    "response.output_item.done",
                ):
                    item = data.get("item", {})
                    call_id = item.get("call_id", "")
                    if call_id in blocked_ids:
                        should_skip = True
                elif event_type in (
                    "response.function_call_arguments.delta",
                    "response.function_call_arguments.done",
                ):
                    item_id = data.get("item_id", "")
                    if item_id in blocked_ids:
                        should_skip = True

                if should_skip:
                    # Remove the preceding event: line too
                    if filtered_lines and filtered_lines[-1].startswith("event: "):
                        filtered_lines.pop()
                    continue
            except (json.JSONDecodeError, TypeError):
                pass

        # Update response.completed to remove blocked items from output
        if line.startswith("data: ") and '"response.completed"' in line:
            try:
                data = json.loads(line[6:])
                resp = data.get("response", {})
                output = resp.get("output", [])
                resp["output"] = [
                    item
                    for item in output
                    if item.get("type") != "function_call" or item.get("call_id", "") not in blocked_ids
                ]
                if not any(item.get("type") == "function_call" for item in resp["output"]):
                    resp["status"] = "completed"
                data["response"] = resp
                line = f"data: {json.dumps(data)}"
            except (json.JSONDecodeError, TypeError):
                pass

        filtered_lines.append(line)
    return filtered_lines


async def _gate_sse_response(
    content: bytes,
    request_body: bytes,
    gate_chain: GateChain,
    event_bus: EventBus,
    session_id: str = "",
    agent_label: str = "",
    api_format: str = API_FORMAT_RESPONSES,
    user_intent: str = "",
    model: str = "",
    conversation_context: list | None = None,
) -> bytes:
    """Parse SSE content (from ChatGPT backend), gate tool calls, and suppress blocked ones.

    The ChatGPT backend returns SSE-formatted responses even for non-streaming
    requests. We parse them to extract tool use events, run gate evaluation,
    and rebuild the SSE content with blocked function_call items removed.

    Since we have the full response before Codex sees it, we can suppress
    blocked tool calls pre-execution — Codex never receives them.
    """
    from cross.stream_formats import get_format

    fmt = get_format(api_format)
    parser = fmt.create_parser()

    async def _publish(ev):
        if hasattr(ev, "session_id") and not ev.session_id:
            ev.session_id = session_id
        if hasattr(ev, "agent") and not ev.agent:
            ev.agent = agent_label
        await event_bus.publish(ev)

    # Use pre-extracted context if available; fall back to parsing request body
    if not user_intent and not model:
        try:
            req_data = json.loads(request_body)
            model = req_data.get("model", "")
            user_intent = _extract_user_intent(req_data)
            conversation_context = _extract_conversation_context(req_data)
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
            pass
    if conversation_context is None:
        conversation_context = []

    # First pass: parse SSE, collect tool events
    lines = content.decode("utf-8", errors="replace").split("\n")
    tool_events: list[ToolUseEvent] = []
    for line in lines:
        events = parser.feed_line(line)
        for ev in events:
            await _publish(ev)
            if isinstance(ev, ToolUseEvent):
                tool_events.append(ev)
    for ev in parser.feed_line(""):
        await _publish(ev)
        if isinstance(ev, ToolUseEvent):
            tool_events.append(ev)

    if not tool_events:
        return content

    # Gate each tool call using the shared helper
    blocked_ids = await _gate_tool_events(
        tool_events,
        gate_chain,
        _publish,
        user_intent,
        model,
        conversation_context,
    )

    if not blocked_ids:
        return content

    # Rebuild SSE content with blocked tool events removed
    filtered_lines = _filter_blocked_sse_lines(lines, blocked_ids)
    return "\n".join(filtered_lines).encode("utf-8")


async def _gate_non_streaming_response(
    content: bytes,
    request_body: bytes,
    gate_chain: GateChain,
    event_bus: EventBus,
    session_id: str = "",
    agent_label: str = "",
    api_format: str = API_FORMAT_ANTHROPIC,
    user_intent: str = "",
    model: str = "",
    conversation_context: list | None = None,
) -> bytes:
    """Run gate evaluation on tool_use blocks in a non-streaming response."""
    data = json.loads(content)

    # Extract tool call blocks based on API format
    if api_format == API_FORMAT_RESPONSES:
        # Responses API: output array with type: "function_call" items
        blocks = data.get("output", [])
    else:
        # Anthropic: content array with type: "tool_use" items
        blocks = data.get("content", [])

    # Use pre-extracted context if available; fall back to parsing request body
    if not user_intent and not model:
        try:
            req_data = json.loads(request_body)
            model = req_data.get("model", "")
            user_intent = _extract_user_intent(req_data)
            conversation_context = _extract_conversation_context(req_data)
        except (json.JSONDecodeError, KeyError):
            pass
    if conversation_context is None:
        conversation_context = []

    async def _publish(ev):
        if hasattr(ev, "session_id") and not ev.session_id:
            ev.session_id = session_id
        if hasattr(ev, "agent") and not ev.agent:
            ev.agent = agent_label
        await event_bus.publish(ev)

    any_halted = False
    halted_indices: list[int] = []

    for i, block in enumerate(blocks):
        block_type = block.get("type", "")
        if api_format == API_FORMAT_RESPONSES:
            if block_type != "function_call":
                continue
            tool_id = block.get("call_id", block.get("id", ""))
            tool_name = block.get("name", "")
            args = block.get("arguments", "")
            try:
                tool_input = json.loads(args) if isinstance(args, str) and args else args
            except json.JSONDecodeError:
                tool_input = {"_raw": args}
        else:
            if block_type != "tool_use":
                continue
            tool_id = block.get("id", "")
            tool_name = block.get("name", "")
            tool_input = block.get("input")

        # Resolve script contents for Bash/exec tool calls
        proxy_cwd = os.getcwd()
        script_contents = _resolve_scripts_for_tool(tool_name, tool_input, cwd=proxy_cwd)

        gate_request = GateRequest(
            tool_use_id=tool_id,
            tool_name=tool_name,
            tool_input=tool_input,
            timestamp=time.time(),
            user_intent=user_intent,
            agent=model,
            tool_index_in_message=i,
            recent_tools=list(_recent_tools),
            script_contents=script_contents,
            cwd=proxy_cwd,
            conversation_context=conversation_context,
        )
        result = await gate_chain.evaluate(gate_request)

        # Record for future context
        _recent_tools.append({"name": tool_name, "input": tool_input})

        # Publish gate decision event
        await _publish(
            GateDecisionEvent(
                tool_use_id=tool_id,
                tool_name=tool_name,
                action=result.action.name.lower(),
                reason=result.reason,
                rule_id=result.rule_id,
                evaluator=result.evaluator,
                confidence=result.confidence,
                tool_input=tool_input,
                script_contents=script_contents or None,
                recent_tools=list(_recent_tools),
                user_intent=user_intent,
                conversation_context=conversation_context,
                eval_system_prompt=result.eval_system_prompt,
                eval_user_message=result.eval_user_message,
                eval_response_text=result.eval_response_text,
            )
        )

        # Build a ToolUseEvent for the shared gate result processor
        tool_event = ToolUseEvent(
            tool_use_id=tool_id,
            name=tool_name,
            input=tool_input,
        )

        blocked, any_halted, _reason = await _process_gate_result(
            result, tool_event, _publish, any_halted, cascade_on_block=True
        )
        if blocked:
            halted_indices.append(i)

    if halted_indices:
        # Remove blocked tool call blocks from response (reverse order to preserve indices)
        for i in reversed(halted_indices):
            blocks.pop(i)
        if api_format == API_FORMAT_RESPONSES:
            data["output"] = blocks
            # Fix status if no function_call items remain
            if not any(b.get("type") == "function_call" for b in blocks):
                data["status"] = "completed"
        else:
            data["content"] = blocks
            # Fix stop_reason if no tool_use blocks remain
            if not any(b.get("type") == "tool_use" for b in blocks):
                data["stop_reason"] = "end_turn"
        return json.dumps(data).encode()
    return content


def _is_tool_use_block_start(line: str) -> bool:
    """Check if an SSE data line is a content_block_start for a tool_use block."""
    return get_format(API_FORMAT_ANTHROPIC).is_tool_start(line)


def _synthetic_message_end_sse() -> list[str]:
    """Generate SSE lines to cleanly end a response (end_turn + message_stop)."""
    return get_format(API_FORMAT_ANTHROPIC).synthetic_stop_sse()


def _synthetic_openai_stop_sse(chunk_id: str = "", model: str = "") -> list[str]:
    """Generate SSE lines to cleanly end an OpenAI response (finish_reason: stop)."""
    return get_format(API_FORMAT_OPENAI).synthetic_stop_sse(chunk_id=chunk_id, model=model)


def _is_openai_tool_call_start(line: str) -> bool:
    """Check if an SSE data line contains the first tool_calls delta."""
    return get_format(API_FORMAT_OPENAI).is_tool_start(line)


async def _evaluate_tool_event(
    tool_event: ToolUseEvent,
    gate_chain: GateChain,
    _publish,
    user_intent: str,
    model: str,
    conversation_context: list[dict[str, str]],
    tool_index: int,
):
    """Evaluate a single tool event through the gate chain.

    Returns (result, script_contents) where result is the EvaluationResponse.
    """
    proxy_cwd = os.getcwd()
    script_contents = _resolve_scripts_for_tool(tool_event.name, tool_event.input, cwd=proxy_cwd)
    tool_event.script_contents = script_contents or None

    gate_request = GateRequest(
        tool_use_id=tool_event.tool_use_id,
        tool_name=tool_event.name,
        tool_input=tool_event.input,
        timestamp=time.time(),
        user_intent=user_intent,
        agent=model,
        tool_index_in_message=tool_index,
        recent_tools=list(_recent_tools),
        script_contents=script_contents,
        cwd=proxy_cwd,
        conversation_context=conversation_context,
    )
    result = await gate_chain.evaluate(gate_request)

    # Record for future context
    _recent_tools.append({"name": tool_event.name, "input": tool_event.input})

    # Publish gate decision
    await _publish(
        GateDecisionEvent(
            tool_use_id=tool_event.tool_use_id,
            tool_name=tool_event.name,
            action=result.action.name.lower(),
            reason=result.reason,
            rule_id=result.rule_id,
            evaluator=result.evaluator,
            confidence=result.confidence,
            tool_input=tool_event.input,
            script_contents=script_contents or None,
            recent_tools=list(_recent_tools),
            user_intent=user_intent,
            conversation_context=conversation_context,
            eval_system_prompt=result.eval_system_prompt,
            eval_user_message=result.eval_user_message,
            eval_response_text=result.eval_response_text,
        )
    )

    return result, script_contents


async def _handle_escalation(tool_event, _publish, any_halted):
    """Handle ESCALATE/REVIEW action — wait for human approval.

    Returns (approved, username, reason_if_denied).
    """
    if any_halted:
        return False, "", "Preceding tool in same message was halted"

    approval_event = asyncio.Event()
    _pending_approvals[tool_event.tool_use_id] = approval_event
    logger.info(
        f"ESCALATED tool {tool_event.name} ({tool_event.tool_use_id}), "
        f"waiting for human approval (timeout={settings.gate_approval_timeout}s)"
    )

    try:
        await asyncio.wait_for(
            approval_event.wait(),
            timeout=settings.gate_approval_timeout,
        )
        approved, username = _approval_results.pop(tool_event.tool_use_id, (False, ""))
    except asyncio.TimeoutError:
        approved, username = False, ""
        logger.warning(f"Gate approval timed out for {tool_event.name} ({tool_event.tool_use_id})")
    finally:
        _pending_approvals.pop(tool_event.tool_use_id, None)

    if approved:
        logger.info(f"APPROVED tool {tool_event.name} by {username}")
        await _publish(
            GateDecisionEvent(
                tool_use_id=tool_event.tool_use_id,
                tool_name=tool_event.name,
                action="allow",
                reason=f"Approved by human reviewer (@{username})",
                evaluator="human",
                tool_input=tool_event.input,
            )
        )
        return True, username, ""
    else:
        reason = f"Denied by human reviewer (@{username})" if username else "Timed out waiting for human approval"
        logger.warning(f"DENIED tool {tool_event.name} ({tool_event.tool_use_id}): {reason}")
        await _publish(
            GateDecisionEvent(
                tool_use_id=tool_event.tool_use_id,
                tool_name=tool_event.name,
                action="block",
                reason=reason,
                evaluator="human",
                tool_input=tool_event.input,
            )
        )
        return False, username, reason


async def _proxy_streaming(
    client: httpx.AsyncClient,
    event_bus: EventBus,
    method: str,
    path: str,
    headers: dict,
    body: bytes,
    gate_chain: GateChain | None = None,
    session_id: str = "",
    agent_label: str = "",
    api_format: str = API_FORMAT_ANTHROPIC,
    body_for_parsing: bytes | None = None,
    user_intent: str = "",
    model: str = "",
    conversation_context: list | None = None,
) -> Response:
    """Unified streaming proxy for both Anthropic and OpenAI formats.

    Buffering differences:
    - Anthropic: buffers per-tool (content_block_start to content_block_stop),
      gates each tool individually as it completes.
    - OpenAI: buffers from first tool_call delta to finish_reason, then gates
      all tool calls as a batch.

    The gate evaluation logic is shared for both formats.
    """
    fmt = get_format(api_format)

    upstream = await client.send(
        client.build_request(method, path, headers=headers, content=body),
        stream=True,
    )

    parser = fmt.create_parser()

    # Use pre-extracted context if available; fall back to parsing request body
    if not user_intent and not model:
        try:
            req_data = json.loads(body_for_parsing or body)
            model = req_data.get("model", "")
            user_intent = _extract_user_intent(req_data)
            conversation_context = _extract_conversation_context(req_data)
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
            pass
    if conversation_context is None:
        conversation_context = []

    async def _publish(ev):
        """Stamp session identity on events before publishing."""
        if hasattr(ev, "session_id") and not ev.session_id:
            ev.session_id = session_id
        if hasattr(ev, "agent") and not ev.agent:
            ev.agent = agent_label
        await event_bus.publish(ev)

    async def generate():
        current_body = body
        retries_remaining = settings.gate_max_retries
        next_content_index = 0  # Track content blocks sent to client (Anthropic only)
        is_first_attempt = True
        current_upstream = upstream
        current_parser = parser
        # OpenAI: track chunk metadata for synthetic responses
        chunk_id = ""
        chunk_model = model

        while True:
            buffer: list[str] = []
            buffering = False
            any_halted = False
            response_terminated = False
            tool_index = 0
            # Anthropic: holds event: line until we see next data: line
            pending_line: str | None = None
            should_retry = False
            blocked_tools_for_retry: list[dict] = []

            try:
                async for line in current_upstream.aiter_lines():
                    events = current_parser.feed_line(line)

                    if gate_chain is None:
                        # No gating — pass through
                        for ev in events:
                            await _publish(ev)
                        yield line + "\n"
                        continue

                    # --- Gated streaming ---

                    # For retry attempts, skip/rewrite format-specific lines
                    if not is_first_attempt:
                        if fmt.should_skip_on_retry(line):
                            for ev in events:
                                await _publish(ev)
                            continue
                        line = fmt.rewrite_line_for_retry(line, next_content_index)

                    # OpenAI: track chunk metadata for synthetic responses
                    if not fmt.uses_event_lines and line.startswith("data: ") and not chunk_id:
                        try:
                            d = json.loads(line[6:])
                            chunk_id = d.get("id", "")
                            chunk_model = d.get("model", model)
                        except (json.JSONDecodeError, TypeError):
                            pass

                    # Detect tool start — begin buffering
                    if not buffering and fmt.is_tool_start(line):
                        buffering = True
                        buffer = []
                        if pending_line is not None:
                            buffer.append(pending_line)
                            pending_line = None
                        buffer.append(line)
                        for ev in events:
                            await _publish(ev)
                        continue

                    # When buffering, ALL lines go into the buffer
                    if buffering:
                        if pending_line is not None:
                            buffer.append(pending_line)
                            pending_line = None
                        buffer.append(line)

                        # Safety: flush buffer if it grows too large
                        if len(buffer) > _MAX_BUFFER_LINES:
                            logger.warning(
                                f"Buffer exceeded {_MAX_BUFFER_LINES} lines, flushing without gate evaluation"
                            )
                            for ev in events:
                                await _publish(ev)
                            for buffered_line in buffer:
                                yield buffered_line + "\n"
                            buffer = []
                            buffering = False
                            continue

                        # Collect completed tool events
                        tool_events = [ev for ev in events if isinstance(ev, ToolUseEvent)]
                        for ev in events:
                            if not isinstance(ev, ToolUseEvent):
                                await _publish(ev)

                        if tool_events:
                            # Gate each completed tool event
                            all_allowed = True

                            for tool_event in tool_events:
                                result, script_contents = await _evaluate_tool_event(
                                    tool_event,
                                    gate_chain,
                                    _publish,
                                    user_intent,
                                    model,
                                    conversation_context,
                                    tool_index,
                                )
                                tool_index += 1

                                if result.action in (Action.REVIEW, Action.ESCALATE):
                                    approved, username, reason = await _handle_escalation(
                                        tool_event,
                                        _publish,
                                        any_halted,
                                    )
                                    if approved:
                                        await _publish(tool_event)
                                        if fmt.uses_event_lines:
                                            # Anthropic: flush buffer per tool
                                            for buffered_line in buffer:
                                                yield buffered_line + "\n"
                                            next_content_index += 1
                                    else:
                                        await _publish(tool_event)
                                        blocked_tools_for_retry.append(
                                            {
                                                "tool_use_id": tool_event.tool_use_id,
                                                "name": tool_event.name,
                                                "input": tool_event.input,
                                                "reason": reason,
                                            }
                                        )
                                        if fmt.uses_event_lines:
                                            # Anthropic: break immediately on denial
                                            should_retry = True
                                            break
                                        else:
                                            all_allowed = False

                                elif result.action == Action.HALT_SESSION or any_halted:
                                    reason = (
                                        result.reason
                                        if result.action == Action.HALT_SESSION
                                        else "Preceding tool in same message was halted"
                                    )
                                    _blocked_tool_ids[tool_event.tool_use_id] = reason
                                    _blocked_tool_info[tool_event.tool_use_id] = {
                                        "name": tool_event.name,
                                        "input": tool_event.input,
                                    }
                                    _blocked_tool_timestamps[tool_event.tool_use_id] = time.time()
                                    logger.warning(
                                        f"HALTED tool {tool_event.name} (id={tool_event.tool_use_id}): {reason}"
                                    )
                                    any_halted = True
                                    all_allowed = False
                                    await _publish(tool_event)
                                    for sse_line in fmt.synthetic_stop_sse(
                                        chunk_id=chunk_id,
                                        model=chunk_model,
                                    ):
                                        yield sse_line + "\n"
                                    response_terminated = True
                                    break

                                elif result.action == Action.BLOCK:
                                    reason = result.reason
                                    logger.warning(
                                        f"BLOCKED tool {tool_event.name} (id={tool_event.tool_use_id}): {reason}"
                                    )
                                    all_allowed = False
                                    await _publish(tool_event)
                                    blocked_tools_for_retry.append(
                                        {
                                            "tool_use_id": tool_event.tool_use_id,
                                            "name": tool_event.name,
                                            "input": tool_event.input,
                                            "reason": reason,
                                        }
                                    )
                                    if fmt.uses_event_lines:
                                        # Anthropic: break immediately on block
                                        should_retry = True
                                        break

                                elif result.action == Action.ALERT:
                                    logger.warning(
                                        f"ALERT on tool {tool_event.name} "
                                        f"(id={tool_event.tool_use_id}): {result.reason}"
                                    )
                                    await _publish(tool_event)
                                    if fmt.uses_event_lines:
                                        for buffered_line in buffer:
                                            yield buffered_line + "\n"
                                        next_content_index += 1
                                else:
                                    # Allow: publish event
                                    await _publish(tool_event)
                                    if fmt.uses_event_lines:
                                        for buffered_line in buffer:
                                            yield buffered_line + "\n"
                                        next_content_index += 1

                            # After processing all tool events in this batch
                            if response_terminated or should_retry:
                                if response_terminated or should_retry:
                                    break

                            if not fmt.uses_event_lines:
                                # OpenAI: check batch results after loop
                                if blocked_tools_for_retry:
                                    should_retry = True
                                    break
                                if all_allowed:
                                    for buf_line in buffer:
                                        yield buf_line + "\n"

                            buffer = []
                            buffering = False
                        # else: still accumulating tool deltas
                        continue

                    # --- Not buffering ---

                    # Track content blocks sent to client (Anthropic only)
                    if fmt.is_content_block_stop(line):
                        next_content_index += 1

                    # Anthropic: hold event: lines until we see next data: line
                    if fmt.uses_event_lines:
                        if pending_line is not None:
                            yield pending_line + "\n"
                            pending_line = None

                        if line.startswith("event: "):
                            pending_line = line
                            for ev in events:
                                await _publish(ev)
                            continue

                    # Stream through immediately
                    for ev in events:
                        await _publish(ev)
                    yield line + "\n"

                if not response_terminated and not should_retry:
                    if pending_line is not None:
                        yield pending_line + "\n"
                    events = current_parser.feed_line("")
                    for ev in events:
                        await _publish(ev)
            finally:
                await current_upstream.aclose()

            # Retry logic
            if should_retry and retries_remaining > 0:
                retries_remaining -= 1
                retry_number = settings.gate_max_retries - retries_remaining
                for bt in blocked_tools_for_retry:
                    await _publish(
                        GateRetryEvent(
                            tool_use_id=bt["tool_use_id"],
                            tool_name=bt["name"],
                            reason=bt["reason"],
                            retry_number=retry_number,
                            max_retries=settings.gate_max_retries,
                        )
                    )
                logger.info(f"Retrying after BLOCK (attempt {retry_number}/{settings.gate_max_retries})")
                current_body = _build_retry_request_body(
                    current_body,
                    blocked_tools_for_retry,
                    api_format=api_format,
                )
                headers["content-length"] = str(len(current_body))
                is_first_attempt = False
                current_parser = fmt.create_parser()
                current_upstream = await client.send(
                    client.build_request(method, path, headers=headers, content=current_body),
                    stream=True,
                )
                continue
            elif should_retry:
                # Retry budget exhausted — fall back to HALT_SESSION behavior
                logger.warning("Retry budget exhausted, halting session")
                for bt in blocked_tools_for_retry:
                    _blocked_tool_ids[bt["tool_use_id"]] = bt["reason"]
                    _blocked_tool_info[bt["tool_use_id"]] = {
                        "name": bt["name"],
                        "input": bt["input"],
                    }
                    _blocked_tool_timestamps[bt["tool_use_id"]] = time.time()
                for sse_line in fmt.synthetic_stop_sse(
                    chunk_id=chunk_id,
                    model=chunk_model,
                ):
                    yield sse_line + "\n"
                break
            else:
                # Normal completion or HALT_SESSION
                break

    resp_headers = dict(upstream.headers)
    resp_headers.pop("transfer-encoding", None)
    resp_headers.pop("content-encoding", None)

    return StreamingResponse(
        generate(),
        status_code=upstream.status_code,
        headers=resp_headers,
    )


async def shutdown():
    if _client:
        await _client.aclose()
    if _openai_client:
        await _openai_client.aclose()
    if _chatgpt_client:
        await _chatgpt_client.aclose()
