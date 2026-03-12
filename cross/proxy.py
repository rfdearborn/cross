"""Core reverse proxy — forwards requests to Anthropic API and parses SSE responses.

When a GateChain is configured, tool_use blocks in streaming responses are buffered
until the gate evaluates them. Text blocks stream through with zero added latency.
Blocked tools are suppressed; the next client request gets a synthetic error tool_result.
"""

from __future__ import annotations

import json
import logging
import time
from collections import deque
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from cross.chain import GateChain
from cross.config import settings
from cross.evaluator import Action, GateRequest
from cross.events import ErrorEvent, EventBus, GateDecisionEvent, RequestEvent, ToolUseEvent
from cross.sse import SSEParser

logger = logging.getLogger("cross.proxy")

_client: httpx.AsyncClient | None = None

# Blocked tool_use_ids -> (reason, timestamp) for injecting error tool_results on next request
_blocked_tool_ids: dict[str, str] = {}
_blocked_tool_timestamps: dict[str, float] = {}

# Recent tool calls for LLM gate context (bounded deque)
_recent_tools: deque[dict[str, Any]] = deque(maxlen=max(settings.llm_gate_context_tools, 1))

# Safety limits for SSE buffering
_MAX_BUFFER_LINES = 500  # Flush buffer if it exceeds this many lines
_BLOCKED_TOOL_TTL = 300.0  # Seconds before stale blocked tool entries are cleaned up


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=settings.anthropic_base_url,
            timeout=httpx.Timeout(connect=10, read=300, write=30, pool=10),
        )
    return _client


def _extract_user_intent(req_data: dict) -> str:
    """Extract the last user text from a parsed request body, skipping system-reminders."""
    msgs = req_data.get("messages", [])
    if not msgs:
        return ""
    last = msgs[-1]
    content = last.get("content", "")
    if isinstance(content, str):
        return content[:500]
    if isinstance(content, list):
        for b in content:
            if b.get("type") == "text":
                text = b.get("text", "")
                if text and not text.startswith("<system-reminder>"):
                    return text[:500]
    return ""


def _extract_request_event(method: str, path: str, body: bytes | None) -> RequestEvent:
    event = RequestEvent(method=method, path=path)
    if body:
        try:
            data = json.loads(body)
            event.model = data.get("model")
            event.stream = data.get("stream", False)
            event.raw_body = data

            msgs = data.get("messages", [])
            event.messages_count = len(msgs)

            tools = data.get("tools", [])
            event.tool_names = [t.get("name", "?") for t in tools]

            if msgs:
                last = msgs[-1]
                event.last_message_role = last.get("role")
                intent = _extract_user_intent(data)
                if intent:
                    event.last_message_preview = intent[:200]
                elif isinstance(last.get("content"), list):
                    types = [b.get("type", "?") for b in last["content"]]
                    event.last_message_preview = f"[{', '.join(types)}]"
        except (json.JSONDecodeError, KeyError):
            pass
    return event


def _intercept_blocked_tool_results(body: bytes) -> bytes:
    """Replace tool_results for blocked tools with synthetic error results."""
    # Clean up stale entries
    if _blocked_tool_timestamps:
        now = time.time()
        stale = [k for k, t in _blocked_tool_timestamps.items() if now - t > _BLOCKED_TOOL_TTL]
        for k in stale:
            _blocked_tool_ids.pop(k, None)
            _blocked_tool_timestamps.pop(k, None)

    if not _blocked_tool_ids:
        return body

    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body

    messages = data.get("messages", [])
    modified = False

    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue

        for i, block in enumerate(content):
            if block.get("type") != "tool_result":
                continue
            tool_use_id = block.get("tool_use_id", "")
            if tool_use_id in _blocked_tool_ids:
                reason = _blocked_tool_ids.pop(tool_use_id)
                _blocked_tool_timestamps.pop(tool_use_id, None)
                content[i] = {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": f"[Cross blocked this tool call: {reason}]",
                    "is_error": True,
                }
                modified = True
                logger.info(f"Injected error tool_result for blocked {tool_use_id}")

    if modified:
        return json.dumps(data).encode()
    return body


async def handle_proxy_request(
    request: Request,
    event_bus: EventBus,
    gate_chain: GateChain | None = None,
) -> Response:
    """Handle a proxy request, publishing events to the given EventBus."""
    body = await request.body()
    path = request.url.path
    if request.url.query:
        path = f"{path}?{request.url.query}"

    # Intercept blocked tool_results before forwarding
    body = _intercept_blocked_tool_results(body)

    # Build upstream headers — forward everything except host
    headers = dict(request.headers)
    headers.pop("host", None)
    headers["host"] = "api.anthropic.com"
    # Fix content-length after potential body modification
    headers["content-length"] = str(len(body))

    # Publish request event
    req_event = _extract_request_event(request.method, path, body)
    await event_bus.publish(req_event)

    client = get_client()

    # Check if this is a streaming request
    is_streaming = False
    if body:
        try:
            is_streaming = json.loads(body).get("stream", False)
        except json.JSONDecodeError:
            pass

    if is_streaming:
        return await _proxy_streaming(client, event_bus, request.method, path, headers, body, gate_chain)
    else:
        return await _proxy_simple(client, event_bus, request.method, path, headers, body, gate_chain)


async def _proxy_simple(
    client: httpx.AsyncClient,
    event_bus: EventBus,
    method: str,
    path: str,
    headers: dict,
    body: bytes,
    gate_chain: GateChain | None = None,
) -> Response:
    resp = await client.request(method, path, headers=headers, content=body)

    if resp.status_code >= 400:
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
            content = await _gate_non_streaming_response(content, body, gate_chain, event_bus)
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


async def _gate_non_streaming_response(
    content: bytes,
    request_body: bytes,
    gate_chain: GateChain,
    event_bus: EventBus,
) -> bytes:
    """Run gate evaluation on tool_use blocks in a non-streaming response."""
    data = json.loads(content)
    blocks = data.get("content", [])
    user_intent = ""
    model = ""
    try:
        req_data = json.loads(request_body)
        model = req_data.get("model", "")
        user_intent = _extract_user_intent(req_data)
    except (json.JSONDecodeError, KeyError):
        pass

    any_blocked = False
    blocked_indices: list[int] = []

    for i, block in enumerate(blocks):
        if block.get("type") != "tool_use":
            continue

        gate_request = GateRequest(
            tool_use_id=block.get("id", ""),
            tool_name=block.get("name", ""),
            tool_input=block.get("input"),
            timestamp=time.time(),
            user_intent=user_intent,
            agent=model,
            tool_index_in_message=i,
            recent_tools=list(_recent_tools),
        )
        result = await gate_chain.evaluate(gate_request)

        # Record for future context
        _recent_tools.append({"name": block.get("name", ""), "input": block.get("input")})

        if result.action in (Action.BLOCK, Action.ESCALATE) or any_blocked:
            reason = (
                result.reason
                if result.action in (Action.BLOCK, Action.ESCALATE)
                else ("Preceding tool in same message was blocked")
            )
            tool_id = block.get("id", "")
            _blocked_tool_ids[tool_id] = reason
            _blocked_tool_timestamps[tool_id] = time.time()
            any_blocked = True
            blocked_indices.append(i)
            logger.warning(f"BLOCKED tool {block.get('name')} (id={block.get('id')}): {reason}")

    if blocked_indices:
        # Remove blocked tool_use blocks from response (reverse order to preserve indices)
        for i in reversed(blocked_indices):
            blocks.pop(i)
        data["content"] = blocks
        return json.dumps(data).encode()
    return content


def _is_tool_use_block_start(line: str) -> bool:
    """Check if an SSE data line is a content_block_start for a tool_use block."""
    if not line.startswith("data: "):
        return False
    try:
        data = json.loads(line[6:])
        return data.get("type") == "content_block_start" and data.get("content_block", {}).get("type") == "tool_use"
    except (json.JSONDecodeError, AttributeError):
        return False


async def _proxy_streaming(
    client: httpx.AsyncClient,
    event_bus: EventBus,
    method: str,
    path: str,
    headers: dict,
    body: bytes,
    gate_chain: GateChain | None = None,
) -> Response:
    upstream = await client.send(
        client.build_request(method, path, headers=headers, content=body),
        stream=True,
    )

    parser = SSEParser()

    # Extract gate context from the request body
    user_intent = ""
    model = ""
    try:
        req_data = json.loads(body)
        model = req_data.get("model", "")
        user_intent = _extract_user_intent(req_data)
    except (json.JSONDecodeError, KeyError):
        pass

    async def generate():
        buffer: list[str] = []
        buffering = False  # True when inside a tool_use content block
        any_blocked = False  # If any tool in this message was blocked, block the rest
        tool_index = 0  # Track tool position in this response
        pending_line: str | None = None  # Holds event: line until we know if next data: starts a buffer

        try:
            async for line in upstream.aiter_lines():
                events = parser.feed_line(line)

                if gate_chain is None:
                    # No gating — pass through as before
                    for ev in events:
                        await event_bus.publish(ev)
                    yield line + "\n"
                    continue

                # --- Gated streaming ---

                # Detect tool_use block start — pull pending event: line into buffer too
                if not buffering and _is_tool_use_block_start(line):
                    buffering = True
                    buffer = []
                    if pending_line is not None:
                        buffer.append(pending_line)
                        pending_line = None
                    buffer.append(line)
                    for ev in events:
                        await event_bus.publish(ev)
                    continue

                # Flush any pending line that wasn't pulled into a buffer
                if pending_line is not None:
                    yield pending_line + "\n"
                    pending_line = None

                # Hold event: lines until we see the next data: line
                if line.startswith("event: "):
                    pending_line = line
                    for ev in events:
                        await event_bus.publish(ev)
                    continue

                if buffering:
                    buffer.append(line)

                    # Safety: flush buffer if it grows too large
                    if len(buffer) > _MAX_BUFFER_LINES:
                        logger.warning(f"Buffer exceeded {_MAX_BUFFER_LINES} lines, flushing without gate evaluation")
                        for ev in events:
                            await event_bus.publish(ev)
                        for buffered_line in buffer:
                            yield buffered_line + "\n"
                        buffer = []
                        buffering = False
                        continue

                    # Check for tool_use completion
                    tool_event = None
                    for ev in events:
                        if isinstance(ev, ToolUseEvent):
                            tool_event = ev
                        else:
                            await event_bus.publish(ev)

                    if tool_event:
                        # Tool_use block complete — run gate evaluation
                        gate_request = GateRequest(
                            tool_use_id=tool_event.tool_use_id,
                            tool_name=tool_event.name,
                            tool_input=tool_event.input,
                            timestamp=time.time(),
                            user_intent=user_intent,
                            agent=model,
                            tool_index_in_message=tool_index,
                            recent_tools=list(_recent_tools),
                        )
                        tool_index += 1
                        result = await gate_chain.evaluate(gate_request)

                        # Record for future context
                        _recent_tools.append({"name": tool_event.name, "input": tool_event.input})

                        # Publish gate decision (include tool_input so sentinel can see blocked calls)
                        await event_bus.publish(
                            GateDecisionEvent(
                                tool_use_id=tool_event.tool_use_id,
                                tool_name=tool_event.name,
                                action=result.action.name.lower(),
                                reason=result.reason,
                                rule_id=result.rule_id,
                                evaluator=result.evaluator,
                                confidence=result.confidence,
                                tool_input=tool_event.input,
                            )
                        )

                        if result.action in (Action.BLOCK, Action.ESCALATE) or any_blocked:
                            # Block (or escalate — treated as block until escalation path exists)
                            reason = (
                                result.reason
                                if result.action in (Action.BLOCK, Action.ESCALATE)
                                else ("Preceding tool in same message was blocked")
                            )
                            _blocked_tool_ids[tool_event.tool_use_id] = reason
                            _blocked_tool_timestamps[tool_event.tool_use_id] = time.time()
                            logger.warning(
                                f"BLOCKED tool {tool_event.name} (id={tool_event.tool_use_id}): {result.reason}"
                            )
                            any_blocked = True
                            # Don't publish the ToolUseEvent
                        elif result.action == Action.ALERT:
                            # Alert: flush buffer (allow execution) but log
                            logger.warning(
                                f"ALERT on tool {tool_event.name} (id={tool_event.tool_use_id}): {result.reason}"
                            )
                            await event_bus.publish(tool_event)
                            for buffered_line in buffer:
                                yield buffered_line + "\n"
                        else:
                            # Allow: flush buffer
                            await event_bus.publish(tool_event)
                            for buffered_line in buffer:
                                yield buffered_line + "\n"

                        buffer = []
                        buffering = False
                    # else: still accumulating tool_use deltas
                    continue

                # Not buffering — stream text and other events through immediately
                for ev in events:
                    await event_bus.publish(ev)
                yield line + "\n"

            # Flush any pending event line
            if pending_line is not None:
                yield pending_line + "\n"

            # Flush any remaining parser state
            events = parser.feed_line("")
            for ev in events:
                await event_bus.publish(ev)
        finally:
            await upstream.aclose()

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
