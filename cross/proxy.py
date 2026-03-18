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
from typing import Any

import httpx
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from cross.chain import GateChain
from cross.config import settings
from cross.evaluator import Action, GateRequest
from cross.events import ErrorEvent, EventBus, GateDecisionEvent, GateRetryEvent, RequestEvent, ToolUseEvent
from cross.script_resolver import resolve_script_contents
from cross.sse import SSEParser

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


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=settings.anthropic_base_url,
            timeout=httpx.Timeout(connect=10, read=300, write=30, pool=10),
        )
    return _client


_SKIP_PREFIXES = ("<system-reminder>", "[Request interrupted by user]")


def _extract_user_intent(req_data: dict) -> str:
    """Extract user text from the last user message."""
    msgs = req_data.get("messages", [])
    for msg in reversed(msgs):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            if content and not content.startswith(_SKIP_PREFIXES):
                return content[:500]
        elif isinstance(content, list):
            for b in reversed(content):
                if b.get("type") == "text":
                    text = b.get("text", "")
                    if text and not text.startswith(_SKIP_PREFIXES):
                        return text[:500]
        return ""
    return ""


def _extract_request_event(method: str, path: str, body: bytes | None) -> RequestEvent:
    from cross.daemon import get_active_agent_label

    event = RequestEvent(method=method, path=path, agent=get_active_agent_label())
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
        except (json.JSONDecodeError, KeyError):
            pass
    return event


def _inject_blocked_tool_feedback(body: bytes) -> bytes:
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

    # Build tool_use blocks (for assistant message) and error tool_results (for user message)
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
        # No assistant message — insert both before the last user message
        insert_at = len(messages)
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                insert_at = i
                break
        messages.insert(insert_at, {"role": "user", "content": tool_result_blocks})
        messages.insert(insert_at, {"role": "assistant", "content": tool_use_blocks})
    else:
        # Append tool_use blocks to the last assistant message
        content = messages[last_assistant_idx].get("content", [])
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        content.extend(tool_use_blocks)
        messages[last_assistant_idx]["content"] = content

        # Find the next user message after the assistant message
        next_user_idx = None
        for i in range(last_assistant_idx + 1, len(messages)):
            if messages[i].get("role") == "user":
                next_user_idx = i
                break

        if next_user_idx is not None:
            # Prepend tool_results to the existing user message
            user_content = messages[next_user_idx].get("content", [])
            if isinstance(user_content, str):
                user_content = [{"type": "text", "text": user_content}]
            messages[next_user_idx]["content"] = tool_result_blocks + user_content
        else:
            # No following user message — append new user message
            messages.append({"role": "user", "content": tool_result_blocks})

    data["messages"] = messages
    logger.info(f"Injected feedback for {len(to_inject)} blocked tool(s) into next request")
    return json.dumps(data).encode()


def _build_retry_request_body(original_body: bytes, blocked_tools: list[dict]) -> bytes:
    """Build a new request body with blocked tool feedback injected for retry.

    Takes the original request body and appends:
    1. An assistant message with tool_use blocks for the blocked tools
    2. A user message with error tool_result blocks (is_error: true)

    Args:
        original_body: The original request body bytes.
        blocked_tools: List of dicts with keys: tool_use_id, name, input, reason.

    Returns:
        New request body bytes with feedback injected.
    """
    data = json.loads(original_body)
    messages = data.get("messages", [])

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

    messages.append({"role": "assistant", "content": tool_use_blocks})
    messages.append({"role": "user", "content": tool_result_blocks})

    data["messages"] = messages
    return json.dumps(data).encode()


def _rewrite_content_block_index(line: str, offset: int) -> str:
    """Rewrite the index field in content_block_start/delta/stop SSE events.

    When stitching retry responses into an existing stream, content block indices
    need to be offset by how many blocks were already sent to the client.

    Args:
        line: An SSE data line (e.g., 'data: {"type":"content_block_start",...}').
        offset: The offset to add to the index field.

    Returns:
        The line with the index field rewritten, or unchanged if not applicable.
    """
    if offset == 0 or not line.startswith("data: "):
        return line
    try:
        data = json.loads(line[6:])
        if data.get("type") in ("content_block_start", "content_block_delta", "content_block_stop"):
            if "index" in data:
                data["index"] = data["index"] + offset
                return f"data: {json.dumps(data)}"
    except (json.JSONDecodeError, TypeError):
        pass
    return line


async def handle_proxy_request(
    request: Request,
    event_bus: EventBus,
    gate_chain: GateChain | None = None,
) -> Response:
    """Handle a proxy request, publishing events to the given EventBus."""
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
    path = request.url.path
    if request.url.query:
        path = f"{path}?{request.url.query}"

    # Intercept blocked tool_results before forwarding
    body = _inject_blocked_tool_feedback(body)

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

    any_halted = False
    halted_indices: list[int] = []

    for i, block in enumerate(blocks):
        if block.get("type") != "tool_use":
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
        )
        result = await gate_chain.evaluate(gate_request)

        # Record for future context
        _recent_tools.append({"name": tool_name, "input": tool_input})

        # Publish gate decision event
        await event_bus.publish(
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
            )
        )

        if result.action == Action.ESCALATE and not any_halted:
            # Hold and wait for human approval via Slack
            approval_event = asyncio.Event()
            _pending_approvals[tool_id] = approval_event
            logger.info(
                f"ESCALATED tool {tool_name} ({tool_id}), "
                f"waiting for human approval (timeout={settings.gate_approval_timeout}s)"
            )

            try:
                await asyncio.wait_for(
                    approval_event.wait(),
                    timeout=settings.gate_approval_timeout,
                )
                approved, username = _approval_results.pop(tool_id, (False, ""))
            except asyncio.TimeoutError:
                approved, username = False, ""
                logger.warning(f"Gate approval timed out for {tool_name} ({tool_id})")
            finally:
                _pending_approvals.pop(tool_id, None)

            if approved:
                logger.info(f"APPROVED tool {tool_name} by {username}")
                await event_bus.publish(
                    GateDecisionEvent(
                        tool_use_id=tool_id,
                        tool_name=tool_name,
                        action="allow",
                        reason=f"Approved by human reviewer (@{username})",
                        evaluator="human",
                        tool_input=tool_input,
                    )
                )
            else:
                reason = (
                    f"Denied by human reviewer (@{username})" if username else "Timed out waiting for human approval"
                )
                _blocked_tool_ids[tool_id] = reason
                _blocked_tool_info[tool_id] = {"name": tool_name, "input": tool_input}
                _blocked_tool_timestamps[tool_id] = time.time()
                any_halted = True
                halted_indices.append(i)
                logger.warning(f"DENIED tool {tool_name} ({tool_id}): {reason}")
                await event_bus.publish(
                    GateDecisionEvent(
                        tool_use_id=tool_id,
                        tool_name=tool_name,
                        action="halt_session",
                        reason=reason,
                        evaluator="human",
                        tool_input=tool_input,
                    )
                )

        elif result.action in (Action.HALT_SESSION, Action.BLOCK) or any_halted:
            if any_halted and result.action not in (Action.HALT_SESSION, Action.BLOCK):
                reason = "Preceding tool in same message was blocked"
            else:
                reason = result.reason
            _blocked_tool_ids[tool_id] = reason
            _blocked_tool_info[tool_id] = {"name": tool_name, "input": tool_input}
            _blocked_tool_timestamps[tool_id] = time.time()
            any_halted = True
            halted_indices.append(i)
            logger.warning(f"BLOCKED tool {tool_name} (id={tool_id}): {reason}")

        elif result.action == Action.ALERT:
            logger.warning(f"ALERT on tool {tool_name} (id={tool_id}): {result.reason}")

    if halted_indices:
        # Remove blocked tool_use blocks from response (reverse order to preserve indices)
        for i in reversed(halted_indices):
            blocks.pop(i)
        data["content"] = blocks
        # Fix stop_reason if no tool_use blocks remain
        if not any(b.get("type") == "tool_use" for b in blocks):
            data["stop_reason"] = "end_turn"
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


def _synthetic_message_end_sse() -> list[str]:
    """Generate SSE lines to cleanly end a response (end_turn + message_stop)."""
    return [
        "event: message_delta",
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":0}}',
        "",
        "event: message_stop",
        'data: {"type":"message_stop"}',
        "",
    ]


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
        current_body = body
        retries_remaining = settings.gate_max_retries
        next_content_index = 0  # Track how many content blocks have been sent to client
        is_first_attempt = True
        current_upstream = upstream
        current_parser = parser

        while True:
            buffer: list[str] = []
            buffering = False  # True when inside a tool_use content block
            any_halted = False  # If any tool in this message was halted, halt the rest
            response_terminated = False  # True when we've sent synthetic message_stop
            tool_index = 0  # Track tool position in this response
            pending_line: str | None = None  # Holds event: line until we know if next data: starts a buffer
            should_retry = False
            blocked_tools_for_retry: list[dict] = []

            try:
                async for line in current_upstream.aiter_lines():
                    events = current_parser.feed_line(line)

                    if gate_chain is None:
                        # No gating — pass through as before
                        for ev in events:
                            await event_bus.publish(ev)
                        yield line + "\n"
                        continue

                    # --- Gated streaming ---

                    # For retry attempts, skip message_start and rewrite content block indices
                    if not is_first_attempt:
                        # Skip message_start events in retry responses
                        if line.startswith("data: "):
                            try:
                                d = json.loads(line[6:])
                                if d.get("type") == "message_start":
                                    for ev in events:
                                        await event_bus.publish(ev)
                                    continue
                            except (json.JSONDecodeError, TypeError):
                                pass
                        if line.startswith("event: message_start"):
                            for ev in events:
                                await event_bus.publish(ev)
                            continue

                        # Rewrite content block indices
                        line = _rewrite_content_block_index(line, next_content_index)

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

                    # When buffering, ALL lines (event: and data:) go into the buffer
                    if buffering:
                        # Pull any held event: line into the buffer too
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
                            # Resolve script contents for Bash/exec tool calls
                            proxy_cwd = os.getcwd()
                            script_contents = _resolve_scripts_for_tool(
                                tool_event.name,
                                tool_event.input,
                                cwd=proxy_cwd,
                            )
                            tool_event.script_contents = script_contents or None

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
                                script_contents=script_contents,
                                cwd=proxy_cwd,
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
                                    script_contents=script_contents or None,
                                )
                            )

                            if result.action == Action.ESCALATE and not any_halted:
                                # Hold stream and wait for human approval
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
                                    logger.warning(
                                        f"Gate approval timed out for {tool_event.name} ({tool_event.tool_use_id})"
                                    )
                                finally:
                                    _pending_approvals.pop(tool_event.tool_use_id, None)

                                if approved:
                                    logger.info(f"APPROVED tool {tool_event.name} by {username}")
                                    # Publish approval so sentinel sees the full flow
                                    await event_bus.publish(
                                        GateDecisionEvent(
                                            tool_use_id=tool_event.tool_use_id,
                                            tool_name=tool_event.name,
                                            action="allow",
                                            reason=f"Approved by human reviewer (@{username})",
                                            evaluator="human",
                                            tool_input=tool_event.input,
                                        )
                                    )
                                    # Flush buffer — tool is approved
                                    await event_bus.publish(tool_event)
                                    for buffered_line in buffer:
                                        yield buffered_line + "\n"
                                    next_content_index += 1
                                else:
                                    # Denied escalation → retry (like BLOCK)
                                    reason = (
                                        f"Denied by human reviewer (@{username})"
                                        if username
                                        else "Timed out waiting for human approval"
                                    )
                                    logger.warning(
                                        f"DENIED tool {tool_event.name} ({tool_event.tool_use_id}): {reason}"
                                    )
                                    await event_bus.publish(
                                        GateDecisionEvent(
                                            tool_use_id=tool_event.tool_use_id,
                                            tool_name=tool_event.name,
                                            action="block",
                                            reason=reason,
                                            evaluator="human",
                                            tool_input=tool_event.input,
                                        )
                                    )
                                    await event_bus.publish(tool_event)
                                    blocked_tools_for_retry.append(
                                        {
                                            "tool_use_id": tool_event.tool_use_id,
                                            "name": tool_event.name,
                                            "input": tool_event.input,
                                            "reason": reason,
                                        }
                                    )
                                    should_retry = True
                                    break

                            elif result.action == Action.HALT_SESSION or any_halted:
                                # HALT_SESSION: freeze the agent (old BLOCK behavior)
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
                                logger.warning(f"HALTED tool {tool_event.name} (id={tool_event.tool_use_id}): {reason}")
                                any_halted = True
                                await event_bus.publish(tool_event)
                                # Suppress tool_use, terminate response cleanly
                                for sse_line in _synthetic_message_end_sse():
                                    yield sse_line + "\n"
                                response_terminated = True
                                break

                            elif result.action == Action.BLOCK:
                                # BLOCK: suppress tool_use, trigger retry
                                reason = result.reason
                                logger.warning(
                                    f"BLOCKED tool {tool_event.name} (id={tool_event.tool_use_id}): {reason}"
                                )
                                await event_bus.publish(tool_event)
                                blocked_tools_for_retry.append(
                                    {
                                        "tool_use_id": tool_event.tool_use_id,
                                        "name": tool_event.name,
                                        "input": tool_event.input,
                                        "reason": reason,
                                    }
                                )
                                should_retry = True
                                break

                            elif result.action == Action.ALERT:
                                # Alert: flush buffer (allow execution) but log
                                logger.warning(
                                    f"ALERT on tool {tool_event.name} (id={tool_event.tool_use_id}): {result.reason}"
                                )
                                await event_bus.publish(tool_event)
                                for buffered_line in buffer:
                                    yield buffered_line + "\n"
                                next_content_index += 1
                            else:
                                # Allow: flush buffer
                                await event_bus.publish(tool_event)
                                for buffered_line in buffer:
                                    yield buffered_line + "\n"
                                next_content_index += 1

                            buffer = []
                            buffering = False
                        # else: still accumulating tool_use deltas
                        continue

                    # --- Not buffering ---

                    # Track content blocks sent to client for index rewriting
                    if line.startswith("data: "):
                        try:
                            d = json.loads(line[6:])
                            if d.get("type") == "content_block_stop":
                                # A non-tool content block completed (text, etc.)
                                next_content_index += 1
                        except (json.JSONDecodeError, TypeError):
                            pass

                    # Flush any pending line that wasn't pulled into a buffer
                    if pending_line is not None:
                        yield pending_line + "\n"
                        pending_line = None

                    # Hold event: lines until we see the next data: line
                    # (so we can pull them into a buffer if the next line starts a tool_use)
                    if line.startswith("event: "):
                        pending_line = line
                        for ev in events:
                            await event_bus.publish(ev)
                        continue

                    # Stream text and other events through immediately
                    for ev in events:
                        await event_bus.publish(ev)
                    yield line + "\n"

                if not response_terminated and not should_retry:
                    # Flush any pending event line
                    if pending_line is not None:
                        yield pending_line + "\n"

                    # Flush any remaining parser state
                    events = current_parser.feed_line("")
                    for ev in events:
                        await event_bus.publish(ev)
            finally:
                await current_upstream.aclose()

            # Retry logic
            if should_retry and retries_remaining > 0:
                retries_remaining -= 1
                retry_number = settings.gate_max_retries - retries_remaining
                for bt in blocked_tools_for_retry:
                    await event_bus.publish(
                        GateRetryEvent(
                            tool_use_id=bt["tool_use_id"],
                            tool_name=bt["name"],
                            reason=bt["reason"],
                            retry_number=retry_number,
                            max_retries=settings.gate_max_retries,
                        )
                    )
                logger.info(f"Retrying after BLOCK (attempt {retry_number}/{settings.gate_max_retries})")
                current_body = _build_retry_request_body(current_body, blocked_tools_for_retry)
                headers["content-length"] = str(len(current_body))
                is_first_attempt = False
                current_parser = SSEParser()
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
                for sse_line in _synthetic_message_end_sse():
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
