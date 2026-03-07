"""Core reverse proxy — forwards requests to Anthropic API and parses SSE responses."""

from __future__ import annotations

import json
import logging

import httpx
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse

from cross.config import settings
from cross.events import EventBus, RequestEvent, ErrorEvent
from cross.sse import SSEParser

logger = logging.getLogger("cross.proxy")

_client: httpx.AsyncClient | None = None


def get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(
            base_url=settings.anthropic_base_url,
            timeout=httpx.Timeout(connect=10, read=300, write=30, pool=10),
        )
    return _client


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
                content = last.get("content", "")
                if isinstance(content, str):
                    event.last_message_preview = content[:200]
                elif isinstance(content, list):
                    # Extract the user's actual text, skipping system-injected blocks
                    for b in content:
                        if b.get("type") == "text":
                            text = b.get("text", "")
                            if text and not text.startswith("<system-reminder>"):
                                event.last_message_preview = text[:200]
                                break
                    if not event.last_message_preview:
                        types = [b.get("type", "?") for b in content]
                        event.last_message_preview = f"[{', '.join(types)}]"
        except (json.JSONDecodeError, KeyError):
            pass
    return event


async def handle_proxy_request(request: Request, event_bus: EventBus) -> Response:
    """Handle a proxy request, publishing events to the given EventBus."""
    body = await request.body()
    path = request.url.path
    if request.url.query:
        path = f"{path}?{request.url.query}"

    # Build upstream headers — forward everything except host
    headers = dict(request.headers)
    headers.pop("host", None)
    headers["host"] = "api.anthropic.com"

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
        return await _proxy_streaming(client, event_bus, request.method, path, headers, body)
    else:
        return await _proxy_simple(client, event_bus, request.method, path, headers, body)


async def _proxy_simple(
    client: httpx.AsyncClient,
    event_bus: EventBus,
    method: str,
    path: str,
    headers: dict,
    body: bytes,
) -> Response:
    resp = await client.request(method, path, headers=headers, content=body)

    if resp.status_code >= 400:
        await event_bus.publish(ErrorEvent(
            status_code=resp.status_code,
            body=resp.text[:500],
        ))

    resp_headers = dict(resp.headers)
    resp_headers.pop("transfer-encoding", None)
    resp_headers.pop("content-encoding", None)

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=resp_headers,
    )


async def _proxy_streaming(
    client: httpx.AsyncClient,
    event_bus: EventBus,
    method: str,
    path: str,
    headers: dict,
    body: bytes,
) -> Response:
    upstream = await client.send(
        client.build_request(method, path, headers=headers, content=body),
        stream=True,
    )

    parser = SSEParser()

    async def generate():
        try:
            async for line in upstream.aiter_lines():
                events = parser.feed_line(line)
                for ev in events:
                    await event_bus.publish(ev)
                yield line + "\n"

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
