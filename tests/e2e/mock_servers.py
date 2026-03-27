"""Mock servers for e2e tests — fake LLM APIs, Slack, and SMTP.

Each mock is a lightweight Starlette/ASGI app that records requests and
returns scripted responses.  They run on ephemeral ports inside the test
process so no real network calls are needed.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

# ---------------------------------------------------------------------------
# Mock Anthropic Messages API
# ---------------------------------------------------------------------------


@dataclass
class MockAnthropicServer:
    """Fake Anthropic Messages API that returns scripted SSE tool_use or text responses."""

    requests: list[dict] = field(default_factory=list)
    # Queue of responses to return (consumed in order, last one repeats)
    responses: list[dict] = field(default_factory=list)
    _idx: int = 0

    def _next_response(self) -> dict:
        if not self.responses:
            return {"type": "text", "text": "Hello from mock Anthropic"}
        resp = self.responses[min(self._idx, len(self.responses) - 1)]
        if self._idx < len(self.responses) - 1:
            self._idx += 1
        return resp

    async def handle_messages(self, request: Request) -> Response:
        body = await request.json()
        self.requests.append(body)
        resp = self._next_response()
        stream = body.get("stream", False)

        if stream:
            return StreamingResponse(
                self._stream_sse(resp),
                media_type="text/event-stream",
                headers={"x-request-id": uuid.uuid4().hex[:16]},
            )
        return JSONResponse(self._non_streaming(resp))

    def _non_streaming(self, resp: dict) -> dict:
        msg_id = f"msg_{uuid.uuid4().hex[:12]}"
        content = self._build_content(resp)
        return {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6-20250514",
            "content": content,
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }

    def _build_content(self, resp: dict) -> list[dict]:
        if resp.get("type") == "tool_use":
            return [
                {
                    "type": "tool_use",
                    "id": resp.get("id", f"toolu_{uuid.uuid4().hex[:12]}"),
                    "name": resp.get("name", "bash"),
                    "input": resp.get("input", {"command": "echo hello"}),
                }
            ]
        return [{"type": "text", "text": resp.get("text", "Hello from mock")}]

    async def _stream_sse(self, resp: dict):
        msg_id = f"msg_{uuid.uuid4().hex[:12]}"
        model = "claude-sonnet-4-6-20250514"
        content = self._build_content(resp)

        # message_start
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': model, 'content': [], 'usage': {'input_tokens': 100, 'output_tokens': 0}}})}\n\n"

        for idx, block in enumerate(content):
            if block["type"] == "text":
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': idx, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': idx, 'delta': {'type': 'text_delta', 'text': block['text']}})}\n\n"
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n"
            elif block["type"] == "tool_use":
                yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': idx, 'content_block': {'type': 'tool_use', 'id': block['id'], 'name': block['name'], 'input': ''}})}\n\n"
                yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': idx, 'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(block['input'])}})}\n\n"
                yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': idx})}\n\n"

        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 50}})}\n\n"
        yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'

    def build_app(self) -> Starlette:
        return Starlette(
            routes=[
                Route("/v1/messages", self.handle_messages, methods=["POST"]),
            ]
        )


# ---------------------------------------------------------------------------
# Mock OpenAI Chat Completions API
# ---------------------------------------------------------------------------


@dataclass
class MockOpenAIServer:
    """Fake OpenAI Chat Completions API."""

    requests: list[dict] = field(default_factory=list)
    responses: list[dict] = field(default_factory=list)
    _idx: int = 0

    def _next_response(self) -> dict:
        if not self.responses:
            return {"type": "text", "text": "Hello from mock OpenAI"}
        resp = self.responses[min(self._idx, len(self.responses) - 1)]
        if self._idx < len(self.responses) - 1:
            self._idx += 1
        return resp

    async def handle_completions(self, request: Request) -> Response:
        body = await request.json()
        self.requests.append(body)
        resp = self._next_response()
        stream = body.get("stream", False)

        if stream:
            return StreamingResponse(
                self._stream_sse(resp),
                media_type="text/event-stream",
            )
        return JSONResponse(self._non_streaming(resp))

    def _non_streaming(self, resp: dict) -> dict:
        choice: dict[str, Any] = {"index": 0, "finish_reason": "stop"}
        if resp.get("type") == "tool_use":
            choice["message"] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": resp.get("id", f"call_{uuid.uuid4().hex[:12]}"),
                        "type": "function",
                        "function": {
                            "name": resp.get("name", "bash"),
                            "arguments": json.dumps(resp.get("input", {"command": "echo hi"})),
                        },
                    }
                ],
            }
            choice["finish_reason"] = "tool_calls"
        else:
            choice["message"] = {"role": "assistant", "content": resp.get("text", "Hello")}

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "model": "gpt-4o-2024-08-06",
            "choices": [choice],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        }

    async def _stream_sse(self, resp: dict):
        chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        empty = {}  # avoid f-string brace escaping issues
        if resp.get("type") == "tool_use":
            tool_id = resp.get("id", f"call_{uuid.uuid4().hex[:12]}")
            delta0 = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "index": 0,
                        "id": tool_id,
                        "type": "function",
                        "function": {"name": resp.get("name", "bash"), "arguments": ""},
                    }
                ],
            }
            yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': delta0}]})}\n\n"
            delta1 = {"tool_calls": [{"index": 0, "function": {"arguments": json.dumps(resp.get("input", {}))}}]}
            yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': delta1}]})}\n\n"
            yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': empty, 'finish_reason': 'tool_calls'}]})}\n\n"
        else:
            text = resp.get("text", "Hello")
            yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}}]})}\n\n"
            yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': {'content': text}}]})}\n\n"
            yield f"data: {json.dumps({'id': chat_id, 'object': 'chat.completion.chunk', 'choices': [{'index': 0, 'delta': empty, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

    def build_app(self) -> Starlette:
        return Starlette(
            routes=[
                Route("/v1/chat/completions", self.handle_completions, methods=["POST"]),
            ]
        )


# ---------------------------------------------------------------------------
# Mock Slack API
# ---------------------------------------------------------------------------


@dataclass
class MockSlackAPI:
    """Records Slack Web API calls and returns canned responses."""

    calls: list[dict] = field(default_factory=list)
    channel_id: str = "C_MOCK_CHAN"
    thread_ts: str = "1700000000.000001"

    async def handle(self, request: Request) -> JSONResponse:
        path = request.url.path
        try:
            body = await request.json()
        except Exception:
            body = dict(request.query_params)
        self.calls.append({"method": path, "body": body})

        if "conversations.list" in path:
            return JSONResponse({"ok": True, "channels": []})
        if "conversations.create" in path:
            return JSONResponse({"ok": True, "channel": {"id": self.channel_id}})
        if "conversations.members" in path:
            return JSONResponse({"ok": True, "members": []})
        if "users.list" in path:
            return JSONResponse({"ok": True, "members": []})
        if "chat.postMessage" in path or "chat.update" in path:
            return JSONResponse({"ok": True, "ts": self.thread_ts, "channel": self.channel_id})
        if "reactions.add" in path:
            return JSONResponse({"ok": True})

        return JSONResponse({"ok": True})

    def build_app(self) -> Starlette:
        return Starlette(
            routes=[
                Route("/{path:path}", self.handle, methods=["GET", "POST"]),
            ]
        )

    def messages_posted(self) -> list[dict]:
        """Return bodies of chat.postMessage calls."""
        return [c["body"] for c in self.calls if "chat.postMessage" in c["method"]]


# ---------------------------------------------------------------------------
# Mock SMTP server (in-process, records sent emails)
# ---------------------------------------------------------------------------


class MockSMTP:
    """In-memory mock that patches smtplib.SMTP to capture sent emails.

    Used as a class-level mock: `patch("smtplib.SMTP", MockSMTPFactory(instance))`.
    """

    def __init__(self):
        self.sent: list[dict] = []

    def sendmail(self, from_addr, to_addrs, msg_string):
        self.sent.append(
            {
                "from": from_addr,
                "to": to_addrs,
                "raw": msg_string,
            }
        )

    def send_message(self, msg, from_addr=None, to_addrs=None):
        self.sent.append(
            {
                "from": from_addr or msg.get("From", ""),
                "to": to_addrs or msg.get("To", ""),
                "subject": msg.get("Subject", ""),
                "body": msg.get_payload() if hasattr(msg, "get_payload") else str(msg),
            }
        )

    def login(self, user, password):
        pass

    def starttls(self, **kw):
        pass

    def ehlo(self):
        pass

    def quit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# Mock LLM gate/sentinel responses (for the Cross LLM reviewer)
# ---------------------------------------------------------------------------


@dataclass
class MockGateLLMServer:
    """Mock LLM that returns ALLOW/BLOCK verdicts for gate reviews.

    Used as the LLM gate model — returns structured verdicts that
    the LLMReviewGate can parse.
    """

    requests: list[dict] = field(default_factory=list)
    verdict: str = "ALLOW"  # Default verdict

    async def handle_messages(self, request: Request) -> Response:
        body = await request.json()
        self.requests.append(body)
        stream = body.get("stream", False)

        # Gate expects: "VERDICT: ALLOW", "VERDICT: BLOCK", "VERDICT: ESCALATE"
        response_text = f"VERDICT: {self.verdict}"

        if stream:
            return StreamingResponse(
                self._stream_verdict(response_text),
                media_type="text/event-stream",
            )
        return JSONResponse(
            {
                "id": f"msg_{uuid.uuid4().hex[:12]}",
                "type": "message",
                "role": "assistant",
                "model": "claude-sonnet-4-6-20250514",
                "content": [{"type": "text", "text": response_text}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 200, "output_tokens": 10},
            }
        )

    async def _stream_verdict(self, text: str):
        msg_id = f"msg_{uuid.uuid4().hex[:12]}"
        yield f"event: message_start\ndata: {json.dumps({'type': 'message_start', 'message': {'id': msg_id, 'type': 'message', 'role': 'assistant', 'model': 'claude-sonnet-4-6-20250514', 'content': [], 'usage': {'input_tokens': 200, 'output_tokens': 0}}})}\n\n"
        yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
        yield f"event: content_block_delta\ndata: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}})}\n\n"
        yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
        yield f"event: message_delta\ndata: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': 'end_turn'}, 'usage': {'output_tokens': 10}})}\n\n"
        yield 'event: message_stop\ndata: {"type": "message_stop"}\n\n'

    def build_app(self) -> Starlette:
        return Starlette(
            routes=[
                Route("/v1/messages", self.handle_messages, methods=["POST"]),
            ]
        )
