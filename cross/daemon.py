"""Cross daemon — central process running the proxy, Slack, session registry, and local API."""

from __future__ import annotations

import json
import logging
import asyncio
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from cross.config import settings
from cross.events import EventBus
from cross.plugins.logger import LoggerPlugin

logger = logging.getLogger("cross.daemon")

# Shared state
event_bus = EventBus()
_slack = None  # SlackPlugin instance, if configured

# Session tracking: session_id -> session metadata
_sessions: dict[str, dict[str, Any]] = {}
# session_id -> WebSocket connection from wrap process
_session_ws: dict[str, WebSocket] = {}


# --- Local API routes (for wrap processes) ---

async def api_register_session(request: Request) -> JSONResponse:
    """POST /cross/sessions — wrap process registers a new session."""
    data = await request.json()
    session_id = data["session_id"]
    _sessions[session_id] = data
    logger.info(f"Session registered: {session_id} ({data.get('agent')} in {data.get('project')})")

    # Notify Slack
    if _slack:
        try:
            _slack.session_started_from_data(data)
        except Exception as e:
            logger.warning(f"Slack session_started failed: {e}")

    return JSONResponse({"status": "ok", "session_id": session_id})


async def api_end_session(request: Request) -> JSONResponse:
    """POST /cross/sessions/{id}/end — wrap process reports session ended."""
    session_id = request.path_params["session_id"]
    data = await request.json()

    session = _sessions.get(session_id, {})
    session.update(data)
    logger.info(f"Session ended: {session_id} (exit code {data.get('exit_code')})")

    # Notify Slack
    if _slack:
        try:
            _slack.session_ended_from_data(session)
        except Exception as e:
            logger.warning(f"Slack session_ended failed: {e}")

    # Clean up
    _sessions.pop(session_id, None)
    _session_ws.pop(session_id, None)

    return JSONResponse({"status": "ok"})


async def api_session_ws(ws: WebSocket):
    """WS /cross/sessions/{id}/io — bidirectional I/O stream with wrap process."""
    session_id = ws.path_params["session_id"]
    await ws.accept()
    _session_ws[session_id] = ws
    logger.info(f"Session WS connected: {session_id}")

    try:
        while True:
            msg = await ws.receive_json()
            msg_type = msg.get("type")

            if msg_type == "pty_output":
                text = msg.get("text", "")
                if text and _slack:
                    _slack.handle_pty_output(session_id, text)

    except WebSocketDisconnect:
        logger.info(f"Session WS disconnected: {session_id}")
    except Exception as e:
        logger.warning(f"Session WS error: {e}")
    finally:
        _session_ws.pop(session_id, None)


async def _inject_to_session(session_id: str, text: str):
    """Send text to a wrap process's PTY via WebSocket."""
    ws = _session_ws.get(session_id)
    if ws:
        try:
            await ws.send_json({"type": "inject", "text": text})
        except Exception as e:
            logger.warning(f"Failed to inject to session {session_id}: {e}")


# --- Proxy routes ---

async def _proxy_handler(request: Request) -> Response:
    """Forward to the actual proxy logic."""
    from cross.proxy import handle_proxy_request
    return await handle_proxy_request(request, event_bus)


# --- App lifecycle ---

async def on_startup():
    global _slack

    # Register logger plugin
    log_plugin = LoggerPlugin()
    event_bus.subscribe(log_plugin.handle)
    logger.info(f"Daemon starting on port {settings.listen_port}")

    # Register Slack plugin
    if settings.slack_bot_token and settings.slack_app_token:
        from cross.plugins.slack import SlackPlugin
        _slack = SlackPlugin(
            inject_callback=_inject_to_session,
            event_loop=asyncio.get_running_loop(),
        )
        try:
            _slack.start()
            event_bus.subscribe(_slack.handle_event)
            logger.info("Slack relay active")
        except Exception as e:
            logger.warning(f"Slack failed to start: {e}")
            _slack = None


async def on_shutdown():
    if _slack:
        _slack.stop()


# --- Build the combined app ---

# Local API routes
_api_routes = [
    Route("/cross/sessions", api_register_session, methods=["POST"]),
    Route("/cross/sessions/{session_id}/end", api_end_session, methods=["POST"]),
]

# WebSocket route
_ws_routes = [
    WebSocketRoute("/cross/sessions/{session_id}/io", api_session_ws),
]

# Proxy catch-all (must be last)
_proxy_routes = [
    Route("/{path:path}", _proxy_handler, methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]),
]

app = Starlette(
    routes=_api_routes + _ws_routes + _proxy_routes,
    on_startup=[on_startup],
    on_shutdown=[on_shutdown],
)
