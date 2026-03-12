"""Cross daemon — central process running the proxy, Slack, session registry, and local API."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from cross.chain import GateChain
from cross.config import settings
from cross.evaluator import Action
from cross.events import EventBus
from cross.plugins.logger import LoggerPlugin

logger = logging.getLogger("cross.daemon")

# Shared state
event_bus = EventBus()
_gate_chain: GateChain | None = None
_slack = None  # SlackPlugin instance, if configured

# Session tracking: session_id -> session metadata
_sessions: dict[str, dict[str, Any]] = {}
# session_id -> WebSocket connection from wrap process
_session_ws: dict[str, WebSocket] = {}
# project -> last known working directory (persists across sessions)
_project_cwds: dict[str, str] = {}
# session_id -> initial message to inject after WS connects
_pending_injects: dict[str, str] = {}


# --- Local API routes (for wrap processes) ---


async def api_register_session(request: Request) -> JSONResponse:
    """POST /cross/sessions — wrap process registers a new session."""
    data = await request.json()
    session_id = data["session_id"]
    _sessions[session_id] = data
    project = data.get("project", "")
    logger.info(f"Session registered: {session_id} ({data.get('agent')} in {project})")

    # Track project CWD for future Slack-initiated sessions
    if project and data.get("cwd"):
        _project_cwds[project] = data["cwd"]

    # Check for pending initial message (Slack-initiated session)
    if project in _pending_injects:
        _pending_injects[session_id] = _pending_injects.pop(project)

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

    # Inject pending initial message (Slack-initiated session)
    if session_id in _pending_injects:
        initial_msg = _pending_injects.pop(session_id)
        asyncio.create_task(_delayed_inject(session_id, initial_msg))

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


async def _delayed_inject(session_id: str, text: str):
    """Wait for Claude Code to initialize, then inject the initial message."""
    await asyncio.sleep(5)
    logger.info(f"Injecting initial message for session {session_id}: {text[:50]}")
    await _inject_to_session(session_id, text + "\r")


async def _spawn_session(project: str, initial_message: str):
    """Spawn a new agent session from Slack."""
    cross_bin = shutil.which("cross")
    if not cross_bin:
        logger.warning("cross binary not found — cannot spawn session")
        return

    cwd = _project_cwds.get(project, os.getcwd())
    _pending_injects[project] = initial_message

    # Clean env: remove CLAUDECODE to avoid nested session detection
    spawn_env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    proc = subprocess.Popen(
        [cross_bin, "wrap", "--", "claude"],
        cwd=cwd,
        env=spawn_env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        start_new_session=True,
    )
    logger.info(f"Spawned new session for project '{project}' in {cwd} (pid {proc.pid})")


# --- Proxy routes ---


async def _proxy_handler(request: Request) -> Response:
    """Forward to the actual proxy logic."""
    from cross.proxy import handle_proxy_request

    return await handle_proxy_request(request, event_bus, gate_chain=_gate_chain)


# --- App lifecycle ---


async def on_startup():
    global _slack, _gate_chain

    # Register logger plugin
    log_plugin = LoggerPlugin()
    event_bus.subscribe(log_plugin.handle)
    logger.info(f"Daemon starting on port {settings.listen_port}")

    # Set up gate chain
    if settings.gating_enabled:
        from pathlib import Path

        from cross.gates.denylist import DenylistGate

        rules_dir = Path(settings.rules_dir).expanduser()
        gate = DenylistGate(rules_dir=rules_dir)

        # LLM review gate (stage 2) — reviews denylist-flagged calls
        review_gate = None
        review_threshold = Action.BLOCK
        if settings.llm_gate_enabled:
            from cross.gates.llm_review import LLMReviewGate
            from cross.llm import LLMConfig, resolve_api_key

            llm_config = LLMConfig(
                model=settings.llm_gate_model,
                api_key=settings.llm_gate_api_key,
                base_url=settings.llm_gate_base_url,
                temperature=settings.llm_gate_temperature,
                max_tokens=settings.llm_gate_max_tokens,
                reasoning=settings.llm_gate_reasoning,
            )
            if resolve_api_key(llm_config):
                review_gate = LLMReviewGate(config=llm_config, timeout_ms=settings.llm_gate_timeout_ms)
                try:
                    review_threshold = Action[settings.llm_gate_threshold.upper()]
                except KeyError:
                    logger.warning(f"Invalid llm_gate_threshold '{settings.llm_gate_threshold}', using BLOCK")
                model_name = settings.llm_gate_model
                logger.info(f"LLM review gate active (model={model_name}, threshold={review_threshold.name})")
            else:
                logger.info("LLM gate enabled but no API key available — denylist operates standalone")

        _gate_chain = GateChain(gates=[gate], review_gate=review_gate, review_threshold=review_threshold)
        logger.info(f"Gating enabled with {len(gate.rules)} denylist rules")

    # Register Slack plugin
    if settings.slack_bot_token and settings.slack_app_token:
        from cross.plugins.slack import SlackPlugin

        _slack = SlackPlugin(
            inject_callback=_inject_to_session,
            spawn_callback=_spawn_session,
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
    # Clean up LLM httpx client
    from cross.llm import close_client

    await close_client()


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
