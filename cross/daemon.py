"""Cross daemon — central process running the proxy, Slack, session registry, and local API."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
import uuid
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from cross.chain import GateChain
from cross.config import settings
from cross.evaluator import Action, GateRequest
from cross.event_store import EventStore
from cross.events import EventBus, GateDecisionEvent, ToolUseEvent
from cross.plugins.dashboard import DASHBOARD_HTML, DashboardPlugin
from cross.plugins.logger import LoggerPlugin
from cross.plugins.notifier import handle_event as notify_event
from cross.plugins.notifier import is_available as native_notifications_available
from cross.plugins.notifier import set_browser_check

logger = logging.getLogger("cross.daemon")

# Shared state
event_bus = EventBus()
_gate_chain: GateChain | None = None
_slack = None  # SlackPlugin instance, if configured
_dashboard: DashboardPlugin | None = None  # always active

# Session tracking: session_id -> session metadata
_sessions: dict[str, dict[str, Any]] = {}
# session_id -> WebSocket connection from wrap process
_session_ws: dict[str, WebSocket] = {}
# project -> last known working directory (persists across sessions)
_project_cwds: dict[str, str] = {}
# session_id -> initial message to inject after WS connects
_pending_injects: dict[str, str] = {}
# Agents seen via the gate API (not registered via cross wrap)
_gate_agents: set[str] = set()


def _detect_hooked_agents() -> set[str]:
    """Detect agents that have the cross hook installed (monitored even before first call)."""
    from pathlib import Path

    hooked: set[str] = set()
    # Check OpenClaw: if the gateway plist has the cross hook, it's monitored
    openclaw_plist = Path.home() / "Library" / "LaunchAgents" / "ai.openclaw.gateway.plist"
    if openclaw_plist.exists():
        try:
            content = openclaw_plist.read_text()
            if "openclaw_hook" in content:
                hooked.add("openclaw")
        except OSError:
            pass
    return hooked


def _detect_running_agents() -> dict[str, list[int]]:
    """Detect running agent processes. Returns {agent_name: [pids]}.

    Claude Desktop Code sessions are reported separately as "claude (desktop)"
    so they can be distinguished from CLI sessions in the dashboard.
    """
    # Pattern per agent — more specific than just the name to avoid false positives
    agent_patterns = {
        "claude": ["-x", "claude"],  # exact binary name match
        "openclaw": ["-f", "openclaw-gateway"],  # runs as openclaw-gateway
    }

    own_pid = str(os.getpid())
    result: dict[str, list[int]] = {}
    for agent, pgrep_args in agent_patterns.items():
        try:
            proc = subprocess.run(
                ["pgrep"] + pgrep_args,
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                pids = [int(p) for p in proc.stdout.strip().splitlines() if p != own_pid]
                if pids:
                    result[agent] = pids
        except (OSError, ValueError):
            continue

    # Distinguish Claude Desktop Code sessions from CLI sessions
    if "claude" in result:
        cli_pids = []
        desktop_pids = []
        for pid in result["claude"]:
            if _is_desktop_pid(pid):
                desktop_pids.append(pid)
            else:
                cli_pids.append(pid)
        result["claude"] = cli_pids
        if not cli_pids:
            del result["claude"]
        if desktop_pids:
            result["claude (desktop)"] = desktop_pids

    return result


def _is_desktop_pid(pid: int) -> bool:
    """Check if a PID is a Claude Desktop app Code session."""
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "comm="],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return "/Claude/claude-code/" in proc.stdout
    except OSError:
        pass
    return False


def get_agent_status() -> dict[str, Any]:
    """Return monitoring coverage: monitored sessions and unmonitored agents."""
    running = _detect_running_agents()
    running_agent_names = set(running.keys())

    # Clean up stale sessions (process no longer running)
    stale = [sid for sid, s in _sessions.items() if s.get("agent") not in running_agent_names]
    for sid in stale:
        del _sessions[sid]

    # Clean up stale gate agents (process no longer running)
    _gate_agents.intersection_update(running_agent_names | {s.get("agent") for s in _sessions.values()})

    monitored = []
    monitored_pids: set[int] = set()
    for session in _sessions.values():
        agent = session.get("agent", "unknown")
        project = session.get("project", "")
        label = f"{agent} - {project}" if project else agent
        pid = session.get("pid")
        monitored.append({"agent": agent, "project": project, "label": label})
        if pid:
            monitored_pids.add(int(pid))

    monitored_agents = {s.get("agent") for s in _sessions.values()} | _gate_agents

    # Add gate-only agents (e.g. OpenClaw) to monitored list
    for agent in _gate_agents:
        if agent not in {s.get("agent") for s in _sessions.values()}:
            monitored.append({"agent": agent, "project": "", "label": agent})

    unmonitored = []
    for agent, pids in running.items():
        # Filter out PIDs we know are monitored
        unmonitored_pids = [p for p in pids if p not in monitored_pids]
        if unmonitored_pids and agent not in monitored_agents:
            unmonitored.append({"agent": agent, "count": len(unmonitored_pids), "pids": unmonitored_pids})

    return {
        "monitored": monitored,
        "unmonitored": unmonitored,
        "monitored_count": len(monitored),
        "unmonitored_count": len(unmonitored),
    }


def get_active_agent_label() -> str:
    """Return 'agent - project' label for the most recently active session."""
    if not _sessions:
        return ""
    # Use the most recently registered session
    last = list(_sessions.values())[-1]
    agent = last.get("agent", "")
    project = last.get("project", "")
    if agent and project:
        return f"{agent} - {project}"
    return agent or project or ""


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


# --- External gate API ---


async def api_gate(request: Request) -> JSONResponse:
    """POST /cross/api/gate — synchronous gate evaluation for external agents (e.g. OpenClaw)."""
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {})
    agent = data.get("agent", "unknown")
    session_id = data.get("session_id", "")
    user_intent = data.get("user_intent", "")
    cwd = data.get("cwd", "")

    # Track this agent as monitored via gate API
    if agent and agent != "unknown":
        _gate_agents.add(agent)

    if not _gate_chain:
        return JSONResponse({"action": "ALLOW", "reason": "No gate chain configured"})

    tool_use_id = f"ext-{uuid.uuid4().hex[:12]}"

    # Resolve script contents for Bash/exec tool calls
    from cross.proxy import _resolve_scripts_for_tool

    script_contents = _resolve_scripts_for_tool(tool_name, tool_input, cwd=cwd)

    gate_request = GateRequest(
        tool_use_id=tool_use_id,
        tool_name=tool_name,
        tool_input=tool_input,
        agent=agent,
        session_id=session_id,
        timestamp=time.time(),
        user_intent=user_intent,
        cwd=cwd,
        script_contents=script_contents,
    )

    result = await _gate_chain.evaluate(gate_request)

    # Publish tool call and gate decision events
    await event_bus.publish(
        ToolUseEvent(
            name=tool_name,
            tool_use_id=tool_use_id,
            input=tool_input if isinstance(tool_input, dict) else {},
            script_contents=script_contents or None,
            agent=agent,
            session_id=session_id,
        )
    )
    await event_bus.publish(
        GateDecisionEvent(
            tool_use_id=tool_use_id,
            tool_name=tool_name,
            action=result.action.name.lower(),
            reason=result.reason,
            rule_id=result.rule_id,
            evaluator=result.evaluator,
            confidence=result.confidence,
            tool_input=tool_input if isinstance(tool_input, dict) else {},
            script_contents=script_contents or None,
            agent=agent,
            session_id=session_id,
        )
    )

    # Handle escalation — wait for human approval
    if result.action in (Action.REVIEW, Action.ESCALATE):
        from cross.proxy import _approval_results, _pending_approvals

        approval_event = asyncio.Event()
        _pending_approvals[tool_use_id] = approval_event
        logger.info(
            f"ESCALATED external tool {tool_name} ({tool_use_id}), "
            f"waiting for human approval (timeout={settings.gate_approval_timeout}s)"
        )

        try:
            await asyncio.wait_for(
                approval_event.wait(),
                timeout=settings.gate_approval_timeout,
            )
            approved, username = _approval_results.pop(tool_use_id, (False, ""))
        except asyncio.TimeoutError:
            approved, username = False, ""
            logger.warning(f"Gate approval timed out for external tool {tool_name} ({tool_use_id})")
        finally:
            _pending_approvals.pop(tool_use_id, None)

        if approved:
            logger.info(f"APPROVED external tool {tool_name} by {username}")
            await event_bus.publish(
                GateDecisionEvent(
                    tool_use_id=tool_use_id,
                    tool_name=tool_name,
                    action="allow",
                    reason=f"Approved by human reviewer (@{username})",
                    evaluator="human",
                    tool_input=tool_input if isinstance(tool_input, dict) else {},
                )
            )
            return JSONResponse(
                {
                    "action": "ALLOW",
                    "reason": f"Approved by human reviewer (@{username})",
                    "evaluator": "human",
                }
            )
        else:
            reason = f"Denied by human reviewer (@{username})" if username else "Timed out waiting for human approval"
            await event_bus.publish(
                GateDecisionEvent(
                    tool_use_id=tool_use_id,
                    tool_name=tool_name,
                    action="block",
                    reason=reason,
                    evaluator="human",
                    tool_input=tool_input if isinstance(tool_input, dict) else {},
                )
            )
            return JSONResponse(
                {
                    "action": "BLOCK",
                    "reason": reason,
                    "evaluator": "human",
                }
            )

    return JSONResponse(
        {
            "action": result.action.name,
            "reason": result.reason,
            "evaluator": result.evaluator,
        }
    )


# --- Dashboard routes ---


async def root_redirect(request: Request) -> RedirectResponse:
    """GET / — redirect to dashboard."""
    return RedirectResponse(url="/cross/dashboard")


async def dashboard_page(request: Request) -> Response:
    """GET /cross/dashboard — serves the single-page dashboard HTML."""
    return Response(content=DASHBOARD_HTML, media_type="text/html")


async def api_events(request: Request) -> JSONResponse:
    """GET /cross/api/events — recent events as JSON."""
    if _dashboard:
        return JSONResponse(_dashboard.get_events())
    return JSONResponse([])


async def api_pending(request: Request) -> JSONResponse:
    """GET /cross/api/pending — pending escalations as JSON."""
    if _dashboard:
        return JSONResponse(_dashboard.get_pending())
    return JSONResponse([])


async def api_resolve_pending(request: Request) -> JSONResponse:
    """POST /cross/api/pending/{tool_use_id}/resolve — approve or deny."""
    tool_use_id = request.path_params["tool_use_id"]
    data = await request.json()
    approved = data.get("approved", False)
    username = data.get("username", "dashboard")

    if _dashboard:
        _dashboard.resolve(tool_use_id, approved, username)
        return JSONResponse({"status": "ok", "tool_use_id": tool_use_id, "approved": approved})
    return JSONResponse({"status": "error", "message": "dashboard not initialized"}, status_code=500)


async def api_dashboard_ws(ws: WebSocket):
    """WS /cross/api/ws — real-time event stream to dashboard clients."""
    if _dashboard:
        await _dashboard.ws_handler(ws)
    else:
        await ws.close()


# --- Proxy routes ---


async def _proxy_handler(request: Request) -> Response:
    """Forward to the actual proxy logic."""
    from cross.proxy import handle_proxy_request

    return await handle_proxy_request(request, event_bus, gate_chain=_gate_chain)


# --- App lifecycle ---


_sentinel = None  # LLMSentinel instance, if configured


async def on_startup():
    global _slack, _gate_chain, _sentinel, _dashboard

    # Register logger plugin
    log_plugin = LoggerPlugin()
    event_bus.subscribe(log_plugin.handle)
    logger.info(f"Daemon starting on port {settings.listen_port}")

    # Register event store (persists events to JSONL)
    store = EventStore()
    event_bus.subscribe(store.handle_event)

    # Register dashboard plugin (always active — no config gating)
    from cross.proxy import resolve_gate_approval as _resolve_gate

    _dashboard = DashboardPlugin(event_store=store, resolve_approval_callback=_resolve_gate)
    event_bus.subscribe(_dashboard.handle_event)
    logger.info("Dashboard active at /cross/dashboard")

    # Pre-populate monitored agents from hook configuration
    hooked = _detect_hooked_agents()
    _gate_agents.update(hooked)
    if hooked:
        logger.info(f"Detected hooked agents: {', '.join(hooked)}")

    # Register native desktop notifications (macOS)
    if native_notifications_available():
        set_browser_check(lambda: len(_dashboard._ws_clients) > 0)
        event_bus.subscribe(notify_event)
        logger.info("Native desktop notifications active")

    # Set up gate chain
    if settings.gating_enabled:
        from pathlib import Path

        from cross.gates.denylist import DenylistGate

        rules_dir = Path(settings.rules_dir).expanduser()
        gate = DenylistGate(rules_dir=rules_dir)

        # LLM review gate (stage 2) — reviews denylist-flagged calls
        review_gate = None
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
            if resolve_api_key(llm_config) or llm_config.provider == "cli":
                review_gate = LLMReviewGate(
                    config=llm_config,
                    timeout_ms=settings.llm_gate_timeout_ms,
                    justification=settings.llm_gate_justification,
                )
                model_name = settings.llm_gate_model
                logger.info(f"LLM review gate active (model={model_name})")
            else:
                logger.info("LLM gate enabled but no API key available — denylist operates standalone")

        _gate_chain = GateChain(gates=[gate], review_gate=review_gate)
        logger.info(f"Gating enabled with {len(gate.rules)} denylist rules")

    # Set up LLM sentinel (async periodic reviewer)
    if settings.llm_sentinel_enabled:
        from cross.llm import LLMConfig, resolve_api_key
        from cross.sentinels.llm_reviewer import LLMSentinel

        sentinel_config = LLMConfig(
            model=settings.llm_sentinel_model,
            api_key=settings.llm_sentinel_api_key,
            base_url=settings.llm_sentinel_base_url,
            temperature=settings.llm_sentinel_temperature,
            max_tokens=settings.llm_sentinel_max_tokens,
            reasoning=settings.llm_sentinel_reasoning,
        )
        if resolve_api_key(sentinel_config) or sentinel_config.provider == "cli":
            _sentinel = LLMSentinel(
                config=sentinel_config,
                event_bus=event_bus,
                interval_seconds=settings.llm_sentinel_interval_seconds,
            )
            event_bus.subscribe(_sentinel.observe)
            _sentinel.start()
            logger.info(
                f"LLM sentinel active (model={settings.llm_sentinel_model}, "
                f"interval={settings.llm_sentinel_interval_seconds}s)"
            )
        else:
            logger.info("LLM sentinel enabled but no API key available — sentinel inactive")

    # Register Slack plugin
    if settings.slack_bot_token and settings.slack_app_token:
        from cross.plugins.slack import SlackPlugin
        from cross.proxy import resolve_gate_approval

        _slack = SlackPlugin(
            inject_callback=_inject_to_session,
            spawn_callback=_spawn_session,
            resolve_approval_callback=resolve_gate_approval,
            event_loop=asyncio.get_running_loop(),
        )
        try:
            _slack.start()
            event_bus.subscribe(_slack.handle_event)
            logger.info("Slack relay active")
        except Exception as e:
            logger.warning(f"Slack failed to start: {e}")
            _slack = None

    # Start auto-update background task
    if settings.auto_update_enabled:
        from cross.auto_update import run_update_loop
        from cross.plugins.notifier import _notify as native_notify

        notify_fn = native_notify if native_notifications_available() else None
        asyncio.create_task(run_update_loop(settings.auto_update_interval_hours, notify_fn=notify_fn))
        logger.info(f"Auto-update active (every {settings.auto_update_interval_hours}h)")


async def on_shutdown():
    if _sentinel:
        _sentinel.stop()
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
    Route("/cross/api/gate", api_gate, methods=["POST"]),
]


# Dashboard routes
async def favicon(request: Request) -> Response:
    """GET /favicon.ico — empty response to avoid proxy forwarding."""
    return Response(status_code=204)


async def api_status(request: Request) -> JSONResponse:
    """GET /cross/api/status — monitoring coverage."""
    return JSONResponse(get_agent_status())


_dashboard_routes = [
    Route("/", root_redirect, methods=["GET"]),
    Route("/favicon.ico", favicon, methods=["GET"]),
    Route("/cross/dashboard", dashboard_page, methods=["GET"]),
    Route("/cross/api/status", api_status, methods=["GET"]),
    Route("/cross/api/events", api_events, methods=["GET"]),
    Route("/cross/api/pending", api_pending, methods=["GET"]),
    Route("/cross/api/pending/{tool_use_id}/resolve", api_resolve_pending, methods=["POST"]),
]

# WebSocket routes
_ws_routes = [
    WebSocketRoute("/cross/sessions/{session_id}/io", api_session_ws),
    WebSocketRoute("/cross/api/ws", api_dashboard_ws),
]

# Proxy catch-all (must be last)
_proxy_routes = [
    Route("/{path:path}", _proxy_handler, methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"]),
]

app = Starlette(
    routes=_api_routes + _dashboard_routes + _ws_routes + _proxy_routes,
    on_startup=[on_startup],
    on_shutdown=[on_shutdown],
)
