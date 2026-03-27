"""Cross daemon — central process running the proxy, Slack, session registry, and local API."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import time
import uuid
from collections import deque
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from cross.chain import GateChain
from cross.config import settings
from cross.custom_instructions import CustomInstructions
from cross.evaluator import Action, GateRequest
from cross.event_store import EventStore
from cross.events import (
    EventBus,
    GateDecisionEvent,
    PermissionPromptEvent,
    PermissionResolvedEvent,
    RequestEvent,
    TextEvent,
    ToolUseEvent,
)
from cross.llm import CLI_PROVIDERS as _CLI_PROVIDERS
from cross.plugins.dashboard import DASHBOARD_HTML, SETTINGS_HTML, DashboardPlugin
from cross.plugins.logger import LoggerPlugin
from cross.plugins.notifier import handle_event as notify_event
from cross.plugins.notifier import is_available as native_notifications_available
from cross.plugins.notifier import set_browser_check

logger = logging.getLogger("cross.daemon")

# Shared state
event_bus = EventBus()
_gate_chain: GateChain | None = None
_slack = None  # SlackPlugin instance, if configured
_email = None  # EmailPlugin instance, if configured
_dashboard: DashboardPlugin | None = None  # always active
_custom_instructions: CustomInstructions | None = None
_conversation_store = None  # ConversationStore instance

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
# Per-session recent tool calls for external agents (keyed by session_id)
_gate_recent_tools: dict[str, deque] = {}
# Per-session last activity timestamp (updated on proxy/gate requests)
_session_last_activity: dict[str, float] = {}
# Permission prompt tracking (hook-based — PermissionRequest hook notifies daemon)
_permission_pending: dict[str, dict] = {}  # session_id -> {tool_desc, ts}
_permission_notify_tasks: dict[str, asyncio.Task] = {}  # session_id -> delayed notify task
# Event loop reference (for thread-safe callbacks)
_event_loop: asyncio.AbstractEventLoop | None = None


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
    # Check Codex: if the hooks.json has the cross hook, it's monitored
    codex_hooks = Path.home() / ".codex" / "hooks.json"
    if codex_hooks.exists():
        try:
            content = codex_hooks.read_text()
            if "codex_hook" in content:
                hooked.add("codex")
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
        "codex": ["-x", "codex"],  # OpenAI Codex CLI
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

    # Filter out codex PIDs that are Cursor extension subprocesses
    # (Cursor bundles a codex binary in its OpenAI ChatGPT extension)
    if "codex" in result:
        real_codex = [p for p in result["codex"] if not _is_cursor_embedded_codex(p)]
        if real_codex:
            result["codex"] = real_codex
        else:
            del result["codex"]

    return result


def _is_cursor_embedded_codex(pid: int) -> bool:
    """Check if a codex PID is Cursor's bundled codex binary (not standalone CLI)."""
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "comm="],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return ".cursor/" in proc.stdout
    except OSError:
        pass
    return False


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


def record_session_activity(session_id: str) -> None:
    """Record that a session has had recent activity (proxy or gate request)."""
    if session_id:
        _session_last_activity[session_id] = time.time()


def get_agent_status() -> dict[str, Any]:
    """Return monitoring coverage: monitored sessions and unmonitored agents."""
    running = _detect_running_agents()
    running_agent_names = set(running.keys())

    # Clean up stale sessions: drop if agent process no longer running,
    # then deduplicate by agent+project (keep newest per group).
    stale = [sid for sid, s in _sessions.items() if s.get("agent") not in running_agent_names]
    # Deduplicate: for each agent+project, keep only the newest session
    groups: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for sid, s in _sessions.items():
        if sid in stale:
            continue
        key = (s.get("agent", ""), s.get("project", ""))
        groups.setdefault(key, []).append((sid, s.get("started_at", 0)))
    for entries in groups.values():
        if len(entries) > 1:
            entries.sort(key=lambda x: x[1], reverse=True)
            for sid, _ in entries[1:]:
                stale.append(sid)
    for sid in stale:
        del _sessions[sid]

    # Clean up stale gate agents (process no longer running)
    _gate_agents.intersection_update(running_agent_names | {s.get("agent") for s in _sessions.values()})

    now = time.time()

    # Clean up stale activity entries
    stale_activity = [sid for sid in _session_last_activity if sid not in _sessions]
    for sid in stale_activity:
        del _session_last_activity[sid]

    monitored = []
    monitored_pids: set[int] = set()
    for sid, session in _sessions.items():
        agent = session.get("agent", "unknown")
        project = session.get("project", "")
        label = f"{agent} - {project}" if project else agent
        pid = session.get("pid")
        last_activity = _session_last_activity.get(sid, 0)
        active = (now - last_activity) < settings.activity_threshold_seconds if last_activity else False
        monitored.append({"agent": agent, "project": project, "label": label, "active": active})
        if pid:
            monitored_pids.add(int(pid))

    monitored_agents = {s.get("agent") for s in _sessions.values()} | _gate_agents

    # Add gate-only agents (e.g. OpenClaw) to monitored list
    for agent in _gate_agents:
        if agent not in {s.get("agent") for s in _sessions.values()}:
            # Gate-only agents: check if any gate request was recent
            gate_active = any(
                (now - _session_last_activity.get(sid, 0)) < settings.activity_threshold_seconds
                for sid, s in _sessions.items()
                if s.get("agent") == agent
            )
            monitored.append({"agent": agent, "project": "", "label": agent, "active": gate_active})

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


def get_agent_label(session_id: str = "") -> str:
    """Return 'agent - project' label for a specific session, or the most recent."""
    if session_id and session_id in _sessions:
        s = _sessions[session_id]
    elif _sessions:
        s = list(_sessions.values())[-1]
    else:
        return ""
    agent = s.get("agent", "")
    project = s.get("project", "")
    if agent and project:
        return f"{agent} - {project}"
    return agent or project or ""


# Backward compat alias
get_active_agent_label = get_agent_label


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

    # Notify Email
    if _email:
        try:
            _email.session_started_from_data(data)
        except Exception as e:
            logger.warning(f"Email session_started failed: {e}")

    # Persist state so this session survives daemon restarts
    _persist_state()

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

    # Notify Email
    if _email:
        try:
            _email.session_ended_from_data(session)
        except Exception as e:
            logger.warning(f"Email session_ended failed: {e}")

    # Clean up
    _sessions.pop(session_id, None)
    _session_ws.pop(session_id, None)

    # Persist state after session cleanup
    _persist_state()

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
                if text:
                    # Forward to Slack for relay
                    if _slack:
                        _slack.handle_pty_output(session_id, text)
                    # Forward to Email for relay
                    if _email:
                        _email.handle_pty_output(session_id, text)

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


async def _delayed_permission_notify(session_id: str, tool_desc: str):
    """Wait, then publish the permission notification if still pending."""
    await asyncio.sleep(settings.permission_notify_delay)

    if session_id not in _permission_pending:
        logger.debug(f"Permission prompt for {session_id} resolved before delay elapsed")
        return

    # Still pending after delay — user hasn't acted, send notification
    # Keep the entry but mark as notified so resolve_permission can still find it
    # and _clear_permission_on_activity won't clear it prematurely
    _permission_pending[session_id]["notified"] = True
    logger.info(
        f"Permission prompt still pending for {session_id} after {settings.permission_notify_delay}s, notifying"
    )
    await event_bus.publish(
        PermissionPromptEvent(
            session_id=session_id,
            tool_desc=tool_desc,
            allow_all_label="Allow all (session)",
        )
    )


async def api_permission_hook(request: Request) -> JSONResponse:
    """POST /cross/sessions/{session_id}/permission — PermissionRequest hook notification.

    Called by the Claude Code PermissionRequest hook when a permission prompt
    appears.  Schedules a delayed notification — if the user approves in
    terminal before the delay elapses, no notification is sent.
    """
    session_id = request.path_params["session_id"]
    data = await request.json()
    tool_desc = data.get("tool_desc", "")

    # If there's already a pending prompt for this session, ignore the new one
    if session_id in _permission_pending:
        return JSONResponse({"status": "already_pending"})

    logger.info(f"Permission hook fired for {session_id}: {tool_desc}")

    _permission_pending[session_id] = {
        "session_id": session_id,
        "tool_desc": tool_desc,
        "allow_all_label": "Allow all (session)",
        "ts": time.time(),
    }

    task = asyncio.create_task(_delayed_permission_notify(session_id, tool_desc))
    _permission_notify_tasks[session_id] = task

    return JSONResponse({"status": "pending"})


async def _resolve_permission_async(session_id: str, action: str, resolver: str):
    """Resolve a pending permission prompt — inject PTY input and publish event."""
    info = _permission_pending.pop(session_id, None)
    if not info:
        return

    # Cancel delayed notification if it hasn't fired yet
    task = _permission_notify_tasks.pop(session_id, None)
    if task and not task.done():
        task.cancel()

    # Map action to PTY keypress
    key = {"approve": "1", "allow_all": "2", "deny": "3"}.get(action, "3")
    await _inject_to_session(session_id, key)

    await event_bus.publish(
        PermissionResolvedEvent(
            session_id=session_id,
            action=action,
            resolver=resolver,
        )
    )
    logger.info(f"Permission {action} for session {session_id} (resolver={resolver})")


async def _clear_permission_on_activity(event):
    """Clear pending permission when new activity arrives (user approved in terminal)."""
    session_id = getattr(event, "session_id", "")
    if not session_id or session_id not in _permission_pending:
        return
    # Don't auto-clear if notification already sent — user may be clicking Approve
    info = _permission_pending.get(session_id, {})
    if info.get("notified"):
        return
    # A new request or tool use means the user approved the permission in terminal
    if isinstance(event, (RequestEvent, ToolUseEvent)):
        logger.debug(f"Clearing pending permission for {session_id} — new activity detected")
        _permission_pending.pop(session_id, None)
        task = _permission_notify_tasks.pop(session_id, None)
        if task and not task.done():
            task.cancel()


def resolve_permission(session_id: str, action: str, resolver: str):
    """Thread-safe: resolve a pending permission prompt from any surface."""
    if session_id not in _permission_pending:
        return
    if _event_loop:
        asyncio.run_coroutine_threadsafe(
            _resolve_permission_async(session_id, action, resolver),
            _event_loop,
        )


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
    conversation_context = data.get("conversation_context", [])

    # Record activity for this session
    record_session_activity(session_id)

    # Track this agent as monitored via gate API
    if agent and agent != "unknown":
        _gate_agents.add(agent)

    if not _gate_chain:
        return JSONResponse({"action": "ALLOW", "reason": "No gate chain configured"})

    tool_use_id = f"ext-{uuid.uuid4().hex[:12]}"

    # Resolve script contents for Bash/exec tool calls
    from cross.proxy import _resolve_scripts_for_tool

    script_contents = _resolve_scripts_for_tool(tool_name, tool_input, cwd=cwd)

    # Per-session recent_tools tracking for external agents
    max_tools = max(settings.llm_gate_context_tools, 1)
    if session_id and session_id not in _gate_recent_tools:
        _gate_recent_tools[session_id] = deque(maxlen=max_tools)
    recent_tools = list(_gate_recent_tools[session_id]) if session_id else []

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
        conversation_context=conversation_context,
        recent_tools=recent_tools,
    )

    result = await _gate_chain.evaluate(gate_request)

    # Record for future context
    if session_id:
        _gate_recent_tools[session_id].append({"name": tool_name, "input": tool_input})

    # Emit conversation context events for sentinel visibility
    if conversation_context:
        # Find last user and last assistant turns
        last_user = None
        last_assistant = None
        for turn in reversed(conversation_context):
            if turn.get("role") == "user" and last_user is None:
                last_user = turn.get("text", "")
            elif turn.get("role") == "assistant" and last_assistant is None:
                last_assistant = turn.get("text", "")
            if last_user is not None and last_assistant is not None:
                break
        if last_user:
            await event_bus.publish(
                RequestEvent(
                    method="POST",
                    path="/cross/api/gate",
                    last_message_role="user",
                    last_message_preview=last_user[:200],
                    agent=agent,
                )
            )
        if last_assistant:
            await event_bus.publish(TextEvent(text=last_assistant[:300]))

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
            recent_tools=recent_tools,
            user_intent=user_intent,
            conversation_context=conversation_context,
            eval_system_prompt=result.eval_system_prompt,
            eval_user_message=result.eval_user_message,
            eval_response_text=result.eval_response_text,
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


async def settings_page(request: Request) -> Response:
    """GET /cross/settings — serves the settings page HTML."""
    return Response(content=SETTINGS_HTML, media_type="text/html")


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


async def api_pending_permissions(request: Request) -> JSONResponse:
    """GET /cross/api/pending-permissions — pending permission prompts as JSON."""
    return JSONResponse(list(_permission_pending.values()))


async def api_resolve_permission(request: Request) -> JSONResponse:
    """POST /cross/api/permission/{session_id}/resolve — approve, deny, or allow-all."""
    session_id = request.path_params["session_id"]
    data = await request.json()
    action = data.get("action", "deny")  # "approve", "allow_all", "deny"
    username = data.get("username", "dashboard")

    if session_id not in _permission_pending:
        return JSONResponse(
            {"status": "not_found", "message": "No pending permission for this session"},
            status_code=404,
        )

    await _resolve_permission_async(session_id, action, f"dashboard (@{username})")
    return JSONResponse({"status": "ok", "session_id": session_id, "action": action})


async def api_halted_sessions(request: Request) -> JSONResponse:
    """GET /cross/api/halted-sessions — list halted sessions."""
    from cross.proxy import get_halted_sessions

    return JSONResponse(get_halted_sessions())


async def api_unhalt_session(request: Request) -> JSONResponse:
    """POST /cross/api/halted-sessions/{session_id}/resolve — un-halt a session."""
    from cross.proxy import clear_sentinel_halt

    session_id = request.path_params["session_id"]
    cleared = clear_sentinel_halt(session_id)
    if cleared:
        return JSONResponse({"status": "ok", "session_id": session_id})
    return JSONResponse({"status": "not_found", "message": "Session not halted"}, status_code=404)


async def api_get_instructions(request: Request) -> JSONResponse:
    """GET /cross/api/instructions — return current custom instructions."""
    content = _custom_instructions.content if _custom_instructions else ""
    return JSONResponse({"content": content})


async def api_put_instructions(request: Request) -> JSONResponse:
    """PUT /cross/api/instructions — update custom instructions."""
    if not _custom_instructions:
        return JSONResponse({"error": "Custom instructions not initialized"}, status_code=500)
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    content = data.get("content", "")
    if not isinstance(content, str):
        return JSONResponse({"error": "content must be a string"}, status_code=400)
    _custom_instructions.save(content)
    return JSONResponse({"status": "ok", "length": len(content)})


async def api_conversation_message(request: Request) -> JSONResponse:
    """POST /cross/api/conversations/{conversation_id}/message — send a message, get LLM reply."""
    conv_id = request.path_params["conversation_id"]
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)
    message = data.get("message", "").strip()
    if not message:
        return JSONResponse({"error": "message is required"}, status_code=400)
    if not _conversation_store:
        return JSONResponse({"error": "Conversations not initialized"}, status_code=500)

    # Lazily register context from event_data if missing (e.g., after restart)
    if not _conversation_store.has_context(conv_id):
        ev = data.get("event_data")
        if ev:
            sys_prompt = ev.get("eval_system_prompt", "")
            resp_text = ev.get("eval_response_text", "")
            if sys_prompt and resp_text:
                c_type = "gate" if conv_id.startswith("gate:") else "sentinel"
                _conversation_store.seed_conversation(
                    conversation_id=conv_id,
                    conv_type=c_type,
                    system_prompt=sys_prompt,
                    user_message=ev.get("eval_user_message", ""),
                    response_text=resp_text,
                )
            elif conv_id.startswith("gate:"):
                _conversation_store.register_gate_context(
                    tool_use_id=ev.get("tool_use_id", conv_id.split(":", 1)[1]),
                    tool_name=ev.get("tool_name", "unknown"),
                    tool_input=ev.get("tool_input"),
                    action=ev.get("action", ""),
                    reason=ev.get("reason", ""),
                    rule_id=ev.get("rule_id", ""),
                    evaluator=ev.get("evaluator", ""),
                    script_contents=ev.get("script_contents"),
                    recent_tools=ev.get("recent_tools"),
                    user_intent=ev.get("user_intent", ""),
                    conversation_context=ev.get("conversation_context"),
                )
            else:
                return JSONResponse({"error": "Unknown conversation"}, status_code=404)
        else:
            return JSONResponse({"error": "Unknown conversation"}, status_code=404)

    reply = await _conversation_store.send_message(conv_id, message)
    if reply is None:
        return JSONResponse({"error": "LLM failed to respond"}, status_code=502)
    return JSONResponse({"conversation_id": conv_id, "reply": reply})


async def api_conversation_history(request: Request) -> JSONResponse:
    """GET /cross/api/conversations/{conversation_id} — return message history."""
    conv_id = request.path_params["conversation_id"]
    if not _conversation_store:
        return JSONResponse({"error": "Conversations not initialized"}, status_code=500)
    messages = _conversation_store.get_messages(conv_id)
    return JSONResponse({"conversation_id": conv_id, "messages": messages})


async def api_dashboard_ws(ws: WebSocket):
    """WS /cross/api/ws — real-time event stream to dashboard clients."""
    if _dashboard:
        await _dashboard.ws_handler(ws)
    else:
        await ws.close()


# --- Proxy routes ---


async def _proxy_handler(request: Request) -> Response:
    """Forward to the actual proxy logic."""
    # Reject WebSocket upgrades early — Codex falls back to HTTP faster
    if request.headers.get("upgrade", "").lower() == "websocket":
        return Response(status_code=426, content=b"WebSocket not supported")

    from cross.proxy import handle_proxy_request

    session_id = request.path_params.get("session_id", "")
    return await handle_proxy_request(request, event_bus, gate_chain=_gate_chain, session_id=session_id)


# --- App lifecycle ---


_sentinel = None  # LLMSentinel instance, if configured


def _persist_state() -> None:
    """Save current daemon state to disk (called on session changes and shutdown)."""
    from cross.state import save_state

    sentinel_events = None
    if _sentinel is not None:
        try:
            sentinel_events = _sentinel.get_events()
        except Exception:
            pass

    try:
        save_state(
            sessions=_sessions,
            project_cwds=_project_cwds,
            gate_agents=_gate_agents,
            sentinel_events=sentinel_events,
        )
    except Exception as e:
        logger.warning(f"Failed to persist state: {e}")


async def on_startup():
    global _slack, _email, _gate_chain, _sentinel, _dashboard, _event_loop, _custom_instructions, _conversation_store

    _event_loop = asyncio.get_running_loop()

    # Initialize custom instructions (hot-reloads on each access)
    _custom_instructions = CustomInstructions(path=settings.custom_instructions_file)
    if _custom_instructions.content:
        logger.info(f"Custom instructions loaded ({len(_custom_instructions.content)} chars)")

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
    event_bus.subscribe(_clear_permission_on_activity)
    logger.info("Dashboard active at /cross/dashboard")

    # Pre-populate monitored agents from hook configuration
    hooked = _detect_hooked_agents()
    _gate_agents.update(hooked)
    if hooked:
        logger.info(f"Detected hooked agents: {', '.join(hooked)}")

    # Restore persisted state from previous daemon run
    from cross.state import load_state

    _restored_state = load_state()
    _sessions.update(_restored_state["sessions"])
    _project_cwds.update(_restored_state["project_cwds"])
    _gate_agents.update(_restored_state["gate_agents"])
    _restored_sentinel_events = _restored_state["sentinel_events"]
    if _sessions:
        logger.info(f"Restored {len(_sessions)} sessions from previous run")

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
            if resolve_api_key(llm_config) or llm_config.provider in _CLI_PROVIDERS:
                # Build backup config if configured
                gate_backup = None
                if settings.llm_gate_backup_model:
                    gate_backup = LLMConfig(
                        model=settings.llm_gate_backup_model,
                        api_key=settings.llm_gate_backup_api_key,
                        base_url=settings.llm_gate_backup_base_url,
                        temperature=settings.llm_gate_temperature,
                        max_tokens=settings.llm_gate_max_tokens,
                        reasoning=settings.llm_gate_reasoning,
                    )
                    if not (resolve_api_key(gate_backup) or gate_backup.provider in _CLI_PROVIDERS):
                        logger.info("Gate backup model configured but no API key — backup disabled")
                        gate_backup = None

                review_gate = LLMReviewGate(
                    config=llm_config,
                    timeout_ms=settings.llm_gate_timeout_ms,
                    justification=settings.llm_gate_justification,
                    get_custom_instructions=lambda: _custom_instructions.content if _custom_instructions else "",
                    backup_config=gate_backup,
                )
                model_name = settings.llm_gate_model
                backup_desc = f", backup={settings.llm_gate_backup_model}" if gate_backup else ""
                logger.info(f"LLM review gate active (model={model_name}{backup_desc})")
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
        if resolve_api_key(sentinel_config) or sentinel_config.provider in _CLI_PROVIDERS:
            # Build backup config if configured
            sentinel_backup = None
            if settings.llm_sentinel_backup_model:
                sentinel_backup = LLMConfig(
                    model=settings.llm_sentinel_backup_model,
                    api_key=settings.llm_sentinel_backup_api_key,
                    base_url=settings.llm_sentinel_backup_base_url,
                    temperature=settings.llm_sentinel_temperature,
                    max_tokens=settings.llm_sentinel_max_tokens,
                    reasoning=settings.llm_sentinel_reasoning,
                )
                if not (resolve_api_key(sentinel_backup) or sentinel_backup.provider in _CLI_PROVIDERS):
                    logger.info("Sentinel backup model configured but no API key — backup disabled")
                    sentinel_backup = None

            _sentinel = LLMSentinel(
                config=sentinel_config,
                event_bus=event_bus,
                interval_seconds=settings.llm_sentinel_interval_seconds,
                max_events=settings.llm_sentinel_max_events,
                get_custom_instructions=lambda: _custom_instructions.content if _custom_instructions else "",
                backup_config=sentinel_backup,
                seed_events=_restored_sentinel_events,
            )
            event_bus.subscribe(_sentinel.observe)
            _sentinel.start()
            backup_desc = f", backup={settings.llm_sentinel_backup_model}" if sentinel_backup else ""
            logger.info(
                f"LLM sentinel active (model={settings.llm_sentinel_model}{backup_desc}, "
                f"interval={settings.llm_sentinel_interval_seconds}s)"
            )
        else:
            logger.info("LLM sentinel enabled but no API key available — sentinel inactive")

    # Set up conversation store (for inline follow-up conversations with reviewers)
    from cross.conversations import ConversationStore

    gate_llm_cfg = None
    sentinel_llm_cfg = None
    if settings.llm_gate_enabled:
        from cross.llm import LLMConfig as _LLMConfig

        gate_llm_cfg = _LLMConfig(
            model=settings.llm_gate_model,
            api_key=settings.llm_gate_api_key,
            base_url=settings.llm_gate_base_url,
            temperature=settings.llm_gate_temperature,
            max_tokens=settings.llm_gate_max_tokens,
            reasoning=settings.llm_gate_reasoning,
        )
    if settings.llm_sentinel_enabled:
        from cross.llm import LLMConfig as _LLMConfig

        sentinel_llm_cfg = _LLMConfig(
            model=settings.llm_sentinel_model,
            api_key=settings.llm_sentinel_api_key,
            base_url=settings.llm_sentinel_base_url,
            temperature=settings.llm_sentinel_temperature,
            max_tokens=settings.llm_sentinel_max_tokens,
            reasoning=settings.llm_sentinel_reasoning,
        )
    _conversation_store = ConversationStore(
        gate_llm_config=gate_llm_cfg,
        sentinel_llm_config=sentinel_llm_cfg,
        get_custom_instructions=lambda: _custom_instructions.content if _custom_instructions else "",
    )
    _dashboard.set_conversation_store(_conversation_store)

    # Register Email plugin
    if settings.email_from and settings.email_to:
        from cross.plugins.email import EmailPlugin
        from cross.proxy import resolve_gate_approval as _resolve_gate_email

        _email = EmailPlugin(
            inject_callback=_inject_to_session,
            resolve_approval_callback=_resolve_gate_email,
            resolve_permission_callback=resolve_permission,
            event_loop=asyncio.get_running_loop(),
            conversation_store=_conversation_store,
        )
        try:
            _email.start()
            event_bus.subscribe(_email.handle_event)
            logger.info("Email relay active")
        except Exception as e:
            logger.warning(f"Email failed to start: {e}")
            _email = None

    # Register Slack plugin
    if settings.slack_bot_token and settings.slack_app_token:
        from cross.plugins.slack import SlackPlugin
        from cross.proxy import resolve_gate_approval

        _slack = SlackPlugin(
            inject_callback=_inject_to_session,
            spawn_callback=_spawn_session,
            resolve_approval_callback=resolve_gate_approval,
            resolve_permission_callback=resolve_permission,
            event_loop=asyncio.get_running_loop(),
            conversation_store=_conversation_store,
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
    # Persist state before stopping services so sentinel events are captured
    _persist_state()

    if _sentinel:
        _sentinel.stop()
    if _slack:
        _slack.stop()
    if _email:
        _email.stop()
    # Clean up LLM httpx client
    from cross.llm import close_client

    await close_client()


# --- Build the combined app ---

# Local API routes
_api_routes = [
    Route("/cross/sessions", api_register_session, methods=["POST"]),
    Route("/cross/sessions/{session_id}/end", api_end_session, methods=["POST"]),
    Route("/cross/sessions/{session_id}/permission", api_permission_hook, methods=["POST"]),
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
    Route("/cross/settings", settings_page, methods=["GET"]),
    Route("/cross/api/status", api_status, methods=["GET"]),
    Route("/cross/api/events", api_events, methods=["GET"]),
    Route("/cross/api/pending", api_pending, methods=["GET"]),
    Route("/cross/api/pending/{tool_use_id}/resolve", api_resolve_pending, methods=["POST"]),
    Route("/cross/api/pending-permissions", api_pending_permissions, methods=["GET"]),
    Route("/cross/api/permission/{session_id}/resolve", api_resolve_permission, methods=["POST"]),
    Route("/cross/api/halted-sessions", api_halted_sessions, methods=["GET"]),
    Route("/cross/api/halted-sessions/{session_id}/resolve", api_unhalt_session, methods=["POST"]),
    Route("/cross/api/instructions", api_get_instructions, methods=["GET"]),
    Route("/cross/api/instructions", api_put_instructions, methods=["PUT"]),
    Route("/cross/api/conversations/{conversation_id:path}/message", api_conversation_message, methods=["POST"]),
    Route("/cross/api/conversations/{conversation_id:path}", api_conversation_history, methods=["GET"]),
]


# WebSocket routes
async def _reject_ws_upgrade(ws: WebSocket):
    """Reject WebSocket upgrades on proxy routes (Codex Responses API)."""
    await ws.close(code=1002, reason="WebSocket proxy not supported")


_ws_routes = [
    WebSocketRoute("/cross/sessions/{session_id}/io", api_session_ws),
    WebSocketRoute("/cross/api/ws", api_dashboard_ws),
    WebSocketRoute("/s/{session_id}/{path:path}", _reject_ws_upgrade),
]

# Proxy routes — session-prefixed route first, then catch-all for backward compat
_PROXY_METHODS = ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]
_proxy_routes = [
    Route("/s/{session_id}/{path:path}", _proxy_handler, methods=_PROXY_METHODS),
    Route("/{path:path}", _proxy_handler, methods=_PROXY_METHODS),
]

app = Starlette(
    routes=_api_routes + _dashboard_routes + _ws_routes + _proxy_routes,
    on_startup=[on_startup],
    on_shutdown=[on_shutdown],
)
