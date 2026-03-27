"""Web dashboard plugin — local approval surface and live event feed.

Subscribes to EventBus events, tracks pending escalations, and broadcasts
to WebSocket clients. Event storage is handled by EventStore.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from starlette.websockets import WebSocket, WebSocketDisconnect

from cross.event_store import EventStore, event_to_dict
from cross.events import (
    CrossEvent,
    GateDecisionEvent,
    PermissionPromptEvent,
    PermissionResolvedEvent,
    SentinelReviewEvent,
)

logger = logging.getLogger("cross.plugins.dashboard")


class DashboardPlugin:
    """Tracks pending escalations and broadcasts events to WebSocket clients."""

    def __init__(self, event_store: EventStore, resolve_approval_callback=None):
        self._store = event_store
        # Pending escalations: tool_use_id -> event dict
        self._pending: dict[str, dict[str, Any]] = {}
        # Pending permission prompts: session_id -> event dict
        self._permission_pending: dict[str, dict[str, Any]] = {}
        # Connected WebSocket clients
        self._ws_clients: set[WebSocket] = set()
        # Callback to resolve gate approvals in the proxy
        self._resolve_approval_callback = resolve_approval_callback
        # Conversation store (set after init via set_conversation_store)
        self._conversation_store = None

    def set_conversation_store(self, store) -> None:
        """Attach the conversation store (called from daemon after both are initialized)."""
        self._conversation_store = store

    async def handle_event(self, event: CrossEvent):
        """EventBus handler — track escalations/permissions and broadcast to WS clients."""
        event_dict = event_to_dict(event)

        # Track pending escalations
        if isinstance(event, GateDecisionEvent):
            if event.action == "escalate":
                self._pending[event.tool_use_id] = event_dict
            elif event.action in ("allow", "block", "halt_session") and event.tool_use_id in self._pending:
                # Escalation resolved (by Slack, dashboard, or timeout)
                del self._pending[event.tool_use_id]

            # Register conversation context for non-trivial gate decisions
            if self._conversation_store and (
                event.action in ("block", "alert", "escalate", "halt_session")
                or (event.evaluator != "denylist" and event.action == "allow")
            ):
                conv_id = f"gate:{event.tool_use_id}"
                # Prefer seeding with original eval conversation when available
                if event.eval_system_prompt and event.eval_response_text:
                    self._conversation_store.seed_conversation(
                        conversation_id=conv_id,
                        conv_type="gate",
                        system_prompt=event.eval_system_prompt,
                        user_message=event.eval_user_message,
                        response_text=event.eval_response_text,
                    )
                else:
                    # Fallback: reconstruct from event metadata
                    self._conversation_store.register_gate_context(
                        tool_use_id=event.tool_use_id,
                        tool_name=event.tool_name,
                        tool_input=event.tool_input,
                        action=event.action,
                        reason=event.reason,
                        rule_id=event.rule_id,
                        evaluator=event.evaluator,
                        script_contents=event.script_contents,
                        recent_tools=event.recent_tools,
                        user_intent=event.user_intent,
                        conversation_context=event.conversation_context,
                    )

        # Register sentinel review conversation context
        elif isinstance(event, SentinelReviewEvent):
            if self._conversation_store and event.review_id:
                # Sentinel reviews always produce eval data (they always go through LLM)
                if event.eval_system_prompt and event.eval_response_text:
                    self._conversation_store.seed_conversation(
                        conversation_id=f"sentinel:{event.review_id}",
                        conv_type="sentinel",
                        system_prompt=event.eval_system_prompt,
                        user_message=event.eval_user_message,
                        response_text=event.eval_response_text,
                    )

        # Track pending permission prompts
        elif isinstance(event, PermissionPromptEvent):
            self._permission_pending[event.session_id] = event_dict

        elif isinstance(event, PermissionResolvedEvent):
            self._permission_pending.pop(event.session_id, None)

        # Broadcast to connected WebSocket clients
        await self._broadcast(event_dict)

    async def _broadcast(self, event_dict: dict[str, Any]):
        """Send event to all connected WebSocket clients."""
        if not self._ws_clients:
            return
        msg = json.dumps(event_dict)
        disconnected = set()
        for ws in self._ws_clients:
            try:
                await ws.send_text(msg)
            except Exception:
                disconnected.add(ws)
        self._ws_clients -= disconnected

    def get_events(self) -> list[dict[str, Any]]:
        """Return recent events as a list (oldest first)."""
        return self._store.get_events()

    def get_pending(self) -> list[dict[str, Any]]:
        """Return pending escalations as a list."""
        return list(self._pending.values())

    def get_pending_permissions(self) -> list[dict[str, Any]]:
        """Return pending permission prompts as a list."""
        return list(self._permission_pending.values())

    def resolve(self, tool_use_id: str, approved: bool, username: str = "dashboard"):
        """Resolve a pending escalation (approve or deny)."""
        self._pending.pop(tool_use_id, None)
        if self._resolve_approval_callback:
            self._resolve_approval_callback(tool_use_id, approved, username)

    async def ws_handler(self, ws: WebSocket):
        """Handle a dashboard WebSocket connection."""
        await ws.accept()
        self._ws_clients.add(ws)
        logger.info(f"Dashboard WS connected ({len(self._ws_clients)} clients)")
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                    if msg.get("type") == "conversation_message":
                        asyncio.create_task(self._handle_conv_message(ws, msg))
                except (json.JSONDecodeError, ValueError):
                    pass  # keepalive ping or malformed
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            self._ws_clients.discard(ws)
            logger.info(f"Dashboard WS disconnected ({len(self._ws_clients)} clients)")

    async def _handle_conv_message(self, ws: WebSocket, msg: dict):
        """Handle a conversation message from a dashboard WS client."""
        conv_id = msg.get("conversation_id", "")
        message = msg.get("message", "").strip()
        if not conv_id or not message or not self._conversation_store:
            return

        # Lazily register context if missing (e.g., after daemon restart).
        # The dashboard sends the original event data so we can reconstruct.
        if not self._conversation_store.has_context(conv_id):
            ev = msg.get("event_data")
            if ev:
                # Prefer seeding with original eval data when available
                sys_prompt = ev.get("eval_system_prompt", "")
                resp_text = ev.get("eval_response_text", "")
                if sys_prompt and resp_text:
                    c_type = "gate" if conv_id.startswith("gate:") else "sentinel"
                    self._conversation_store.seed_conversation(
                        conversation_id=conv_id,
                        conv_type=c_type,
                        system_prompt=sys_prompt,
                        user_message=ev.get("eval_user_message", ""),
                        response_text=resp_text,
                    )
                elif conv_id.startswith("gate:"):
                    self._conversation_store.register_gate_context(
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
                # Sentinel without eval data — no conversation available

        # Send typing indicator
        try:
            await ws.send_text(
                json.dumps(
                    {
                        "type": "conversation_typing",
                        "conversation_id": conv_id,
                    }
                )
            )
        except Exception:
            return

        reply = await self._conversation_store.send_message(conv_id, message)

        try:
            await ws.send_text(
                json.dumps(
                    {
                        "type": "conversation_reply",
                        "conversation_id": conv_id,
                        "reply": reply or "Sorry, I couldn't generate a response.",
                        "ts": time.time(),
                    }
                )
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Dashboard HTML (inline single-page app)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cross dashboard</title>
<link rel="icon" type="image/svg+xml"
  href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg'
  viewBox='0 0 24 24'><path d='M12 2v20M2 12h20'
  stroke='white' stroke-width='3' stroke-linecap='round'/></svg>">
<style>
  :root {
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --text-dim: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --red: #f85149;
    --orange: #d29922;
    --yellow: #e3b341;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }
  header {
    border-bottom: 1px solid var(--border);
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  header h1 {
    font-size: 20px;
    font-weight: 600;
    letter-spacing: -0.5px;
  }
  header .header-right {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  header .notif-btn {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 12px;
    color: var(--text-dim);
    cursor: pointer;
    transition: opacity 0.15s;
  }
  header .notif-btn:hover { opacity: 0.85; }
  header .notif-btn.granted { color: var(--green); border-color: var(--green); }
  header .notif-btn.denied { color: var(--red); border-color: var(--red); cursor: not-allowed; }
  main {
    max-width: 960px;
    margin: 0 auto;
    padding: 24px;
  }
  section { margin-bottom: 32px; }
  section h2 {
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-dim);
    margin-bottom: 12px;
  }
  .empty {
    color: var(--text-dim);
    font-style: italic;
    padding: 16px 0;
  }

  /* Pending approvals */
  .pending-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--orange);
    border-radius: 6px;
    padding: 16px;
    margin-bottom: 12px;
  }
  .pending-card .tool-name {
    font-weight: 600;
    color: var(--accent);
    font-size: 15px;
  }
  .pending-card .reason {
    color: var(--text-dim);
    margin: 6px 0;
    font-size: 13px;
  }
  .pending-card .input-preview {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 8px 12px;
    font-family: "SF Mono", "Fira Code", monospace;
    font-size: 12px;
    max-height: 120px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-break: break-all;
    margin: 8px 0;
  }
  .pending-card .actions {
    display: flex;
    gap: 8px;
    margin-top: 12px;
  }
  .btn {
    border: none;
    border-radius: 6px;
    padding: 6px 16px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.15s;
  }
  .btn:hover { opacity: 0.85; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }
  .btn-approve { background: var(--green); color: #fff; }
  .btn-allow-all { background: var(--accent); color: #fff; }
  .btn-deny { background: var(--red); color: #fff; }

  /* Permission prompt cards */
  .permission-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--yellow);
    border-radius: 6px;
    padding: 16px;
    margin-bottom: 12px;
  }
  .permission-card .tool-name {
    font-weight: 600;
    color: var(--yellow);
    font-size: 15px;
  }
  .permission-card .session-id {
    color: var(--text-dim);
    font-size: 12px;
    margin-top: 2px;
  }
  .permission-card .actions {
    display: flex;
    gap: 8px;
    margin-top: 12px;
  }

  /* Halted session cards */
  .halted-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--red);
    border-radius: 6px;
    padding: 16px;
    margin-bottom: 12px;
  }
  .halted-card .halted-label {
    font-weight: 600;
    color: var(--red);
    font-size: 15px;
  }
  .halted-card .session-id {
    color: var(--text-dim);
    font-size: 12px;
    margin-top: 2px;
  }
  .halted-card .reason {
    color: var(--text-dim);
    margin: 6px 0;
    font-size: 13px;
  }
  .halted-card .actions {
    display: flex;
    gap: 8px;
    margin-top: 12px;
  }
  .btn-unhalt { background: var(--green); color: #fff; }

  /* Agent status bar */
  .agent-status {
    display: flex;
    gap: 16px;
    align-items: center;
    flex-wrap: wrap;
    padding: 8px 0;
    font-size: 13px;
  }
  .agent-status .status-summary {
    color: var(--text-dim);
    font-weight: 600;
  }
  .agent-status .agent-chip {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 12px;
    border: 1px solid var(--border);
  }
  .agent-chip.monitored { color: var(--green); border-color: var(--green); }
  .agent-chip.unmonitored { color: var(--text-dim); border-color: var(--border); }
  .agent-chip .dot {
    width: 6px; height: 6px; border-radius: 50%;
    display: inline-block;
  }
  .agent-chip.monitored .dot { background: var(--green); }
  .agent-chip.unmonitored .dot { background: var(--text-dim); }
  .agent-chip.active .dot {
    animation: pulse 2s ease-in-out infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 var(--green); }
    50% { opacity: 0.7; box-shadow: 0 0 4px 2px var(--green); }
  }
  .agent-chip.stopped .dot { opacity: 0.5; }

  /* Event feed */
  .event-feed {
    display: flex;
    flex-direction: column;
    gap: 4px;
  }
  .event-row {
    display: flex;
    gap: 12px;
    padding: 6px 12px;
    border-radius: 4px;
    font-size: 13px;
    align-items: baseline;
    cursor: pointer;
  }
  .event-row:hover { background: var(--surface); }
  .event-row .time {
    color: var(--text-dim);
    font-family: monospace;
    font-size: 12px;
    flex-shrink: 0;
    width: 72px;
  }
  .event-row .badge {
    display: inline-block;
    padding: 1px 6px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    flex-shrink: 0;
    min-width: 80px;
    text-align: center;
  }
  .badge-tool_use { background: #2d333b; color: var(--text); }
  .badge-gate_allow { background: #1a3a2a; color: var(--green); }
  .badge-gate_block { background: #3d1f1f; color: var(--red); }
  .badge-gate_alert { background: #3d2f1f; color: var(--orange); }
  .badge-gate_escalate { background: #3d2f1f; color: var(--yellow); }
  .badge-sentinel { background: #2a1f3d; color: #bc8cff; }
  .badge-request { background: #1f2a3d; color: var(--accent); }
  .badge-text { background: #2d333b; color: var(--text); }
  .event-row .agent {
    color: var(--text-dim);
    font-size: 12px;
    flex-shrink: 0;
    width: 80px;
  }
  .event-row .detail {
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .event-row.expanded {
    align-items: flex-start;
    user-select: text;
    cursor: default;
  }
  .event-row.expanded .detail {
    white-space: pre-wrap;
    overflow: visible;
    text-overflow: unset;
    word-break: break-word;
  }
  .event-row.hidden { display: none; }
  .event-row.conversable { cursor: pointer; }
  .event-row.expanded { flex-wrap: wrap; }

  /* Conversation chat */
  .conv-chat {
    width: 100%;
    margin-top: 10px;
    border-top: 1px solid var(--border);
    padding-top: 10px;
  }
  .conv-messages {
    max-height: 300px;
    overflow-y: auto;
    margin-bottom: 8px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .conv-msg {
    padding: 8px 12px;
    border-radius: 10px;
    font-size: 13px;
    max-width: 85%;
    line-height: 1.4;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .conv-msg.user {
    background: #1f3d5c;
    align-self: flex-end;
  }
  .conv-msg.assistant {
    background: var(--surface);
    border: 1px solid var(--border);
    align-self: flex-start;
  }
  .conv-input-row {
    display: flex;
    gap: 8px;
  }
  .conv-input {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 13px;
    color: var(--text);
    font-family: inherit;
  }
  .conv-input:focus { border-color: var(--accent); outline: none; }
  .conv-send {
    background: var(--accent);
    color: white;
    border: none;
    border-radius: 6px;
    padding: 6px 16px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
  }
  .conv-send:disabled { opacity: 0.5; cursor: default; }
  .conv-send:hover:not(:disabled) { opacity: 0.9; }
  .conv-typing {
    color: var(--text-dim);
    font-size: 12px;
    font-style: italic;
    padding: 4px 0;
  }

  /* Filter bar */
  .filter-bar {
    display: flex;
    gap: 8px;
    margin-bottom: 12px;
    flex-wrap: wrap;
    align-items: center;
  }
  .filter-bar input {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 12px;
    color: var(--text);
    outline: none;
    width: 200px;
  }
  .filter-bar input:focus { border-color: var(--accent); }
  .filter-bar input::placeholder { color: var(--text-dim); }
  .filter-pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 600;
    cursor: pointer;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text-dim);
    transition: all 0.15s;
    user-select: none;
  }
  .filter-pill.active {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }
  .filter-pill:hover { opacity: 0.85; }
  .filter-sep {
    width: 1px;
    height: 18px;
    background: var(--border);
    margin: 0 2px;
  }

  /* Notification modal */
  .notif-modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }
  .notif-modal {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 32px;
    max-width: 400px;
    text-align: center;
  }
  .notif-modal h3 {
    font-size: 18px;
    margin-bottom: 12px;
  }
  .notif-modal p {
    color: var(--text-dim);
    font-size: 14px;
    line-height: 1.5;
    margin-bottom: 20px;
  }
  .notif-modal .btn-row {
    display: flex;
    gap: 12px;
    justify-content: center;
  }
  .notif-modal .btn-enable {
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 6px;
    padding: 8px 24px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
  }
  .notif-modal .btn-enable:hover { opacity: 0.85; }
  .notif-modal .btn-skip {
    background: transparent;
    color: var(--text-dim);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 24px;
    font-size: 14px;
    cursor: pointer;
  }
  .notif-modal .btn-skip:hover { opacity: 0.85; }

  header .settings-link {
    color: var(--text-dim);
    text-decoration: none;
    transition: color 0.15s;
    display: flex;
    align-items: center;
  }
  header .settings-link:hover { color: var(--text); }
</style>
</head>
<body>
<div class="notif-modal-overlay" id="notif-modal" style="display:none">
  <div class="notif-modal">
    <h3>Enable notifications</h3>
    <p>Get notified when an agent needs approval or the sentinel flags something.</p>
    <div class="btn-row">
      <button class="btn-enable" id="notif-modal-enable">Enable</button>
      <button class="btn-skip" id="notif-modal-skip">Not now</button>
    </div>
  </div>
</div>
<header>
  <h1><svg width="24" height="24" viewBox="0 0 24 24" fill="none"
    style="display:block"><path d="M12 2v20M2 12h20"
    stroke="white" stroke-width="3" stroke-linecap="round"/></svg></h1>
  <div class="header-right">
    <button class="notif-btn" id="notif-btn" style="display:none"
      onclick="requestNotifPermission()">Enable notifications</button>
    <a class="settings-link" href="/cross/settings" title="Settings"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg></a>
  </div>
</header>
<main>
  <section id="status-section">
    <h2>Agents</h2>
    <div id="agent-status" class="agent-status"></div>
  </section>
  <section id="halted-section" style="display:none">
    <h2>Halted Sessions</h2>
    <div id="halted-list"></div>
  </section>
  <section id="pending-section">
    <h2>Pending Approvals</h2>
    <div id="pending-list"><p class="empty">No pending approvals</p></div>
  </section>
  <section>
    <h2>Live Event Feed</h2>
    <div class="filter-bar" id="filter-bar">
      <input type="text" id="filter-search" placeholder="Search events...">
      <div class="filter-sep"></div>
      <span id="agent-filters"></span>
      <div class="filter-sep"></div>
      <span class="filter-pill active" data-filter="type" data-value="all">all</span>
      <span class="filter-pill" data-filter="type" data-value="request">request</span>
      <span class="filter-pill" data-filter="type" data-value="tool call">tool call</span>
      <span class="filter-pill" data-filter="type" data-value="response">response</span>
      <span class="filter-pill" data-filter="type" data-value="gate">gate</span>
      <span class="filter-pill" data-filter="type" data-value="sentinel">sentinel</span>
    </div>
    <div id="event-feed" class="event-feed"><p class="empty" id="feed-empty">Waiting for events...</p></div>
  </section>
</main>
<script>
(function() {
  const pendingList = document.getElementById("pending-list");
  const eventFeed = document.getElementById("event-feed");
  const feedEmpty = document.getElementById("feed-empty");

  const MAX_FEED = 200;
  let pendingMap = {};
  let permissionMap = {};
  let haltedMap = {};
  let ws = null;
  const haltedSection = document.getElementById("halted-section");
  const haltedList = document.getElementById("halted-list");

  function formatTime(ts) {
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
  }

  function truncate(s, n) {
    if (!s) return "";
    s = String(s);
    return s.length > n ? s.slice(0, n) + "..." : s;
  }

  const BASE_TITLE = "cross dashboard";

  function updateTabBadge(count) {
    document.title = count > 0 ? "(" + count + ") " + BASE_TITLE : BASE_TITLE;
    // Update or create a favicon badge
    updateFaviconBadge(count);
  }

  function updateFaviconBadge(count) {
    var canvas = document.createElement("canvas");
    canvas.width = 32;
    canvas.height = 32;
    var ctx = canvas.getContext("2d");
    // Base icon: filled circle
    ctx.fillStyle = "#30363d";
    ctx.beginPath();
    ctx.arc(16, 16, 15, 0, 2 * Math.PI);
    ctx.fill();
    ctx.strokeStyle = "#3fb950";
    ctx.lineWidth = 2;
    ctx.stroke();
    if (count > 0) {
      // Red badge circle
      ctx.fillStyle = "#f85149";
      ctx.beginPath();
      ctx.arc(24, 8, 8, 0, 2 * Math.PI);
      ctx.fill();
      // Badge number
      ctx.fillStyle = "#fff";
      ctx.font = "bold 10px sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(count > 9 ? "9+" : String(count), 24, 8);
    }
    var link = document.querySelector("link[rel='icon']");
    if (!link) {
      link = document.createElement("link");
      link.rel = "icon";
      document.head.appendChild(link);
    }
    link.href = canvas.toDataURL("image/png");
  }

  function renderPending() {
    const gateKeys = Object.keys(pendingMap);
    const permKeys = Object.keys(permissionMap);
    const totalPending = gateKeys.length + permKeys.length;
    updateTabBadge(totalPending);
    if (totalPending === 0) {
      pendingList.innerHTML = '<p class="empty">No pending approvals</p>';
      return;
    }
    let html = "";
    // Permission prompts first (more urgent — blocking terminal)
    for (const sid of permKeys) {
      const p = permissionMap[sid];
      const safeSid = escHtml(sid);
      const toolDesc = p.tool_desc || "unknown tool";
      const allowLabel = escHtml(p.allow_all_label || "Allow all (session)");
      html += '<div class="permission-card" data-sid="' + safeSid + '">'
        + '<div class="tool-name">Permission needed' + (toolDesc ? ' for ' + escHtml(toolDesc) : '') + '</div>'
        + '<div class="session-id">Session: ' + safeSid + '</div>'
        + '<div class="actions">'
        + '<button class="btn btn-approve" onclick="resolvePerm(&apos;'
        + safeSid + '&apos;, &apos;approve&apos;)">Approve</button>'
        + '<button class="btn btn-allow-all" onclick="resolvePerm(&apos;'
        + safeSid + '&apos;, &apos;allow_all&apos;)">' + allowLabel + '</button>'
        + '<button class="btn btn-deny" onclick="resolvePerm(&apos;'
        + safeSid + '&apos;, &apos;deny&apos;)">Deny</button>'
        + '</div></div>';
    }
    // Gate escalations
    for (const id of gateKeys) {
      const ev = pendingMap[id];
      const inputStr = ev.tool_input ? JSON.stringify(ev.tool_input, null, 2) : "";
      const safeId = escHtml(id);
      html += '<div class="pending-card" data-id="' + safeId + '">'
        + '<div class="tool-name">' + escHtml(ev.tool_name || "unknown") + '</div>'
        + '<div class="reason">' + escHtml(ev.reason || "") + '</div>'
        + (inputStr ? '<div class="input-preview">' + escHtml(truncate(inputStr, 500)) + '</div>' : '')
        + '<div class="actions">'
        + '<button class="btn btn-approve" onclick="resolveGate(&apos;' + safeId + '&apos;, true)">Approve</button>'
        + '<button class="btn btn-deny" onclick="resolveGate(&apos;' + safeId + '&apos;, false)">Deny</button>'
        + '</div></div>';
    }
    pendingList.innerHTML = html;
  }

  function escHtml(s) {
    const d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }

  function badgeClass(ev) {
    const t = ev.event_type;
    if (t === "ToolUseEvent") return "badge-tool_use";
    if (t === "GateDecisionEvent") return "badge-gate_" + (ev.action || "allow");
    if (t === "SentinelReviewEvent") return "badge-sentinel";
    if (t === "RequestEvent") return "badge-request";
    if (t === "TextEvent") return "badge-text";
    return "";
  }

  function badgeLabel(ev) {
    const t = ev.event_type;
    if (t === "ToolUseEvent") return "tool call";
    if (t === "GateDecisionEvent") return "gate:" + (ev.action || "?");
    if (t === "SentinelReviewEvent") return "sentinel";
    if (t === "RequestEvent") return "request";
    if (t === "TextEvent") return "response";
    return t.replace("Event", "").toLowerCase();
  }

  function detailText(ev) {
    const t = ev.event_type;
    if (t === "ToolUseEvent") {
      var name = ev.name || "unknown";
      var detail = "";
      if (ev.input && ev.input.command) detail = ev.input.command;
      else if (ev.input && ev.input.file_path) detail = ev.input.file_path;
      else if (ev.input && ev.input.pattern) detail = ev.input.pattern;
      else if (ev.input) {
        var s = JSON.stringify(ev.input);
        if (s !== "{}") detail = s;
      }
      return detail ? name + ": " + detail : name;
    }
    if (t === "GateDecisionEvent") return (ev.tool_name || "") + (ev.reason ? " — " + ev.reason : "");
    if (t === "SentinelReviewEvent") return ev.summary || ev.concerns || "";
    if (t === "RequestEvent") return ev.last_message_preview || (ev.model || "");
    if (t === "TextEvent") return ev.text || "";
    return "";
  }

  // Propagate agent label from request events to subsequent events
  let currentAgent = "";

  function agentLabel(ev) {
    if (ev.agent) { currentAgent = ev.agent; return ev.agent; }
    return currentAgent;
  }

  function addEventRow(ev) {
    if (feedEmpty) feedEmpty.remove();
    const full = detailText(ev);
    var agent = agentLabel(ev);
    var label = badgeLabel(ev);
    const row = document.createElement("div");
    row.className = "event-row";
    // Store filterable data on the element
    row.dataset.type = label.startsWith("gate:") ? "gate" : label;
    row.dataset.agent = agent;
    row.dataset.search = (agent + " " + label + " " + full).toLowerCase();
    row.innerHTML =
      '<span class="time">' + formatTime(ev.ts || Date.now()/1000) + '</span>'
      + '<span class="agent">' + escHtml(agent) + '</span>'
      + '<span class="badge ' + badgeClass(ev) + '">' + badgeLabel(ev) + '</span>'
      + '<span class="detail">' + escHtml(truncate(full, 120)) + '</span>';

    // Determine if this is a conversable gate/sentinel event
    var isConversable = false;
    var convId = "";
    if (ev.event_type === "GateDecisionEvent" && ev.evaluator !== "denylist") {
      isConversable = true;
      convId = "gate:" + ev.tool_use_id;
    } else if (ev.event_type === "GateDecisionEvent" && ev.action && ev.action !== "allow") {
      isConversable = true;
      convId = "gate:" + ev.tool_use_id;
    } else if (ev.event_type === "SentinelReviewEvent" && ev.review_id) {
      isConversable = true;
      convId = "sentinel:" + ev.review_id;
    }

    if (isConversable) {
      row.classList.add("conversable");
      row.addEventListener("click", function(e) {
        // Don't toggle if clicking inside the chat area
        if (e.target.closest(".conv-chat")) return;
        // Don't toggle if user is selecting text
        if (window.getSelection().toString()) return;
        var detail = row.querySelector(".detail");
        var isExpanded = row.classList.toggle("expanded");
        detail.textContent = isExpanded ? full : truncate(full, 120);
        var chat = row.querySelector(".conv-chat");
        if (isExpanded && !chat) {
          openConversation(convId, row, ev);
        }
        if (chat) chat.style.display = isExpanded ? "" : "none";
      });
    } else if (full.length > 120) {
      row.addEventListener("click", function() {
        // Don't toggle if user is selecting text
        if (window.getSelection().toString()) return;
        const detail = row.querySelector(".detail");
        const isExpanded = row.classList.toggle("expanded");
        detail.textContent = isExpanded ? full : truncate(full, 120);
      });
    }
    // Track agent for agent filter pills
    if (agent && !knownAgents.has(agent)) {
      knownAgents.add(agent);
      addAgentPill(agent);
    }
    eventFeed.prepend(row);
    while (eventFeed.children.length > MAX_FEED) {
      eventFeed.removeChild(eventFeed.lastChild);
    }
    applyFiltersToRow(row);
  }

  // --- Filtering ---
  const knownAgents = new Set();
  let activeTypeFilter = "all";
  let activeAgentFilter = "all";
  let searchQuery = "";

  function addAgentPill(agent) {
    var pill = document.createElement("span");
    pill.className = "filter-pill";
    pill.dataset.filter = "agent";
    pill.dataset.value = agent;
    pill.textContent = agent;
    pill.addEventListener("click", function() { toggleFilter(pill); });
    document.getElementById("agent-filters").appendChild(pill);
  }

  function applyFiltersToRow(row) {
    var show = true;
    if (activeTypeFilter !== "all" && row.dataset.type !== activeTypeFilter) show = false;
    if (activeAgentFilter !== "all" && row.dataset.agent !== activeAgentFilter) show = false;
    if (searchQuery && row.dataset.search.indexOf(searchQuery) < 0) show = false;
    row.classList.toggle("hidden", !show);
  }

  function applyAllFilters() {
    var rows = eventFeed.querySelectorAll(".event-row");
    for (var i = 0; i < rows.length; i++) applyFiltersToRow(rows[i]);
  }

  function toggleFilter(pill) {
    var group = pill.dataset.filter;
    var value = pill.dataset.value;
    // Deactivate siblings
    document.querySelectorAll('.filter-pill[data-filter="' + group + '"]').forEach(function(p) {
      p.classList.remove("active");
    });
    pill.classList.add("active");
    if (group === "type") activeTypeFilter = value;
    if (group === "agent") activeAgentFilter = value;
    applyAllFilters();
  }

  // Wire up type filter pills
  document.querySelectorAll('.filter-pill[data-filter="type"]').forEach(function(pill) {
    pill.addEventListener("click", function() { toggleFilter(pill); });
  });

  // Wire up search
  document.getElementById("filter-search").addEventListener("input", function(e) {
    searchQuery = e.target.value.toLowerCase();
    applyAllFilters();
  });

  // Add "all" agent pill
  (function() {
    var allPill = document.createElement("span");
    allPill.className = "filter-pill active";
    allPill.dataset.filter = "agent";
    allPill.dataset.value = "all";
    allPill.textContent = "all";
    allPill.addEventListener("click", function() { toggleFilter(allPill); });
    document.getElementById("agent-filters").appendChild(allPill);
  })();

  // --- Browser Notifications ---
  const notifBtn = document.getElementById("notif-btn");
  let notifMuted = localStorage.getItem("cross-notif-muted") === "1";

  function updateNotifBtn() {
    if (!("Notification" in window)) { notifBtn.style.display = "none"; return; }
    notifBtn.style.display = "";
    var perm = Notification.permission;
    if (perm === "granted" && !notifMuted) {
      notifBtn.textContent = "Notifications on";
      notifBtn.className = "notif-btn granted";
    } else if (perm === "granted" && notifMuted) {
      notifBtn.textContent = "Notifications off";
      notifBtn.className = "notif-btn";
    } else if (perm === "denied") {
      notifBtn.textContent = "Notifications blocked";
      notifBtn.className = "notif-btn denied";
    } else {
      notifBtn.textContent = "Enable notifications";
      notifBtn.className = "notif-btn";
    }
  }
  window.requestNotifPermission = function() {
    if (!("Notification" in window)) return;
    if (Notification.permission === "denied") return;
    if (Notification.permission === "granted") {
      // Toggle mute
      notifMuted = !notifMuted;
      localStorage.setItem("cross-notif-muted", notifMuted ? "1" : "0");
      updateNotifBtn();
      return;
    }
    Notification.requestPermission().then(updateNotifBtn);
  };
  updateNotifBtn();

  // Show modal on first visit if notifications not yet decided
  (function() {
    var modal = document.getElementById("notif-modal");
    if (!("Notification" in window)) return;
    if (Notification.permission !== "default") return;
    if (localStorage.getItem("cross-notif-dismissed")) return;
    modal.style.display = "flex";
    document.getElementById("notif-modal-enable").addEventListener("click", function() {
      modal.style.display = "none";
      notifMuted = false;
      localStorage.setItem("cross-notif-muted", "0");
      Notification.requestPermission().then(updateNotifBtn);
    });
    document.getElementById("notif-modal-skip").addEventListener("click", function() {
      modal.style.display = "none";
      localStorage.setItem("cross-notif-dismissed", "1");
    });
  })();

  function showNotification(title, body, tag) {
    if (notifMuted) return;
    if (!("Notification" in window) || Notification.permission !== "granted") return;
    try {
      var n = new Notification(title, {body: body, tag: tag || "", icon: ""});
      n.onclick = function() { window.focus(); n.close(); };
    } catch(e) {}
  }

  // Events not worth showing in the feed
  const HIDDEN_EVENTS = new Set(["MessageDeltaEvent", "MessageStartEvent"]);

  let lastRequestPreview = "";

  function shouldHide(ev) {
    if (HIDDEN_EVENTS.has(ev.event_type)) return true;
    // Hide denylist allows (pass-through noise), but show LLM gate allows (reviewed)
    if (ev.event_type === "GateDecisionEvent" && ev.action === "allow" && ev.evaluator === "denylist") return true;
    if (ev.event_type === "RequestEvent") {
      if (!ev.last_message_preview) return true;
      if (ev.last_message_preview === lastRequestPreview) return true;
      lastRequestPreview = ev.last_message_preview;
    }
    return false;
  }

  function handleEvent(ev) {
    // Handle permission prompt events (don't add to feed — they're shown as cards)
    if (ev.event_type === "PermissionPromptEvent") {
      permissionMap[ev.session_id] = ev;
      renderPending();
      showNotification(
        "cross — permission needed",
        ev.tool_desc || "Claude Code needs approval",
        "permission-" + ev.session_id
      );
      return;
    }
    if (ev.event_type === "PermissionResolvedEvent") {
      delete permissionMap[ev.session_id];
      renderPending();
      return;
    }

    if (shouldHide(ev)) return;
    addEventRow(ev);
    if (ev.event_type === "GateDecisionEvent") {
      if (ev.action === "escalate") {
        pendingMap[ev.tool_use_id] = ev;
        renderPending();
        showNotification(
          "cross — approval needed",
          (ev.tool_name || "unknown tool") + (ev.reason ? ": " + ev.reason : ""),
          "escalate-" + ev.tool_use_id
        );
      } else if (ev.action !== "escalate" && pendingMap[ev.tool_use_id]) {
        delete pendingMap[ev.tool_use_id];
        renderPending();
      }
    }
    if (ev.event_type === "SentinelReviewEvent" && ev.action && ev.action !== "allow") {
      showNotification(
        "cross — sentinel " + ev.action,
        ev.summary || ev.concerns || "",
        "sentinel-" + (ev.ts || "")
      );
    }
  }

  window.resolveGate = function(toolUseId, approved) {
    // Disable buttons immediately
    const card = document.querySelector('.pending-card[data-id="' + toolUseId + '"]');
    if (card) {
      card.querySelectorAll("button").forEach(function(b) { b.disabled = true; });
    }
    fetch("/cross/api/pending/" + toolUseId + "/resolve", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({approved: approved, username: "dashboard"})
    }).then(function(r) {
      if (r.ok) {
        delete pendingMap[toolUseId];
        renderPending();
      }
    }).catch(function(e) {
      console.error("Failed to resolve:", e);
      if (card) card.querySelectorAll("button").forEach(function(b) { b.disabled = false; });
    });
  };

  window.resolvePerm = function(sessionId, action) {
    // Disable buttons immediately
    const card = document.querySelector('.permission-card[data-sid="' + sessionId + '"]');
    if (card) {
      card.querySelectorAll("button").forEach(function(b) { b.disabled = true; });
    }
    fetch("/cross/api/permission/" + sessionId + "/resolve", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({action: action, username: "dashboard"})
    }).then(function(r) {
      if (r.ok) {
        delete permissionMap[sessionId];
        renderPending();
      }
    }).catch(function(e) {
      console.error("Failed to resolve permission:", e);
      if (card) card.querySelectorAll("button").forEach(function(b) { b.disabled = false; });
    });
  };

  // --- Halted Sessions ---
  function renderHalted() {
    var keys = Object.keys(haltedMap);
    if (keys.length === 0) {
      haltedSection.style.display = "none";
      return;
    }
    haltedSection.style.display = "";
    var html = "";
    for (var i = 0; i < keys.length; i++) {
      var sid = keys[i];
      var reason = haltedMap[sid];
      var safeSid = escHtml(sid);
      var label = sid === "__global__" ? "Global halt" : "Session " + safeSid;
      html += '<div class="halted-card" data-sid="' + safeSid + '">'
        + '<div class="halted-label">' + escHtml(label) + '</div>'
        + (sid !== "__global__" ? '<div class="session-id">' + safeSid + '</div>' : '')
        + '<div class="reason">' + escHtml(reason) + '</div>'
        + '<div class="actions">'
        + '<button class="btn btn-unhalt" onclick="unhaltSession(&apos;'
        + safeSid + '&apos;)">Un-halt</button>'
        + '</div></div>';
    }
    haltedList.innerHTML = html;
  }

  window.unhaltSession = function(sessionId) {
    var card = document.querySelector('.halted-card[data-sid="' + sessionId + '"]');
    if (card) card.querySelectorAll("button").forEach(function(b) { b.disabled = true; });
    fetch("/cross/api/halted-sessions/" + encodeURIComponent(sessionId) + "/resolve", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({username: "dashboard"})
    }).then(function(r) {
      if (r.ok) {
        delete haltedMap[sessionId];
        renderHalted();
      }
    }).catch(function(e) {
      console.error("Failed to un-halt session:", e);
      if (card) card.querySelectorAll("button").forEach(function(b) { b.disabled = false; });
    });
  };

  function refreshHalted() {
    fetch("/cross/api/halted-sessions").then(function(r) { return r.json(); }).then(function(data) {
      haltedMap = data;
      renderHalted();
    }).catch(function() {});
  }

  // --- Conversations ---
  function openConversation(convId, row, evData) {
    var chat = document.createElement("div");
    chat.className = "conv-chat";
    chat.dataset.convId = convId;
    // Prevent clicks inside chat from toggling row expand
    chat.addEventListener("click", function(e) { e.stopPropagation(); });

    var msgsDiv = document.createElement("div");
    msgsDiv.className = "conv-messages";

    var inputRow = document.createElement("div");
    inputRow.className = "conv-input-row";

    var input = document.createElement("input");
    input.className = "conv-input";
    input.type = "text";
    input.placeholder = "Ask about this evaluation...";

    var sendBtn = document.createElement("button");
    sendBtn.className = "conv-send";
    sendBtn.textContent = "Send";

    var sentEventData = false;

    function doSend() {
      var text = input.value.trim();
      if (!text || sendBtn.disabled) return;
      input.value = "";
      // Render user message
      var userMsg = document.createElement("div");
      userMsg.className = "conv-msg user";
      userMsg.textContent = text;
      msgsDiv.appendChild(userMsg);
      msgsDiv.scrollTop = msgsDiv.scrollHeight;
      // Show typing indicator
      var typing = document.createElement("div");
      typing.className = "conv-typing";
      typing.textContent = "Thinking...";
      typing.dataset.convId = convId;
      msgsDiv.appendChild(typing);
      msgsDiv.scrollTop = msgsDiv.scrollHeight;
      sendBtn.disabled = true;
      // Send via WebSocket — include event data on first message for lazy context registration
      if (ws && ws.readyState === WebSocket.OPEN) {
        var payload = {
          type: "conversation_message",
          conversation_id: convId,
          message: text
        };
        if (!sentEventData && evData) {
          payload.event_data = evData;
          sentEventData = true;
        }
        ws.send(JSON.stringify(payload));
      }
    }

    sendBtn.addEventListener("click", doSend);
    input.addEventListener("keydown", function(e) {
      if (e.key === "Enter") { e.preventDefault(); doSend(); }
    });

    inputRow.appendChild(input);
    inputRow.appendChild(sendBtn);
    chat.appendChild(msgsDiv);
    chat.appendChild(inputRow);
    row.appendChild(chat);

    // Load existing messages
    fetch("/cross/api/conversations/" + encodeURIComponent(convId))
      .then(function(r) { return r.json(); })
      .then(function(data) {
        if (data.messages && data.messages.length > 0) {
          data.messages.forEach(function(m) {
            var msgEl = document.createElement("div");
            msgEl.className = "conv-msg " + m.role;
            msgEl.textContent = m.content;
            msgsDiv.appendChild(msgEl);
          });
          msgsDiv.scrollTop = msgsDiv.scrollHeight;
        }
      })
      .catch(function() {});

    // Focus the input
    setTimeout(function() { input.focus(); }, 50);
  }

  function renderConvReply(convId, text) {
    var chat = document.querySelector('.conv-chat[data-conv-id="' + convId + '"]');
    if (!chat) return;
    var msgsDiv = chat.querySelector(".conv-messages");
    // Remove typing indicator
    var typing = msgsDiv.querySelector(".conv-typing");
    if (typing) typing.remove();
    // Add assistant message
    var msgEl = document.createElement("div");
    msgEl.className = "conv-msg assistant";
    msgEl.textContent = text;
    msgsDiv.appendChild(msgEl);
    msgsDiv.scrollTop = msgsDiv.scrollHeight;
    // Re-enable send button
    var sendBtn = chat.querySelector(".conv-send");
    if (sendBtn) sendBtn.disabled = false;
  }

  function showConvTyping(convId) {
    var chat = document.querySelector('.conv-chat[data-conv-id="' + convId + '"]');
    if (!chat) return;
    var msgsDiv = chat.querySelector(".conv-messages");
    // Only add if not already showing
    if (!msgsDiv.querySelector(".conv-typing")) {
      var typing = document.createElement("div");
      typing.className = "conv-typing";
      typing.textContent = "Thinking...";
      msgsDiv.appendChild(typing);
      msgsDiv.scrollTop = msgsDiv.scrollHeight;
    }
  }

  function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(proto + "//" + location.host + "/cross/api/ws");
    ws.onopen = function() {};
    ws.onclose = function() {
      setTimeout(connect, 2000);
    };
    ws.onerror = function() { ws.close(); };
    ws.onmessage = function(msg) {
      try {
        var data = JSON.parse(msg.data);
        if (data.type === "conversation_reply") {
          renderConvReply(data.conversation_id, data.reply);
          return;
        }
        if (data.type === "conversation_typing") {
          showConvTyping(data.conversation_id);
          return;
        }
        handleEvent(data);
      } catch(e) {}
    };
  }

  // --- Agent Status ---
  const agentStatusEl = document.getElementById("agent-status");

  function renderStatus(data) {
    var html = "";
    for (var i = 0; i < data.monitored.length; i++) {
      var m = data.monitored[i];
      var state = m.active ? "active" : "stopped";
      html += '<span class="agent-chip monitored ' + state + '"><span class="dot"></span>' + escHtml(m.label) + '</span>';
    }
    for (var i = 0; i < data.unmonitored.length; i++) {
      var u = data.unmonitored[i];
      html += '<span class="agent-chip unmonitored stopped"><span class="dot"></span>' + escHtml(u.agent) + '</span>';
    }
    if (!data.monitored.length && !data.unmonitored.length) {
      html = '<span class="status-summary">No agents detected</span>';
    }
    agentStatusEl.innerHTML = html;
  }

  function refreshStatus() {
    fetch("/cross/api/status").then(function(r) { return r.json(); }).then(renderStatus).catch(function() {});
  }
  refreshStatus();
  setInterval(refreshStatus, 5000);

  // Initial load: fetch existing events and pending
  fetch("/cross/api/events").then(function(r) { return r.json(); }).then(function(events) {
    for (const ev of events) {
      if (!shouldHide(ev)) addEventRow(ev);
    }
  }).catch(function() {});

  fetch("/cross/api/pending").then(function(r) { return r.json(); }).then(function(pending) {
    for (const ev of pending) { pendingMap[ev.tool_use_id] = ev; }
    renderPending();
  }).catch(function() {});

  fetch("/cross/api/pending-permissions").then(function(r) { return r.json(); }).then(function(perms) {
    for (const p of perms) { permissionMap[p.session_id] = p; }
    renderPending();
  }).catch(function() {});

  refreshHalted();
  setInterval(refreshHalted, 10000);

  connect();
})();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Settings HTML (separate page)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Shared fragments for header across pages
# ---------------------------------------------------------------------------

_GEAR_SVG = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>'

_SHARED_HEADER_CSS = """
  header {
    border-bottom: 1px solid var(--border);
    padding: 16px 24px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  header h1 { font-size: 20px; font-weight: 600; letter-spacing: -0.5px; }
  header .header-right {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  header .notif-btn {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 12px;
    font-size: 12px;
    color: var(--text-dim);
    cursor: pointer;
    transition: opacity 0.15s;
  }
  header .notif-btn:hover { opacity: 0.85; }
  header .notif-btn.granted { color: var(--green); border-color: var(--green); }
  header .notif-btn.denied { color: var(--red); border-color: var(--red); cursor: not-allowed; }
  header .settings-link {
    color: var(--text-dim);
    text-decoration: none;
    transition: color 0.15s;
    display: flex;
    align-items: center;
  }
  header .settings-link:hover { color: var(--text); }
"""

_NOTIF_MODAL_CSS = """
  .notif-modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(0,0,0,0.6);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
  }
  .notif-modal {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 32px;
    max-width: 400px;
    text-align: center;
  }
  .notif-modal h3 { font-size: 18px; margin-bottom: 12px; }
  .notif-modal p {
    color: var(--text-dim);
    font-size: 14px;
    line-height: 1.5;
    margin-bottom: 20px;
  }
  .notif-modal .btn-row { display: flex; gap: 12px; justify-content: center; }
  .notif-modal .btn-enable {
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 6px;
    padding: 8px 24px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
  }
  .notif-modal .btn-enable:hover { opacity: 0.85; }
  .notif-modal .btn-skip {
    background: transparent;
    color: var(--text-dim);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 8px 24px;
    font-size: 14px;
    cursor: pointer;
  }
  .notif-modal .btn-skip:hover { opacity: 0.85; }
"""

_NOTIF_MODAL_HTML = """
<div class="notif-modal-overlay" id="notif-modal" style="display:none">
  <div class="notif-modal">
    <h3>Enable notifications</h3>
    <p>Get notified when an agent needs approval or the sentinel flags something.</p>
    <div class="btn-row">
      <button class="btn-enable" id="notif-modal-enable">Enable</button>
      <button class="btn-skip" id="notif-modal-skip">Not now</button>
    </div>
  </div>
</div>
"""

_HEADER_HTML = f"""
<header>
  <h1><svg width="24" height="24" viewBox="0 0 24 24" fill="none"
    style="display:block"><path d="M12 2v20M2 12h20"
    stroke="white" stroke-width="3" stroke-linecap="round"/></svg></h1>
  <div class="header-right">
    <button class="notif-btn" id="notif-btn" style="display:none"
      onclick="requestNotifPermission()">Enable notifications</button>
    <a class="settings-link" href="/cross/settings" title="Settings">{_GEAR_SVG}</a>
  </div>
</header>
"""

_NOTIF_JS = """
  // --- Browser Notifications ---
  const notifBtn = document.getElementById("notif-btn");
  let notifMuted = localStorage.getItem("cross-notif-muted") === "1";

  function updateNotifBtn() {
    if (!("Notification" in window)) { notifBtn.style.display = "none"; return; }
    notifBtn.style.display = "";
    var perm = Notification.permission;
    if (perm === "granted" && !notifMuted) {
      notifBtn.textContent = "Notifications on";
      notifBtn.className = "notif-btn granted";
    } else if (perm === "granted" && notifMuted) {
      notifBtn.textContent = "Notifications off";
      notifBtn.className = "notif-btn";
    } else if (perm === "denied") {
      notifBtn.textContent = "Notifications blocked";
      notifBtn.className = "notif-btn denied";
    } else {
      notifBtn.textContent = "Enable notifications";
      notifBtn.className = "notif-btn";
    }
  }
  window.requestNotifPermission = function() {
    if (!("Notification" in window)) return;
    if (Notification.permission === "denied") return;
    if (Notification.permission === "granted") {
      notifMuted = !notifMuted;
      localStorage.setItem("cross-notif-muted", notifMuted ? "1" : "0");
      updateNotifBtn();
      return;
    }
    Notification.requestPermission().then(updateNotifBtn);
  };
  updateNotifBtn();

  // Show modal on first visit if notifications not yet decided
  (function() {
    var modal = document.getElementById("notif-modal");
    if (!("Notification" in window)) return;
    if (Notification.permission !== "default") return;
    if (localStorage.getItem("cross-notif-dismissed")) return;
    modal.style.display = "flex";
    document.getElementById("notif-modal-enable").addEventListener("click", function() {
      modal.style.display = "none";
      notifMuted = false;
      localStorage.setItem("cross-notif-muted", "0");
      Notification.requestPermission().then(updateNotifBtn);
    });
    document.getElementById("notif-modal-skip").addEventListener("click", function() {
      modal.style.display = "none";
      localStorage.setItem("cross-notif-dismissed", "1");
    });
  })();
"""

# ---------------------------------------------------------------------------
# Settings HTML (separate page)
# ---------------------------------------------------------------------------

SETTINGS_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cross settings</title>
<link rel="icon" type="image/svg+xml"
  href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg'
  viewBox='0 0 24 24'><path d='M12 2v20M2 12h20'
  stroke='white' stroke-width='3' stroke-linecap='round'/></svg>">
<style>
  :root {{
    --bg: #0d1117;
    --surface: #161b22;
    --border: #30363d;
    --text: #e6edf3;
    --text-dim: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --red: #f85149;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
  }}
  {_SHARED_HEADER_CSS}
  {_NOTIF_MODAL_CSS}
  main {{
    max-width: 960px;
    margin: 0 auto;
    padding: 24px;
  }}
  .breadcrumb {{
    font-size: 13px;
    color: var(--text-dim);
    margin-bottom: 4px;
  }}
  .breadcrumb a {{
    color: var(--accent);
    text-decoration: none;
  }}
  .breadcrumb a:hover {{ text-decoration: underline; }}
  .page-title {{
    font-size: 24px;
    font-weight: 600;
    margin-bottom: 24px;
  }}
  section {{ margin-bottom: 32px; }}
  section h2 {{
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: var(--text-dim);
    margin-bottom: 12px;
  }}
  .instructions-editor {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px;
  }}
  .instructions-editor textarea {{
    width: 100%;
    min-height: 200px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 10px 12px;
    font-family: "SF Mono", "Fira Code", monospace;
    font-size: 13px;
    color: var(--text);
    resize: vertical;
    outline: none;
    line-height: 1.5;
  }}
  .instructions-editor textarea:focus {{ border-color: var(--accent); }}
  .instructions-editor textarea::placeholder {{ color: var(--text-dim); }}
  .instructions-actions {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 10px;
  }}
  .instructions-actions .btn-save {{
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: 6px;
    padding: 6px 18px;
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: opacity 0.15s;
  }}
  .instructions-actions .btn-save:hover {{ opacity: 0.85; }}
  .instructions-actions .btn-save:disabled {{ opacity: 0.5; cursor: not-allowed; }}
  .instructions-actions .save-status {{
    font-size: 12px;
    color: var(--text-dim);
  }}
  .instructions-actions .save-status.ok {{ color: var(--green); }}
  .instructions-actions .save-status.err {{ color: var(--red); }}
  .instructions-hint {{
    font-size: 12px;
    color: var(--text-dim);
    margin-bottom: 8px;
  }}
</style>
</head>
<body>
{_NOTIF_MODAL_HTML}
{_HEADER_HTML}
<main>
  <div class="breadcrumb"><a href="/cross/dashboard">&larr; Dashboard</a></div>
  <h1 class="page-title">Settings</h1>
  <section>
    <h2>Custom Instructions</h2>
    <div class="instructions-editor">
      <p class="instructions-hint">Included with every gate and sentinel prompt. Changes apply immediately.</p>
      <textarea id="instructions-text"
        placeholder="Add custom instructions for gate and sentinel reviewers..."></textarea>
      <div class="instructions-actions">
        <button class="btn-save" id="instructions-save">Save</button>
        <span class="save-status" id="instructions-status"></span>
      </div>
    </div>
  </section>
</main>
<script>
(function() {{
  {_NOTIF_JS}

  var instrText = document.getElementById("instructions-text");
  var instrStatus = document.getElementById("instructions-status");
  var instrSaveBtn = document.getElementById("instructions-save");

  fetch("/cross/api/instructions").then(function(r) {{ return r.json(); }}).then(function(data) {{
    instrText.value = data.content || "";
  }}).catch(function() {{}});

  function saveInstructions() {{
    instrSaveBtn.disabled = true;
    instrStatus.textContent = "Saving...";
    instrStatus.className = "save-status";
    fetch("/cross/api/instructions", {{
      method: "PUT",
      headers: {{"Content-Type": "application/json"}},
      body: JSON.stringify({{content: instrText.value}})
    }}).then(function(r) {{
      if (r.ok) {{
        instrStatus.textContent = "Saved";
        instrStatus.className = "save-status ok";
      }} else {{
        r.json().then(function(d) {{
          instrStatus.textContent = "Error: " + (d.error || "unknown");
          instrStatus.className = "save-status err";
        }});
      }}
    }}).catch(function(e) {{
      instrStatus.textContent = "Error: " + e.message;
      instrStatus.className = "save-status err";
    }}).finally(function() {{
      instrSaveBtn.disabled = false;
      setTimeout(function() {{ instrStatus.textContent = ""; }}, 4000);
    }});
  }}

  instrSaveBtn.addEventListener("click", saveInstructions);

  instrText.addEventListener("keydown", function(e) {{
    if ((e.ctrlKey || e.metaKey) && e.key === "s") {{
      e.preventDefault();
      saveInstructions();
    }}
  }});
}})();
</script>
</body>
</html>"""
