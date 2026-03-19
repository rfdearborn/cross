"""Web dashboard plugin — local approval surface and live event feed.

Subscribes to EventBus events, tracks pending escalations, and broadcasts
to WebSocket clients. Event storage is handled by EventStore.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from starlette.websockets import WebSocket, WebSocketDisconnect

from cross.event_store import EventStore, event_to_dict
from cross.events import (
    CrossEvent,
    GateDecisionEvent,
)

logger = logging.getLogger("cross.plugins.dashboard")


class DashboardPlugin:
    """Tracks pending escalations and broadcasts events to WebSocket clients."""

    def __init__(self, event_store: EventStore, resolve_approval_callback=None):
        self._store = event_store
        # Pending escalations: tool_use_id -> event dict
        self._pending: dict[str, dict[str, Any]] = {}
        # Connected WebSocket clients
        self._ws_clients: set[WebSocket] = set()
        # Callback to resolve gate approvals in the proxy
        self._resolve_approval_callback = resolve_approval_callback

    async def handle_event(self, event: CrossEvent):
        """EventBus handler — track escalations and broadcast to WS clients."""
        event_dict = event_to_dict(event)

        # Track pending escalations
        if isinstance(event, GateDecisionEvent):
            if event.action == "escalate":
                self._pending[event.tool_use_id] = event_dict
            elif event.action in ("allow", "block") and event.tool_use_id in self._pending:
                # Escalation resolved (by Slack, dashboard, or timeout)
                del self._pending[event.tool_use_id]

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
                # Keep connection alive; client may send pings
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        except Exception:
            pass
        finally:
            self._ws_clients.discard(ws)
            logger.info(f"Dashboard WS disconnected ({len(self._ws_clients)} clients)")


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
  header .notif-btn {
    margin-left: auto;
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
  header .status {
    font-size: 12px;
    color: var(--text-dim);
  }
  header .status .dot {
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 4px;
    vertical-align: middle;
  }
  header .status .dot.connected { background: var(--green); }
  header .status .dot.disconnected { background: var(--red); }
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
  .btn-deny { background: var(--red); color: #fff; }

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
    user-select: none;
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
  }
  .event-row.expanded .detail {
    white-space: pre-wrap;
    overflow: visible;
    text-overflow: unset;
    word-break: break-word;
  }
  .event-row.hidden { display: none; }

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
  <button class="notif-btn" id="notif-btn" style="display:none"
    onclick="requestNotifPermission()">Enable notifications</button>
  <span class="status">
    <span class="dot disconnected" id="ws-dot"></span>
    <span id="ws-status">connecting...</span>
  </span>
</header>
<main>
  <section id="status-section">
    <h2>Agents</h2>
    <div id="agent-status" class="agent-status"></div>
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
  const wsDot = document.getElementById("ws-dot");
  const wsStatus = document.getElementById("ws-status");
  const feedEmpty = document.getElementById("feed-empty");

  const MAX_FEED = 200;
  let pendingMap = {};
  let ws = null;

  function formatTime(ts) {
    const d = new Date(ts * 1000);
    return d.toLocaleTimeString([], {hour: "2-digit", minute: "2-digit", second: "2-digit"});
  }

  function truncate(s, n) {
    if (!s) return "";
    s = String(s);
    return s.length > n ? s.slice(0, n) + "..." : s;
  }

  function renderPending() {
    const keys = Object.keys(pendingMap);
    if (keys.length === 0) {
      pendingList.innerHTML = '<p class="empty">No pending approvals</p>';
      return;
    }
    let html = "";
    for (const id of keys) {
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
    if (full.length > 120) {
      row.addEventListener("click", function() {
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
      } else if ((ev.action === "allow" || ev.action === "block") && pendingMap[ev.tool_use_id]) {
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

  function connect() {
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(proto + "//" + location.host + "/cross/api/ws");
    ws.onopen = function() {
      wsDot.className = "dot connected";
      wsStatus.textContent = "connected";
    };
    ws.onclose = function() {
      wsDot.className = "dot disconnected";
      wsStatus.textContent = "disconnected — reconnecting...";
      setTimeout(connect, 2000);
    };
    ws.onerror = function() { ws.close(); };
    ws.onmessage = function(msg) {
      try { handleEvent(JSON.parse(msg.data)); } catch(e) {}
    };
  }

  // --- Agent Status ---
  const agentStatusEl = document.getElementById("agent-status");

  function renderStatus(data) {
    var html = "";
    for (var i = 0; i < data.monitored.length; i++) {
      var m = data.monitored[i];
      html += '<span class="agent-chip monitored"><span class="dot"></span>' + escHtml(m.label) + '</span>';
    }
    for (var i = 0; i < data.unmonitored.length; i++) {
      var u = data.unmonitored[i];
      html += '<span class="agent-chip unmonitored"><span class="dot"></span>' + escHtml(u.agent) + '</span>';
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
  setInterval(refreshStatus, 10000);

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

  connect();
})();
</script>
</body>
</html>"""
