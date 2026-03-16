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
  .badge-tool_use { background: #1f3a5f; color: var(--accent); }
  .badge-gate_allow { background: #1a3a2a; color: var(--green); }
  .badge-gate_block { background: #3d1f1f; color: var(--red); }
  .badge-gate_alert { background: #3d2f1f; color: var(--orange); }
  .badge-gate_escalate { background: #3d2f1f; color: var(--yellow); }
  .badge-sentinel { background: #2a1f3d; color: #bc8cff; }
  .badge-request { background: #1f2a3d; color: var(--text-dim); }
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
</style>
</head>
<body>
<header>
  <h1><svg width="24" height="24" viewBox="0 0 24 24" fill="none"
    style="display:block"><path d="M12 2v20M2 12h20"
    stroke="white" stroke-width="3" stroke-linecap="round"/></svg></h1>
  <button class="notif-btn" id="notif-btn" style="display:none" onclick="requestNotifPermission()">Enable notifications</button>
  <span class="status">
    <span class="dot disconnected" id="ws-dot"></span>
    <span id="ws-status">connecting...</span>
  </span>
</header>
<main>
  <section id="pending-section">
    <h2>Pending Approvals</h2>
    <div id="pending-list"><p class="empty">No pending approvals</p></div>
  </section>
  <section>
    <h2>Live Event Feed</h2>
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
    return "";
  }

  function badgeLabel(ev) {
    const t = ev.event_type;
    if (t === "ToolUseEvent") return "tool_use";
    if (t === "GateDecisionEvent") return "gate:" + (ev.action || "?");
    if (t === "SentinelReviewEvent") return "sentinel";
    if (t === "RequestEvent") return "request";
    return t.replace("Event", "").toLowerCase();
  }

  function detailText(ev) {
    const t = ev.event_type;
    if (t === "ToolUseEvent") {
      let s = ev.name || "";
      if (ev.input && ev.input.command) s += ": " + ev.input.command;
      else if (ev.input && ev.input.file_path) s += ": " + ev.input.file_path;
      return s;
    }
    if (t === "GateDecisionEvent") return (ev.tool_name || "") + (ev.reason ? " — " + ev.reason : "");
    if (t === "SentinelReviewEvent") return ev.summary || ev.concerns || "";
    if (t === "RequestEvent") return (ev.method || "") + " " + (ev.path || "") + " model=" + (ev.model || "?");
    return "";
  }

  function addEventRow(ev) {
    if (feedEmpty) feedEmpty.remove();
    const full = detailText(ev);
    const row = document.createElement("div");
    row.className = "event-row";
    row.innerHTML =
      '<span class="time">' + formatTime(ev.ts || Date.now()/1000) + '</span>'
      + '<span class="badge ' + badgeClass(ev) + '">' + badgeLabel(ev) + '</span>'
      + '<span class="detail">' + escHtml(truncate(full, 120)) + '</span>';
    if (full.length > 120) {
      row.addEventListener("click", function() {
        const detail = row.querySelector(".detail");
        const isExpanded = row.classList.toggle("expanded");
        detail.textContent = isExpanded ? full : truncate(full, 120);
      });
    }
    eventFeed.prepend(row);
    while (eventFeed.children.length > MAX_FEED) {
      eventFeed.removeChild(eventFeed.lastChild);
    }
  }

  // --- Browser Notifications ---
  const notifBtn = document.getElementById("notif-btn");
  function updateNotifBtn() {
    if (!("Notification" in window)) { notifBtn.style.display = "none"; return; }
    notifBtn.style.display = "";
    var perm = Notification.permission;
    if (perm === "granted") {
      notifBtn.textContent = "Notifications on";
      notifBtn.className = "notif-btn granted";
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
    Notification.requestPermission().then(updateNotifBtn);
  };
  updateNotifBtn();

  function showNotification(title, body, tag) {
    if (!("Notification" in window) || Notification.permission !== "granted") return;
    try {
      var n = new Notification(title, {body: body, tag: tag || "", icon: ""});
      n.onclick = function() { window.focus(); n.close(); };
    } catch(e) {}
  }

  function handleEvent(ev) {
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

  // Initial load: fetch existing events and pending
  fetch("/cross/api/events").then(function(r) { return r.json(); }).then(function(events) {
    for (const ev of events) { addEventRow(ev); }
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
