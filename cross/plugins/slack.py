"""Slack relay plugin — mirrors agent sessions to Slack channels/threads."""

from __future__ import annotations

import json
import logging
import re
import threading

from slack_sdk import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.request import SocketModeRequest
from slack_sdk.socket_mode.response import SocketModeResponse

from cross.config import settings
from cross.events import (
    CrossEvent,
    ErrorEvent,
    GateDecisionEvent,
    PermissionPromptEvent,
    PermissionResolvedEvent,
    SentinelReviewEvent,
    TextEvent,
    ToolUseEvent,
)

logger = logging.getLogger("cross.plugins.slack")


class SlackPlugin:
    """Manages Slack channels/threads and relays agent events."""

    def __init__(
        self,
        inject_callback=None,
        spawn_callback=None,
        resolve_approval_callback=None,
        resolve_permission_callback=None,
        event_loop=None,
        conversation_store=None,
    ):
        self._web = WebClient(token=settings.slack_bot_token)
        self._socket: SocketModeClient | None = None
        self._event_loop = event_loop
        self._spawn_callback = spawn_callback
        self._resolve_approval_callback = resolve_approval_callback
        self._resolve_permission_callback = resolve_permission_callback
        self._conversation_store = conversation_store
        # channel name -> channel ID
        self._channels: dict[str, str] = {}
        # session_id -> (channel_id, thread_ts)
        self._threads: dict[str, tuple[str, str]] = {}
        # session_id -> session data dict
        self._sessions: dict[str, dict] = {}
        # bot user ID (resolved on connect)
        self._bot_user_id: str | None = None
        # Slack username (resolved on connect, for channel naming)
        self._username: str | None = None
        # Cached non-bot user IDs (resolved on connect)
        self._user_ids: list[str] = []
        self._lock = threading.Lock()
        # async callback to inject input into a session: (session_id, text) -> None
        self._inject_callback = inject_callback
        # Debounce: session_id -> last permission prompt time
        self._last_permission_post: dict[str, float] = {}
        self._PERMISSION_DEBOUNCE_SECS = 3.0
        # Track last tool use per session (for permission context)
        self._last_tool_desc: dict[str, str] = {}
        # Sessions with pending permission prompts: session_id -> (channel_id, message_ts)
        self._permission_pending: dict[str, tuple[str, str]] = {}
        # Slack-initiated sessions: project -> (channel_id, message_ts) for threading
        self._pending_thread_ts: dict[str, tuple[str, str]] = {}
        # Pending gate escalations: tool_use_id -> (channel_id, message_ts)
        self._gate_pending: dict[str, tuple[str, str]] = {}
        # Conversation threads: message_ts -> conversation_id (for gate/sentinel follow-ups)
        self._conv_threads: dict[str, str] = {}

    def start(self):
        """Connect to Slack via Socket Mode."""
        # Resolve bot user ID
        auth = self._web.auth_test()
        self._bot_user_id = auth["user_id"]
        logger.info(f"Slack connected as {auth['user']} ({self._bot_user_id})")

        # Resolve workspace users (for channel naming and invites)
        try:
            resp = self._web.users_list()
            for u in resp.get("members", []):
                if not u.get("is_bot") and not u.get("deleted") and u["id"] != "USLACKBOT":
                    self._user_ids.append(u["id"])
                    if not self._username and settings.slack_channel_append_user:
                        self._username = u.get("name", "")
                        logger.info(f"Slack channel username: {self._username}")
            logger.info(f"Resolved {len(self._user_ids)} workspace user(s)")
        except Exception as e:
            logger.warning(f"Failed to resolve workspace users: {e}")

        # Start socket mode in background thread
        self._socket = SocketModeClient(
            app_token=settings.slack_app_token,
            web_client=WebClient(token=settings.slack_bot_token),
        )
        self._socket.socket_mode_request_listeners.append(self._on_socket_event)
        self._socket.connect()
        logger.info("Slack Socket Mode connected")

    def stop(self):
        """Disconnect from Slack."""
        if self._socket:
            try:
                self._socket.disconnect()
            except Exception:
                pass

    # --- Session lifecycle ---

    def session_started_from_data(self, data: dict):
        """Create or find the channel and start a thread for this session."""
        session_id = data["session_id"]
        project = data.get("project", "unknown")
        agent = data.get("agent", "agent")
        cwd = data.get("cwd", "")

        with self._lock:
            self._sessions[session_id] = data

        # Check if this is a Slack-initiated session (thread under the @mention)
        pending = self._pending_thread_ts.pop(project, None)
        if pending:
            channel_id, thread_ts = pending
            self._web.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text=f"*{agent}* session started in `{project}` (`{cwd}`)",
            )
        else:
            channel_id = self._ensure_channel(project)
            resp = self._web.chat_postMessage(
                channel=channel_id,
                text=f"*{agent}* session started in `{project}` (`{cwd}`)",
            )
            thread_ts = resp["ts"]

        with self._lock:
            self._threads[session_id] = (channel_id, thread_ts)

    def session_ended_from_data(self, data: dict):
        """Post session-end message in the thread."""
        session_id = data.get("session_id", "")
        with self._lock:
            thread_info = self._threads.get(session_id)
            self._sessions.pop(session_id, None)

        if thread_info:
            channel_id, thread_ts = thread_info
            exit_code = data.get("exit_code", "?")
            started = data.get("started_at", 0)
            ended = data.get("ended_at", 0)
            duration = ""
            if started and ended:
                mins = int((ended - started) / 60)
                duration = f" after {mins}m" if mins > 0 else ""
            self._web.chat_postMessage(
                channel=channel_id,
                thread_ts=thread_ts,
                text=f"Session ended (exit code {exit_code}){duration}",
            )

    # --- PTY output handler ---

    def handle_pty_output(self, session_id: str, text: str):
        """Handle cleaned PTY output from a wrap process.

        Permission prompt detection is now centralized in the daemon
        (via _check_permission_prompt → PermissionPromptEvent), which
        applies delayed notification logic to avoid false positives from
        TUI redraws and auto-approved tool calls. This method is kept
        for future PTY-based signals but no longer detects prompts.
        """

    def _post_permission_prompt(self, session_id: str, tool_desc: str, allow_all_label: str):
        """Post a permission prompt message to Slack (called from event handler)."""
        with self._lock:
            thread_info = self._threads.get(session_id)

        if not thread_info:
            return
        channel_id, thread_ts = thread_info

        prompt_text = "⚠️ *Permission needed*"
        if tool_desc:
            prompt_text += f" for {tool_desc}"

        blocks = [
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": prompt_text},
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Approve"},
                        "style": "primary",
                        "action_id": "permission_approve",
                        "value": session_id,
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": allow_all_label},
                        "action_id": "permission_allow_all",
                        "value": session_id,
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Deny"},
                        "style": "danger",
                        "action_id": "permission_deny",
                        "value": session_id,
                    },
                ],
            },
        ]

        resp = self._web.chat_postMessage(
            channel=channel_id,
            thread_ts=thread_ts,
            reply_broadcast=True,
            text=prompt_text,
            blocks=blocks,
        )
        self._permission_pending[session_id] = (channel_id, resp["ts"])

    # --- EventBus handler (proxy events) ---

    def _resolve_thread(self, event_agent: str, event_session: str):
        """Resolve (session_id, thread_info) for an event.

        Returns the session thread if the event's session_id matches a
        registered session.  Otherwise returns (key, None) so the caller
        can post at the top level.
        """
        with self._lock:
            if event_session and event_session in self._threads:
                return event_session, self._threads[event_session]

            if event_agent and event_agent not in ("", "unknown"):
                return event_agent, None

            if self._threads:
                session_id = list(self._threads.keys())[-1]
                return session_id, self._threads.get(session_id)

            return None, None

    async def handle_event(self, event: CrossEvent):
        """Handle events from the network proxy EventBus.

        Only relays high-signal events: gate decisions (block/alert/escalate),
        sentinel reviews (alert/escalate/halt), and errors. Full conversation
        relay is handled by the JSONL logger, not Slack.
        """
        event_agent = getattr(event, "agent", "")
        event_session = getattr(event, "session_id", "")

        session_id, thread_info = self._resolve_thread(event_agent, event_session)

        if thread_info:
            channel_id, thread_ts = thread_info
            _unthreaded_note = ""
        else:
            # No session thread — post at top level in the agent's channel
            agent_for_channel = event_agent if event_agent and event_agent not in ("", "unknown") else "proxy"
            try:
                channel_id = self._ensure_channel(agent_for_channel)
            except Exception as e:
                logger.warning(f"Failed to get channel for {agent_for_channel}: {e}")
                return
            thread_ts = None
            _unthreaded_note = "\n_⚠️ Could not match to a session thread_"

        match event:
            case ToolUseEvent() if session_id:
                # Track tool description for permission prompt context (not posted to Slack)
                tool_desc = f"`{event.name}`"
                if event.name == "Bash":
                    cmd = event.input.get("command", "")
                    tool_desc = f"`Bash`: `{cmd[:80]}`"
                elif event.name in ("Read", "Write", "Edit"):
                    path = event.input.get("file_path", "")
                    tool_desc = f"`{event.name}`: `{path}`"
                self._last_tool_desc[session_id] = tool_desc

            case TextEvent() if session_id:
                # If permission was pending, it's now resolved (approved from terminal)
                if session_id in self._permission_pending:
                    perm_channel, perm_ts = self._permission_pending.pop(session_id)
                    result_text = "✅ *Approved* (terminal)"
                    try:
                        self._web.chat_update(
                            channel=perm_channel,
                            ts=perm_ts,
                            text=result_text,
                            blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": result_text}}],
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update permission message: {e}")

            case GateDecisionEvent() if event.action in ("block", "escalate", "alert", "allow", "halt_session"):
                # Check if this resolves a pending escalation (from dashboard, CLI, or timeout)
                pending = self._gate_pending.pop(event.tool_use_id, None)
                if pending and event.action != "escalate":
                    pend_channel, pend_ts = pending
                    # Extract username from reason like "Approved by human reviewer (@alice)"
                    m = re.search(r"@(\w+)", event.reason or "")
                    username = m.group(1) if m else "unknown"
                    if event.action == "allow":
                        result_text = f"✅ *Approved* by @{username}"
                    else:
                        result_text = f"❌ *Denied* by @{username}"
                    try:
                        self._web.chat_update(
                            channel=pend_channel,
                            ts=pend_ts,
                            text=result_text,
                            blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": result_text}}],
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update gate escalation message: {e}")
                elif event.action in ("block", "escalate", "alert", "halt_session"):
                    # New gate decision (not resolving a pending escalation) — post to channel
                    icon = {"block": "🛑", "escalate": "⚠️", "alert": "🔔", "halt_session": "🚨"}.get(event.action, "❓")
                    text = f"{icon} *Gate {event.action.upper()}*: `{event.tool_name}`"
                    if event.reason:
                        text += f"\n>{event.reason[:300]}"
                    text += _unthreaded_note
                    if event.tool_input:
                        input_str = json.dumps(event.tool_input, indent=2)
                        if len(input_str) > 500:
                            input_str = input_str[:500] + "\n..."
                        text += f"\n```\n{input_str}\n```"

                    msg_kwargs: dict = {
                        "channel": channel_id,
                        "thread_ts": thread_ts,
                        "text": text,
                        "reply_broadcast": event.action in ("block", "escalate", "halt_session"),
                    }

                    # Add Allow/Deny buttons for escalations
                    if event.action == "escalate":
                        msg_kwargs["blocks"] = [
                            {"type": "section", "text": {"type": "mrkdwn", "text": text}},
                            {
                                "type": "actions",
                                "elements": [
                                    {
                                        "type": "button",
                                        "text": {"type": "plain_text", "text": "Allow"},
                                        "action_id": "gate_approve",
                                        "value": event.tool_use_id,
                                        "style": "primary",
                                    },
                                    {
                                        "type": "button",
                                        "text": {"type": "plain_text", "text": "Deny"},
                                        "action_id": "gate_deny",
                                        "value": event.tool_use_id,
                                        "style": "danger",
                                    },
                                ],
                            },
                        ]

                    resp = self._web.chat_postMessage(**msg_kwargs)
                    if resp.get("ok"):
                        msg_ts = resp["ts"]
                        if event.action == "escalate":
                            self._gate_pending[event.tool_use_id] = (channel_id, msg_ts)
                        # Track for conversation follow-ups
                        conv_id = f"gate:{event.tool_use_id}"
                        with self._lock:
                            self._conv_threads[msg_ts] = conv_id
                            # Also map the session thread parent ts so replies
                            # within the session thread route to this conversation
                            # (most recent gate/sentinel wins per thread)
                            if thread_ts:
                                self._conv_threads[thread_ts] = conv_id

            case SentinelReviewEvent() if event.action in ("alert", "escalate", "halt_session", "error"):
                icon = {"alert": "🔔", "escalate": "⚠️", "halt_session": "🚨", "error": "❌"}.get(event.action, "❓")
                text = f"{icon} *Sentinel {event.action.upper()}* ({event.event_count} events reviewed)"
                if event.summary:
                    text += f"\n*Summary:* {event.summary[:300]}"
                if event.concerns and event.concerns.lower() != "none":
                    text += f"\n*Concerns:* {event.concerns[:500]}"
                text += _unthreaded_note
                resp = self._web.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text=text,
                    reply_broadcast=event.action in ("escalate", "halt_session"),
                )
                # Track for conversation follow-ups
                if resp.get("ok") and event.review_id:
                    conv_id = f"sentinel:{event.review_id}"
                    with self._lock:
                        self._conv_threads[resp["ts"]] = conv_id
                        if thread_ts:
                            self._conv_threads[thread_ts] = conv_id

            case PermissionPromptEvent():
                self._post_permission_prompt(
                    event.session_id,
                    event.tool_desc,
                    event.allow_all_label,
                )

            case PermissionResolvedEvent() if not event.resolver.startswith("slack"):
                # Permission resolved from another surface (dashboard, terminal, CLI)
                # Update the Slack message if we had one pending
                perm_info = self._permission_pending.pop(event.session_id, None)
                if perm_info:
                    perm_channel, perm_ts = perm_info
                    if event.action == "deny":
                        result_text = f"❌ *Denied* ({event.resolver})"
                    elif event.action == "allow_all":
                        result_text = f"✅ *Approved (allow all)* ({event.resolver})"
                    else:
                        result_text = f"✅ *Approved* ({event.resolver})"
                    try:
                        self._web.chat_update(
                            channel=perm_channel,
                            ts=perm_ts,
                            text=result_text,
                            blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": result_text}}],
                        )
                    except Exception as e:
                        logger.warning(f"Failed to update permission message: {e}")

            case ErrorEvent():
                text = f"*Error {event.status_code}*\n```\n{event.body[:500]}\n```"
                text += _unthreaded_note
                self._web.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text=text,
                )

    # --- Incoming Slack messages ---

    def _on_socket_event(self, client: SocketModeClient, req: SocketModeRequest):
        """Handle incoming Slack events (messages and interactive buttons)."""
        client.send_socket_mode_response(SocketModeResponse(envelope_id=req.envelope_id))

        if req.type == "interactive":
            self._handle_interactive(req.payload)
            return

        if req.type != "events_api":
            return

        event = req.payload.get("event", {})
        if event.get("type") != "message":
            return

        # Ignore bot's own messages
        if event.get("user") == self._bot_user_id:
            return
        if event.get("bot_id"):
            return

        text = event.get("text", "").strip()
        if not text:
            return

        thread_ts = event.get("thread_ts")
        channel_id = event.get("channel")

        # Non-threaded message mentioning the bot → spawn new session
        if not thread_ts:
            if not self._spawn_callback or not self._bot_user_id:
                return
            # Require @mention of the bot
            if f"<@{self._bot_user_id}>" not in event.get("text", ""):
                return
            # Strip the mention from the message
            text = text.replace(f"<@{self._bot_user_id}>", "").strip()
            if not text:
                return
            # Look up project from channel
            project = self._project_for_channel(channel_id)
            if not project:
                return
            logger.info(f"Slack new session in {project}: {text[:100]}")
            msg_ts = event.get("ts")
            self._web.reactions_add(
                channel=channel_id,
                timestamp=msg_ts,
                name="robot_face",
            )
            # Store thread info so session threads under the original message
            self._pending_thread_ts[project] = (channel_id, msg_ts)
            import asyncio

            if self._event_loop:
                asyncio.run_coroutine_threadsafe(
                    self._spawn_callback(project, text),
                    self._event_loop,
                )
            return

        # Check if this thread is a conversation thread (gate/sentinel follow-up)
        with self._lock:
            conv_id = self._conv_threads.get(thread_ts)
        if conv_id and self._conversation_store:
            import asyncio

            async def _do_conv():
                reply = await self._conversation_store.send_message(conv_id, text)
                if reply:
                    self._web.chat_postMessage(
                        channel=channel_id,
                        thread_ts=thread_ts,
                        text=reply,
                    )

            if self._event_loop:
                asyncio.run_coroutine_threadsafe(_do_conv(), self._event_loop)
            return

        # Find which session this thread belongs to
        with self._lock:
            session_id = None
            for sid, (ch_id, ts) in self._threads.items():
                if ts == thread_ts:
                    session_id = sid
                    break

        if not session_id:
            return

        if not self._inject_callback:
            logger.warning("No inject callback configured")
            return

        # If permission is pending, treat message as "deny + feedback"
        if session_id in self._permission_pending:
            logger.info(f"Slack -> session {session_id}: deny + feedback: {text[:100]}")
            self._permission_pending.pop(session_id, None)
            # Deny the permission prompt, wait for Claude to process, then send feedback
            self._inject(session_id, "3")
            import time

            time.sleep(0.5)
            self._inject(session_id, text + "\r")

            # Update the permission message in the thread
            with self._lock:
                thread_info_inner = self._threads.get(session_id)
            if thread_info_inner:
                ch, ts = thread_info_inner
                self._web.chat_postMessage(
                    channel=ch,
                    thread_ts=ts,
                    text=f"❌ *Denied* with feedback: {text}",
                )
            return

        # Regular message injection
        logger.info(f"Slack -> session {session_id}: {text[:100]}")
        self._inject(session_id, text + "\r")

    def _handle_interactive(self, payload: dict):
        """Handle interactive button clicks (permission approve/deny)."""
        if payload.get("type") != "block_actions":
            return

        actions = payload.get("actions", [])
        if not actions:
            return

        action = actions[0]
        action_id = action.get("action_id", "")
        session_id = action.get("value", "")
        user_name = payload.get("user", {}).get("username", "someone")
        channel_id = payload.get("channel", {}).get("id")
        message_ts = payload.get("message", {}).get("ts")

        # --- Gate approval buttons ---
        if action_id in ("gate_approve", "gate_deny"):
            tool_use_id = action.get("value", "")
            approved = action_id == "gate_approve"
            result_text = f"✅ *Approved* by @{user_name}" if approved else f"❌ *Denied* by @{user_name}"
            logger.info(f"Gate {'approved' if approved else 'denied'} by {user_name} ({tool_use_id})")

            # Remove from pending so the EventBus handler doesn't double-update
            self._gate_pending.pop(tool_use_id, None)

            # Signal the proxy's waiting approval event (thread-safe)
            if self._resolve_approval_callback and self._event_loop:
                self._event_loop.call_soon_threadsafe(self._resolve_approval_callback, tool_use_id, approved, user_name)

            # Update message: keep original context, replace buttons with result
            if channel_id and message_ts:
                try:
                    original_blocks = payload.get("message", {}).get("blocks", [])
                    # Keep all non-actions blocks (the command/analysis context)
                    updated_blocks = [b for b in original_blocks if b.get("type") != "actions"]
                    updated_blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": result_text}})
                    self._web.chat_update(
                        channel=channel_id,
                        ts=message_ts,
                        text=result_text,
                        blocks=updated_blocks,
                    )
                except Exception as e:
                    logger.warning(f"Failed to update gate approval message: {e}")
            return

        # --- Permission buttons (PTY session) ---
        # value is session_id for these
        session_id = action.get("value", "")
        if action_id == "permission_approve":
            perm_action = "approve"
            logger.info(f"Permission approved by {user_name} for session {session_id}")
            result_text = f"✅ *Approved* by @{user_name}"
        elif action_id == "permission_allow_all":
            perm_action = "allow_all"
            logger.info(f"Permission allow-all by {user_name} for session {session_id}")
            result_text = f"✅ *Approved (allow all)* by @{user_name}"
        elif action_id == "permission_deny":
            perm_action = "deny"
            logger.info(f"Permission denied by {user_name} for session {session_id}")
            result_text = f"❌ *Denied* by @{user_name}"
        else:
            return

        self._permission_pending.pop(session_id, None)

        # Resolve via daemon callback (handles PTY injection + event publishing)
        if self._resolve_permission_callback:
            self._resolve_permission_callback(session_id, perm_action, f"slack (@{user_name})")
        else:
            # Fallback: inject directly if no callback
            key = {"approve": "1", "allow_all": "2", "deny": "3"}.get(perm_action, "3")
            self._inject(session_id, key)

        # Update the message to replace buttons with the result
        if channel_id and message_ts:
            try:
                self._web.chat_update(
                    channel=channel_id,
                    ts=message_ts,
                    text=result_text,
                    blocks=[{"type": "section", "text": {"type": "mrkdwn", "text": result_text}}],
                )
            except Exception as e:
                logger.warning(f"Failed to update permission message: {e}")

    def _inject(self, session_id: str, text: str):
        """Inject text into a session's PTY via the daemon callback."""
        if not self._inject_callback:
            logger.warning("No inject callback configured")
            return
        import asyncio

        if self._event_loop:
            asyncio.run_coroutine_threadsafe(
                self._inject_callback(session_id, text),
                self._event_loop,
            )
        else:
            logger.warning("No event loop available for injection")

    # --- Channel management ---

    def _ensure_channel(self, project: str) -> str:
        """Find or create the private channel for this project/user."""
        name = self._channel_name(project)

        if name in self._channels:
            return self._channels[name]

        channel_id = self._find_channel(name)
        if channel_id:
            self._channels[name] = channel_id
            self._ensure_users_invited(channel_id)
            return channel_id

        # Channel not found — try to create it
        channel_id = self._try_create_channel(name)
        if channel_id:
            self._channels[name] = channel_id
            self._ensure_users_invited(channel_id)
            return channel_id

        raise RuntimeError(f"Could not find or create channel #{name}")

    def _try_create_channel(self, name: str) -> str | None:
        """Try to create a channel, handling name_taken by looking it up."""
        for is_private in (True, False):
            try:
                resp = self._web.conversations_create(
                    name=name,
                    is_private=is_private,
                )
                channel_id = resp["channel"]["id"]
                logger.info(f"Created {'private' if is_private else 'public'} channel #{name} ({channel_id})")
                return channel_id
            except Exception as e:
                error_str = str(e)
                if "name_taken" in error_str:
                    # Channel exists but _find_channel missed it (ratelimit, pagination, etc.)
                    # Look it up directly via conversations_list with name search
                    channel_id = self._find_channel(name)
                    if channel_id:
                        return channel_id
                    # If still not found, log and return None
                    logger.warning(f"Channel #{name} exists but could not be found via API")
                    return None
                if is_private:
                    logger.warning(f"Failed to create private channel #{name}: {e}, trying public")
                    continue
                logger.warning(f"Failed to create channel #{name}: {e}")
        return None

    def _ensure_users_invited(self, channel_id: str):
        """Ensure non-bot workspace users are members of the channel."""
        if not self._user_ids:
            return
        try:
            members_resp = self._web.conversations_members(channel=channel_id)
            current_members = set(members_resp.get("members", []))
            missing = [uid for uid in self._user_ids if uid not in current_members]
            if missing:
                self._web.conversations_invite(
                    channel=channel_id,
                    users=",".join(missing),
                )
                logger.info(f"Invited {len(missing)} user(s) to channel {channel_id}")
        except Exception as e:
            logger.warning(f"Failed to ensure users invited to {channel_id}: {e}")

    def _find_channel(self, name: str) -> str | None:
        """Search for an existing channel by name."""
        try:
            for channel_type in ("private_channel", "public_channel"):
                resp = self._web.conversations_list(types=channel_type, limit=200)
                for ch in resp.get("channels", []):
                    if ch["name"] == name:
                        return ch["id"]
        except Exception as e:
            logger.warning(f"Error searching channels: {e}")
        return None

    def _project_for_channel(self, channel_id: str) -> str | None:
        """Reverse lookup: channel ID -> project name."""
        name = None

        # Check cache first
        with self._lock:
            for cached_name, cid in self._channels.items():
                if cid == channel_id:
                    name = cached_name
                    break

        # Cache miss — look up from Slack API
        if not name:
            try:
                info = self._web.conversations_info(channel=channel_id)
                name = info["channel"]["name"]
                with self._lock:
                    self._channels[name] = channel_id
            except Exception as e:
                logger.warning(f"Failed to look up channel {channel_id}: {e}")
                return None

        # Channel name is "{base}[-{project}][-{username}]" — extract project
        base_slug = _slugify(settings.slack_channel_base)
        if not name.startswith(base_slug):
            return None
        remainder = name[len(base_slug) :].strip("-")
        if not remainder:
            return None
        parts = remainder.split("-")
        # Strip username suffix if it matches the known user
        if self._username and parts and parts[-1] == _slugify(self._username):
            parts = parts[:-1]
        return "-".join(parts) if parts else None

    def _channel_name(self, project: str) -> str:
        """Generate channel name: {base}[-{project}][-{username}]."""
        parts = [_slugify(settings.slack_channel_base)]
        if settings.slack_channel_append_project:
            parts.append(_slugify(project))
        if settings.slack_channel_append_user and self._username:
            parts.append(_slugify(self._username))
        return "-".join(parts)[:80]


def _slugify(text: str) -> str:
    """Convert text to a Slack-safe channel name component."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9-]", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")
