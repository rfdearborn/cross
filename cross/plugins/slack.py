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
    RequestEvent,
    TextEvent,
    ToolUseEvent,
)

logger = logging.getLogger("cross.plugins.slack")


class SlackPlugin:
    """Manages Slack channels/threads and relays agent events."""

    def __init__(self, inject_callback=None, spawn_callback=None, event_loop=None):
        self._web = WebClient(token=settings.slack_bot_token)
        self._socket: SocketModeClient | None = None
        self._event_loop = event_loop
        self._spawn_callback = spawn_callback
        # channel name -> channel ID
        self._channels: dict[str, str] = {}
        # session_id -> (channel_id, thread_ts)
        self._threads: dict[str, tuple[str, str]] = {}
        # session_id -> session data dict
        self._sessions: dict[str, dict] = {}
        # bot user ID (resolved on connect)
        self._bot_user_id: str | None = None
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
        # Track messages injected from Slack to suppress proxy echo
        self._injected_texts: dict[str, str] = {}  # session_id -> last injected text
        # Slack-initiated sessions: project -> (channel_id, message_ts) for threading
        self._pending_thread_ts: dict[str, tuple[str, str]] = {}

    def start(self):
        """Connect to Slack via Socket Mode."""
        # Resolve bot user ID
        auth = self._web.auth_test()
        self._bot_user_id = auth["user_id"]
        logger.info(f"Slack connected as {auth['user']} ({self._bot_user_id})")

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

        Only posts high-signal events (permission prompts) to Slack.
        Regular output is already captured by the network proxy.
        """
        with self._lock:
            thread_info = self._threads.get(session_id)

        if not thread_info:
            return
        channel_id, thread_ts = thread_info

        # Check for permission prompts — require the specific Claude Code prompt
        # structure, not broad patterns that match TUI redraws
        if not _is_permission_prompt(text):
            return

        # Debounce — don't spam channel for the same permission prompt
        import time
        now = time.time()
        last = self._last_permission_post.get(session_id, 0)
        if now - last < self._PERMISSION_DEBOUNCE_SECS:
            return
        self._last_permission_post[session_id] = now

        # Build permission message with interactive buttons
        tool_desc = self._last_tool_desc.get(session_id, "")
        prompt_text = "⚠️ *Permission needed*"
        if tool_desc:
            prompt_text += f" for {tool_desc}"

        # Extract the "allow all" option text from the PTY output
        allow_all_label = _extract_allow_all(text) or "Allow all (session)"

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
        # Clear stale tool description so it can't be reused for a false match
        self._last_tool_desc.pop(session_id, None)

    # --- EventBus handler (proxy events) ---

    async def handle_event(self, event: CrossEvent):
        """Handle events from the network proxy EventBus.

        Called from async context (proxy), but we use sync WebClient
        so it's safe to call from any thread.
        """
        with self._lock:
            if not self._threads:
                return
            # Post to the most recently active session's thread
            # TODO: correlate proxy events to specific sessions
            session_id = list(self._threads.keys())[-1]
            thread_info = self._threads.get(session_id)

        if not thread_info:
            return
        channel_id, thread_ts = thread_info

        match event:
            case RequestEvent() if event.stream and event.last_message_role == "user":
                # Only relay user messages from real conversation turns (streaming),
                # not internal system requests (non-streaming haiku calls)
                preview = event.last_message_preview or ""
                if preview and not preview.startswith("[tool_result"):
                    # Suppress echo of messages injected from Slack
                    last_injected = self._injected_texts.pop(session_id, None)
                    if last_injected and preview.startswith(last_injected[:100]):
                        return
                    if len(preview) > 3000:
                        preview = preview[:3000] + "\n..."
                    self._web.chat_postMessage(
                        channel=channel_id,
                        thread_ts=thread_ts,
                        text=f"*You:* {preview}",
                    )

            case ToolUseEvent():
                input_str = json.dumps(event.input, indent=2)
                if len(input_str) > 500:
                    input_str = input_str[:500] + "\n..."

                # Track tool description for permission prompt context
                tool_desc = f"`{event.name}`"
                if event.name == "Bash":
                    cmd = event.input.get("command", "")
                    tool_desc = f"`Bash`: `{cmd[:80]}`"
                elif event.name in ("Read", "Write", "Edit"):
                    path = event.input.get("file_path", "")
                    tool_desc = f"`{event.name}`: `{path}`"
                self._last_tool_desc[session_id] = tool_desc
                self._web.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text=f"*{event.name}*\n```\n{input_str}\n```",
                )

            case TextEvent():
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

                text = event.text
                if len(text) > 3000:
                    text = text[:3000] + "\n..."
                self._web.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text=f"*Claude:* {text}",
                )

            case ErrorEvent():
                self._web.chat_postMessage(
                    channel=channel_id,
                    thread_ts=thread_ts,
                    text=f"*Error {event.status_code}*\n```\n{event.body[:500]}\n```",
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
        self._injected_texts[session_id] = text
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

        if action_id == "permission_approve":
            # Claude Code's TUI acts on keypress immediately — no Enter needed
            self._inject(session_id, "1")
            logger.info(f"Permission approved by {user_name} for session {session_id}")
            result_text = f"✅ *Approved* by @{user_name}"
        elif action_id == "permission_allow_all":
            self._inject(session_id, "2")
            logger.info(f"Permission allow-all by {user_name} for session {session_id}")
            result_text = f"✅ *Approved (allow all)* by @{user_name}"
        elif action_id == "permission_deny":
            self._inject(session_id, "3")
            logger.info(f"Permission denied by {user_name} for session {session_id}")
            result_text = f"❌ *Denied* by @{user_name}"
        else:
            return

        self._permission_pending.pop(session_id, None)

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
            return channel_id

        try:
            resp = self._web.conversations_create(
                name=name,
                is_private=True,
            )
            channel_id = resp["channel"]["id"]
            self._channels[name] = channel_id
            logger.info(f"Created channel #{name} ({channel_id})")
            self._invite_workspace_users(channel_id)
            return channel_id
        except Exception as e:
            logger.warning(f"Failed to create private channel #{name}: {e}, trying public")
            resp = self._web.conversations_create(name=name)
            channel_id = resp["channel"]["id"]
            self._channels[name] = channel_id
            return channel_id

    def _invite_workspace_users(self, channel_id: str):
        """Invite non-bot workspace users to the channel."""
        try:
            resp = self._web.users_list()
            user_ids = [
                u["id"] for u in resp.get("members", [])
                if not u.get("is_bot") and not u.get("deleted") and u["id"] != "USLACKBOT"
                and u["id"] != self._bot_user_id
            ]
            if user_ids:
                self._web.conversations_invite(
                    channel=channel_id,
                    users=",".join(user_ids),
                )
                logger.info(f"Invited {len(user_ids)} user(s) to channel")
        except Exception as e:
            logger.warning(f"Failed to invite users: {e}")

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

        # Channel name is "{prefix}-cross-{project}" — extract project
        parts = name.split("-")
        try:
            idx = parts.index("cross")
            return "-".join(parts[idx + 1:])
        except ValueError:
            return None

    def _channel_name(self, project: str) -> str:
        """Generate channel name: {prefix}-cross-{project}."""
        parts = []
        if settings.slack_channel_prefix:
            parts.append(settings.slack_channel_prefix)
        parts.append("cross")
        parts.append(_slugify(project))
        return "-".join(parts)[:80]



def _is_permission_prompt(text: str) -> bool:
    """Detect actual Claude Code permission prompts vs TUI redraws.

    Requires the specific "Do you want to" prompt structure that Claude Code
    always shows, rather than broad pattern matching that catches redraws.
    """
    # "Do you want to" — may appear garbled as "Doyouwant" or "Do you want t"
    if re.search(r"Do\s*you\s*want\s*t", text, re.IGNORECASE):
        return True
    # Numbered options structure: "1. Yes" + "3. No" together
    if re.search(r"1\.\s*Yes", text) and re.search(r"3\.\s*No", text):
        return True
    return False


def _extract_allow_all(text: str) -> str | None:
    """Extract the 'allow all...' option text from PTY output.

    PTY text may have spaces preserved or garbled together (no spaces),
    so we try both patterns.
    """
    # Clean text: "allow all edits in Downloads/"
    m = re.search(r"allow all (\w+) in (\S+/)", text, re.IGNORECASE)
    if m:
        return f"Allow all {m.group(1)} in {m.group(2)}"

    # Garbled text (no spaces): "allowalleditsinDownloads/"
    # Non-greedy \w+? backtracks until "in" matches
    m = re.search(r"allowall(\w+?)in(\S+?/)", text, re.IGNORECASE)
    if m:
        return f"Allow all {m.group(1)} in {m.group(2)}"

    # Bash commands (no directory)
    if re.search(r"allow\s*all\s*bash", text, re.IGNORECASE):
        return "Allow all Bash"

    return None


def _slugify(text: str) -> str:
    """Convert text to a Slack-safe channel name component."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9-]", "-", text)
    text = re.sub(r"-+", "-", text)
    return text.strip("-")
