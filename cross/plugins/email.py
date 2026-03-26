"""Email relay plugin — mirrors agent sessions to email threads."""

from __future__ import annotations

import asyncio
import email.mime.text
import email.utils
import json
import logging
import re
import smtplib
import threading
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from imaplib import IMAP4, IMAP4_SSL

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

logger = logging.getLogger("cross.plugins.email")


class EmailPlugin:
    """Manages email threads and relays agent events, mirroring SlackPlugin's threading model."""

    def __init__(
        self,
        inject_callback=None,
        resolve_approval_callback=None,
        resolve_permission_callback=None,
        event_loop=None,
        conversation_store=None,
    ):
        self._event_loop = event_loop
        self._inject_callback = inject_callback
        self._resolve_approval_callback = resolve_approval_callback
        self._resolve_permission_callback = resolve_permission_callback
        self._conversation_store = conversation_store

        # Thread tracking: session_id -> message_id (for In-Reply-To threading)
        self._threads: dict[str, str] = {}
        # session_id -> session data dict
        self._sessions: dict[str, dict] = {}
        self._lock = threading.Lock()

        # Debounce permission prompts
        self._last_permission_post: dict[str, float] = {}
        self._PERMISSION_DEBOUNCE_SECS = 3.0
        # Track last tool use per session (for permission context)
        self._last_tool_desc: dict[str, str] = {}
        # Sessions with pending permission prompts: session_id -> message_id
        self._permission_pending: dict[str, str] = {}
        # Pending gate escalations: tool_use_id -> message_id
        self._gate_pending: dict[str, str] = {}
        # Conversation threads: message_id -> conversation_id
        self._conv_threads: dict[str, str] = {}

        # IMAP polling state
        self._imap_thread: threading.Thread | None = None
        self._imap_stop = threading.Event()

    def start(self):
        """Start the IMAP polling thread for incoming replies."""
        if settings.email_imap_host:
            self._imap_thread = threading.Thread(target=self._poll_imap, daemon=True)
            self._imap_thread.start()
            logger.info("Email IMAP polling started")
        else:
            logger.info("Email plugin started (outbound only, no IMAP configured)")

    def stop(self):
        """Stop the IMAP polling thread."""
        self._imap_stop.set()
        if self._imap_thread:
            self._imap_thread.join(timeout=5)

    # --- Session lifecycle ---

    def session_started_from_data(self, data: dict):
        """Send session-start email and create a thread."""
        session_id = data["session_id"]
        project = data.get("project", "unknown")
        agent = data.get("agent", "agent")
        cwd = data.get("cwd", "")

        with self._lock:
            self._sessions[session_id] = data

        subject = f"[cross] {agent} session started in {project}"
        body = f"{agent} session started in {project}\nWorking directory: {cwd}\nSession ID: {session_id}"

        message_id = self._send_email(subject, body, thread_session_id=session_id)
        if message_id:
            with self._lock:
                self._threads[session_id] = message_id

    def session_ended_from_data(self, data: dict):
        """Send session-end email in the thread."""
        session_id = data.get("session_id", "")
        with self._lock:
            thread_msg_id = self._threads.get(session_id)
            self._sessions.pop(session_id, None)

        if not thread_msg_id:
            return

        exit_code = data.get("exit_code", "?")
        started = data.get("started_at", 0)
        ended = data.get("ended_at", 0)
        duration = ""
        if started and ended:
            mins = int((ended - started) / 60)
            duration = f" after {mins}m" if mins > 0 else ""

        subject = "Re: [cross] session update"
        body = f"Session ended (exit code {exit_code}){duration}"
        self._send_email(subject, body, in_reply_to=thread_msg_id, thread_session_id=session_id)

    # --- PTY output handler ---

    def handle_pty_output(self, session_id: str, text: str):
        """Handle cleaned PTY output from a wrap process.

        Permission prompt detection is now centralized in the daemon
        (via _check_permission_prompt → PermissionPromptEvent). This
        method is kept for future PTY-based signals.
        """

    def _post_permission_prompt(self, session_id: str, tool_desc: str, allow_all_label: str):
        """Send a permission prompt email (called from event handler)."""
        with self._lock:
            thread_msg_id = self._threads.get(session_id)

        if not thread_msg_id:
            return

        prompt_text = "Permission needed"
        if tool_desc:
            prompt_text += f" for {tool_desc}"

        subject = "Re: [cross] Permission needed"
        body = (
            f"⚠️ {prompt_text}\n\n"
            f"Reply with one of:\n"
            f"  APPROVE - approve this action\n"
            f"  ALLOW ALL - {allow_all_label}\n"
            f"  DENY - deny this action\n\n"
            f"Or reply with any text to deny and send feedback.\n\n"
            f"Session: {session_id}"
        )

        msg_id = self._send_email(subject, body, in_reply_to=thread_msg_id, thread_session_id=session_id)
        if msg_id:
            self._permission_pending[session_id] = msg_id

    # --- EventBus handler ---

    async def handle_event(self, event: CrossEvent):
        """Handle events from the network proxy EventBus."""
        event_session = getattr(event, "session_id", "")

        with self._lock:
            if event_session and event_session in self._threads:
                session_id = event_session
                thread_msg_id = self._threads[session_id]
            elif self._threads:
                session_id = list(self._threads.keys())[-1]
                thread_msg_id = self._threads.get(session_id)
            else:
                session_id = None
                thread_msg_id = None

        match event:
            case ToolUseEvent() if session_id:
                tool_desc = f"`{event.name}`"
                if event.name == "Bash":
                    cmd = event.input.get("command", "")
                    tool_desc = f"`Bash`: `{cmd[:80]}`"
                elif event.name in ("Read", "Write", "Edit"):
                    path = event.input.get("file_path", "")
                    tool_desc = f"`{event.name}`: `{path}`"
                self._last_tool_desc[session_id] = tool_desc

            case TextEvent() if session_id:
                if session_id in self._permission_pending:
                    self._permission_pending.pop(session_id)
                    # Permission resolved from terminal — no email update needed
                    # (no way to "edit" a sent email like Slack's chat_update)

            case GateDecisionEvent() if event.action in ("block", "escalate", "alert", "allow", "halt_session"):
                pending_msg_id = self._gate_pending.pop(event.tool_use_id, None)
                if pending_msg_id and event.action != "escalate":
                    # Resolved externally — send follow-up
                    m = re.search(r"@(\w+)", event.reason or "")
                    username = m.group(1) if m else "unknown"
                    if event.action == "allow":
                        body = f"✅ Approved by @{username}"
                    else:
                        body = f"❌ Denied by @{username}"
                    self._send_email(
                        "Re: [cross] Gate decision update",
                        body,
                        in_reply_to=pending_msg_id,
                        thread_session_id=session_id,
                    )
                elif event.action in ("block", "escalate", "alert", "halt_session"):
                    icon = {"block": "🛑", "escalate": "⚠️", "alert": "🔔", "halt_session": "🚨"}.get(event.action, "❓")
                    body = f"{icon} Gate {event.action.upper()}: {event.tool_name}"
                    if event.reason:
                        body += f"\nReason: {event.reason[:300]}"
                    if event.tool_input:
                        input_str = json.dumps(event.tool_input, indent=2)
                        if len(input_str) > 500:
                            input_str = input_str[:500] + "\n..."
                        body += f"\n\n{input_str}"

                    if event.action == "escalate":
                        body += (
                            "\n\nReply with:\n"
                            "  APPROVE - allow this action\n"
                            "  DENY - block this action\n"
                            f"\nTool Use ID: {event.tool_use_id}"
                        )

                    reply_to = thread_msg_id if thread_msg_id else None
                    msg_id = self._send_email(
                        f"Re: [cross] Gate {event.action.upper()}: {event.tool_name}",
                        body,
                        in_reply_to=reply_to,
                        thread_session_id=session_id,
                    )
                    if msg_id:
                        if event.action == "escalate":
                            self._gate_pending[event.tool_use_id] = msg_id
                        # Track for conversation follow-ups
                        with self._lock:
                            self._conv_threads[msg_id] = f"gate:{event.tool_use_id}"

            case SentinelReviewEvent() if event.action in ("alert", "escalate", "halt_session", "error"):
                icon = {"alert": "🔔", "escalate": "⚠️", "halt_session": "🚨", "error": "❌"}.get(event.action, "❓")
                body = f"{icon} Sentinel {event.action.upper()} ({event.event_count} events reviewed)"
                if event.summary:
                    body += f"\nSummary: {event.summary[:300]}"
                if event.concerns and event.concerns.lower() != "none":
                    body += f"\nConcerns: {event.concerns[:500]}"
                msg_id = self._send_email(
                    f"Re: [cross] Sentinel {event.action.upper()}",
                    body,
                    in_reply_to=thread_msg_id,
                    thread_session_id=session_id,
                )
                # Track for conversation follow-ups
                if msg_id and event.review_id:
                    with self._lock:
                        self._conv_threads[msg_id] = f"sentinel:{event.review_id}"

            case PermissionPromptEvent():
                self._post_permission_prompt(
                    event.session_id,
                    event.tool_desc,
                    event.allow_all_label,
                )

            case PermissionResolvedEvent() if not event.resolver.startswith("email"):
                perm_msg_id = self._permission_pending.pop(event.session_id, None)
                if perm_msg_id:
                    if event.action == "deny":
                        body = f"❌ Denied ({event.resolver})"
                    elif event.action == "allow_all":
                        body = f"✅ Approved (allow all) ({event.resolver})"
                    else:
                        body = f"✅ Approved ({event.resolver})"
                    self._send_email(
                        "Re: [cross] Permission resolved",
                        body,
                        in_reply_to=perm_msg_id,
                        thread_session_id=event.session_id,
                    )

            case ErrorEvent():
                self._send_email(
                    "Re: [cross] Error",
                    f"Error {event.status_code}\n\n{event.body[:500]}",
                    in_reply_to=thread_msg_id,
                    thread_session_id=session_id,
                )

    # --- Email sending ---

    def _send_email(
        self,
        subject: str,
        body: str,
        in_reply_to: str | None = None,
        thread_session_id: str | None = None,
    ) -> str | None:
        """Send an email and return the Message-ID. Returns None on failure."""
        msg = MIMEMultipart("alternative")
        msg["From"] = settings.email_from
        msg["To"] = settings.email_to
        msg["Subject"] = subject
        domain = settings.email_from.split("@")[-1] if "@" in settings.email_from else "cross.local"
        message_id = email.utils.make_msgid(domain=domain)
        msg["Message-ID"] = message_id

        if in_reply_to:
            msg["In-Reply-To"] = in_reply_to
            msg["References"] = in_reply_to

        # Thread grouping via References header chain
        if thread_session_id:
            with self._lock:
                root_msg_id = self._threads.get(thread_session_id)
            if root_msg_id and root_msg_id != in_reply_to:
                existing_refs = msg.get("References", "")
                msg.replace_header("References", f"{root_msg_id} {existing_refs}".strip()) if existing_refs else None
                if not existing_refs:
                    msg["References"] = root_msg_id

        msg.attach(MIMEText(body, "plain"))

        # HTML version for richer rendering
        html_body = _text_to_html(body)
        msg.attach(MIMEText(html_body, "html"))

        try:
            if settings.email_smtp_ssl:
                smtp = smtplib.SMTP_SSL(settings.email_smtp_host, settings.email_smtp_port, timeout=10)
            else:
                smtp = smtplib.SMTP(settings.email_smtp_host, settings.email_smtp_port, timeout=10)
                if settings.email_smtp_starttls:
                    smtp.starttls()

            if settings.email_smtp_username and settings.email_smtp_password:
                smtp.login(settings.email_smtp_username, settings.email_smtp_password)

            smtp.sendmail(settings.email_from, [settings.email_to], msg.as_string())
            smtp.quit()
            logger.debug(f"Email sent: {subject} ({message_id})")
            return message_id
        except Exception as e:
            logger.warning(f"Failed to send email: {e}")
            return None

    # --- IMAP polling (inbound replies) ---

    def _poll_imap(self):
        """Poll IMAP for replies to cross emails. Runs in a background thread."""
        while not self._imap_stop.is_set():
            try:
                self._check_imap()
            except Exception as e:
                logger.warning(f"IMAP poll error: {e}")
            self._imap_stop.wait(timeout=settings.email_imap_poll_interval)

    def _check_imap(self):
        """Check IMAP for new replies and process them."""
        try:
            if settings.email_imap_ssl:
                imap = IMAP4_SSL(settings.email_imap_host, settings.email_imap_port)
            else:
                imap = IMAP4(settings.email_imap_host, settings.email_imap_port)

            imap.login(
                settings.email_imap_username or settings.email_smtp_username,
                settings.email_imap_password or settings.email_smtp_password,
            )
            imap.select("INBOX")

            # Search for unread replies to cross emails
            _, data = imap.search(None, '(UNSEEN SUBJECT "[cross]")')
            if not data[0]:
                imap.logout()
                return

            for num in data[0].split():
                try:
                    self._process_imap_message(imap, num)
                except Exception as e:
                    logger.warning(f"Failed to process IMAP message {num}: {e}")

            imap.logout()
        except Exception as e:
            logger.warning(f"IMAP connection error: {e}")

    def _process_imap_message(self, imap: IMAP4 | IMAP4_SSL, msg_num: bytes):
        """Process a single IMAP message (reply)."""
        import email as email_mod

        _, data = imap.fetch(msg_num, "(RFC822)")
        raw = data[0][1]
        msg = email_mod.message_from_bytes(raw)

        # Extract reply text
        body = ""
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    body = part.get_payload(decode=True).decode("utf-8", errors="replace")
                    break
        else:
            body = msg.get_payload(decode=True).decode("utf-8", errors="replace")

        # Strip quoted text (lines starting with >)
        lines = body.strip().splitlines()
        reply_lines = []
        for line in lines:
            if line.startswith(">") or line.startswith("On ") and "wrote:" in line:
                break
            reply_lines.append(line)
        reply_text = "\n".join(reply_lines).strip()

        if not reply_text:
            return

        in_reply_to = msg.get("In-Reply-To", "")
        from_addr = email.utils.parseaddr(msg.get("From", ""))[1]

        # Check if this is a reply to a conversation thread (gate/sentinel follow-up)
        with self._lock:
            conv_id = self._conv_threads.get(in_reply_to)
        if conv_id and self._conversation_store and self._event_loop:

            async def _do_conv():
                reply = await self._conversation_store.send_message(conv_id, reply_text)
                if reply:
                    self._send_email(
                        "Re: [cross] Follow-up",
                        reply,
                        in_reply_to=in_reply_to,
                    )

            asyncio.run_coroutine_threadsafe(_do_conv(), self._event_loop)
            return

        # Determine action from reply text
        reply_upper = reply_text.upper().strip()

        # Check for gate approval replies
        tool_use_id = self._find_gate_for_reply(in_reply_to)
        if tool_use_id:
            approved = reply_upper in ("APPROVE", "YES", "ALLOW")
            denied = reply_upper in ("DENY", "NO", "BLOCK")
            if approved or denied:
                self._gate_pending.pop(tool_use_id, None)
                if self._resolve_approval_callback and self._event_loop:
                    self._event_loop.call_soon_threadsafe(
                        self._resolve_approval_callback, tool_use_id, approved, from_addr
                    )
                return

        # Check for permission replies
        session_id = self._find_session_for_reply(in_reply_to)
        if not session_id:
            return

        if session_id in self._permission_pending:
            if reply_upper in ("APPROVE", "YES"):
                self._permission_pending.pop(session_id, None)
                if self._resolve_permission_callback:
                    self._resolve_permission_callback(session_id, "approve", f"email ({from_addr})")
                else:
                    self._inject(session_id, "1")
                return
            elif reply_upper in ("ALLOW ALL", "ALLOW-ALL"):
                self._permission_pending.pop(session_id, None)
                if self._resolve_permission_callback:
                    self._resolve_permission_callback(session_id, "allow_all", f"email ({from_addr})")
                else:
                    self._inject(session_id, "2")
                return
            elif reply_upper in ("DENY", "NO"):
                self._permission_pending.pop(session_id, None)
                if self._resolve_permission_callback:
                    self._resolve_permission_callback(session_id, "deny", f"email ({from_addr})")
                else:
                    self._inject(session_id, "3")
                return
            else:
                # Treat as deny + feedback
                self._permission_pending.pop(session_id, None)
                self._inject(session_id, "3")
                time.sleep(0.5)
                self._inject(session_id, reply_text + "\r")
                return

        # Regular message injection (like Slack threaded messages)
        if self._inject_callback:
            self._inject(session_id, reply_text + "\r")

    def _find_session_for_reply(self, in_reply_to: str) -> str | None:
        """Find the session_id for a given In-Reply-To message ID."""
        with self._lock:
            for session_id, msg_id in self._threads.items():
                if msg_id == in_reply_to:
                    return session_id
        # Also check permission pending messages
        for session_id, msg_id in self._permission_pending.items():
            if msg_id == in_reply_to:
                return session_id
        return None

    def _find_gate_for_reply(self, in_reply_to: str) -> str | None:
        """Find the tool_use_id for a given In-Reply-To message ID."""
        for tool_use_id, msg_id in self._gate_pending.items():
            if msg_id == in_reply_to:
                return tool_use_id
        return None

    def _inject(self, session_id: str, text: str):
        """Inject text into a session's PTY via the daemon callback."""
        if not self._inject_callback:
            logger.warning("No inject callback configured")
            return
        if self._event_loop:
            asyncio.run_coroutine_threadsafe(
                self._inject_callback(session_id, text),
                self._event_loop,
            )
        else:
            logger.warning("No event loop available for injection")


def _text_to_html(text: str) -> str:
    """Convert plain text to simple HTML for email."""
    import html

    escaped = html.escape(text)
    # Convert emoji-prefixed lines to styled blocks
    escaped = escaped.replace("\n", "<br>\n")
    return f"<html><body style='font-family: monospace; font-size: 14px;'><p>{escaped}</p></body></html>"
