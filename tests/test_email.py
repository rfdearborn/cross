"""Tests for the Email relay plugin — threading, event handling, IMAP replies."""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from cross.events import (
    ErrorEvent,
    GateDecisionEvent,
    MessageDeltaEvent,
    MessageStartEvent,
    PermissionResolvedEvent,
    RequestEvent,
    SentinelReviewEvent,
    TextEvent,
    ToolUseEvent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SETTINGS_DEFAULTS = {
    "email_from": "cross@example.com",
    "email_to": "admin@example.com",
    "email_smtp_host": "smtp.example.com",
    "email_smtp_port": 587,
    "email_smtp_ssl": False,
    "email_smtp_starttls": True,
    "email_smtp_username": "cross@example.com",
    "email_smtp_password": "secret",
    "email_imap_host": "",
    "email_imap_port": 993,
    "email_imap_ssl": True,
    "email_imap_username": "",
    "email_imap_password": "",
    "email_imap_poll_interval": 30,
}


def _mock_settings(**overrides):
    vals = dict(_SETTINGS_DEFAULTS, **overrides)
    mock = MagicMock()
    for k, v in vals.items():
        setattr(mock, k, v)
    return mock


@pytest.fixture()
def email_env():
    """Yield a factory that creates EmailPlugin instances with mocked SMTP."""
    settings_mock = _mock_settings()

    with (
        patch("cross.plugins.email.settings", settings_mock),
        patch("cross.plugins.email.smtplib") as mock_smtplib,
    ):
        mock_smtp_instance = MagicMock()
        mock_smtplib.SMTP.return_value = mock_smtp_instance
        mock_smtplib.SMTP_SSL.return_value = mock_smtp_instance

        from cross.plugins.email import EmailPlugin

        def factory(inject_callback=None, event_loop=None, **kw):
            for k, v in kw.items():
                setattr(settings_mock, k, v)
            p = EmailPlugin(
                inject_callback=inject_callback,
                event_loop=event_loop,
            )
            return p, mock_smtp_instance

        yield factory, settings_mock, mock_smtplib


def _register_session(plugin, session_id="sess-1", project="myproj", agent="claude", cwd="/tmp/myproj"):
    """Helper to register a session."""
    plugin.session_started_from_data(
        {
            "session_id": session_id,
            "project": project,
            "agent": agent,
            "cwd": cwd,
        }
    )


def _get_email_body(mock_smtp) -> str:
    """Extract the plain-text body from the last sent email (handles base64 MIME encoding)."""
    import email as email_mod

    raw = mock_smtp.sendmail.call_args[0][2]
    msg = email_mod.message_from_string(raw)
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                return part.get_payload(decode=True).decode("utf-8")
    return msg.get_payload(decode=True).decode("utf-8")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


class TestIsPermissionPrompt:
    def test_do_you_want_to(self, email_env):
        from cross.plugins.email import _is_permission_prompt

        assert _is_permission_prompt("Do you want to allow this?") is True

    def test_not_a_prompt(self, email_env):
        from cross.plugins.email import _is_permission_prompt

        assert _is_permission_prompt("Compiling project...") is False


class TestExtractAllowAll:
    def test_clean_text(self, email_env):
        from cross.plugins.email import _extract_allow_all

        result = _extract_allow_all("2. allow all edits in Downloads/")
        assert result == "Allow all edits in Downloads/"

    def test_no_match(self, email_env):
        from cross.plugins.email import _extract_allow_all

        assert _extract_allow_all("some random text") is None


class TestTextToHtml:
    def test_basic(self, email_env):
        from cross.plugins.email import _text_to_html

        html = _text_to_html("Hello\nWorld")
        assert "<br>" in html
        assert "Hello" in html


# ---------------------------------------------------------------------------
# EmailPlugin.__init__
# ---------------------------------------------------------------------------


class TestEmailPluginInit:
    def test_init_creates_plugin(self, email_env):
        factory, _, _ = email_env
        plugin, _ = factory()
        assert plugin._imap_thread is None
        assert plugin._threads == {}


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------


class TestStartStop:
    def test_start_no_imap(self, email_env):
        factory, settings_mock, _ = email_env
        settings_mock.email_imap_host = ""
        plugin, _ = factory()
        plugin.start()
        assert plugin._imap_thread is None

    def test_start_with_imap(self, email_env):
        factory, settings_mock, _ = email_env
        settings_mock.email_imap_host = "imap.example.com"
        plugin, _ = factory()

        # Stop immediately after start
        plugin._imap_stop.set()
        plugin.start()
        assert plugin._imap_thread is not None
        plugin.stop()

    def test_stop_without_start(self, email_env):
        factory, _, _ = email_env
        plugin, _ = factory()
        plugin.stop()  # Should not raise


# ---------------------------------------------------------------------------
# session_started_from_data
# ---------------------------------------------------------------------------


class TestSessionStartedFromData:
    def test_basic_session_start(self, email_env):
        factory, _, mock_smtplib = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)

        assert "sess-1" in plugin._threads
        assert "sess-1" in plugin._sessions
        mock_smtp.sendmail.assert_called_once()

    def test_sends_correct_content(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)

        call_args = mock_smtp.sendmail.call_args
        from_addr = call_args[0][0]
        to_addrs = call_args[0][1]
        msg_text = call_args[0][2]

        assert from_addr == "cross@example.com"
        assert "admin@example.com" in to_addrs
        assert "claude" in msg_text
        assert "myproj" in msg_text

    def test_default_values(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        plugin.session_started_from_data({"session_id": "s1"})

        msg_text = mock_smtp.sendmail.call_args[0][2]
        assert "agent" in msg_text
        assert "unknown" in msg_text

    def test_smtp_failure(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        mock_smtp.sendmail.side_effect = Exception("SMTP error")

        _register_session(plugin)
        # Should not raise, but thread should not be stored
        assert "sess-1" not in plugin._threads


# ---------------------------------------------------------------------------
# session_ended_from_data
# ---------------------------------------------------------------------------


class TestSessionEndedFromData:
    def test_basic_end(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)
        mock_smtp.reset_mock()

        plugin.session_ended_from_data({"session_id": "sess-1", "exit_code": 0})
        mock_smtp.sendmail.assert_called_once()
        msg_text = mock_smtp.sendmail.call_args[0][2]
        assert "exit code 0" in msg_text

    def test_with_duration(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)
        mock_smtp.reset_mock()

        now = time.time()
        plugin.session_ended_from_data(
            {
                "session_id": "sess-1",
                "exit_code": 0,
                "started_at": now - 300,
                "ended_at": now,
            }
        )
        msg_text = mock_smtp.sendmail.call_args[0][2]
        assert "5m" in msg_text

    def test_unknown_session(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        plugin.session_ended_from_data({"session_id": "unknown", "exit_code": 1})
        mock_smtp.sendmail.assert_not_called()

    def test_cleans_up_sessions(self, email_env):
        factory, _, _ = email_env
        plugin, _ = factory()
        _register_session(plugin)
        assert "sess-1" in plugin._sessions
        plugin.session_ended_from_data({"session_id": "sess-1", "exit_code": 0})
        assert "sess-1" not in plugin._sessions

    def test_in_reply_to_set(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)
        thread_msg_id = plugin._threads["sess-1"]
        mock_smtp.reset_mock()

        plugin.session_ended_from_data({"session_id": "sess-1", "exit_code": 0})
        msg_text = mock_smtp.sendmail.call_args[0][2]
        assert "In-Reply-To:" in msg_text
        assert thread_msg_id in msg_text


# ---------------------------------------------------------------------------
# handle_pty_output
# ---------------------------------------------------------------------------


class TestHandlePtyOutput:
    def test_permission_prompt_sends_email(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)
        mock_smtp.reset_mock()

        plugin.handle_pty_output("sess-1", "Do you want to allow this? 1. Yes 2. Allow all 3. No")
        mock_smtp.sendmail.assert_called_once()
        body = _get_email_body(mock_smtp)
        assert "Permission needed" in body
        assert "APPROVE" in body
        assert "sess-1" in plugin._permission_pending

    def test_non_permission_text_ignored(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)
        mock_smtp.reset_mock()

        plugin.handle_pty_output("sess-1", "Compiling project...")
        mock_smtp.sendmail.assert_not_called()

    def test_no_thread_returns(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        plugin.handle_pty_output("unknown", "Do you want to proceed?")
        mock_smtp.sendmail.assert_not_called()

    def test_debounce(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)
        mock_smtp.reset_mock()

        plugin.handle_pty_output("sess-1", "Do you want to allow this?")
        plugin.handle_pty_output("sess-1", "Do you want to allow this?")
        assert mock_smtp.sendmail.call_count == 1

    def test_tool_desc_included(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)
        plugin._last_tool_desc["sess-1"] = "`Bash`: `rm -rf /`"
        mock_smtp.reset_mock()

        plugin.handle_pty_output("sess-1", "Do you want to allow this?")
        body = _get_email_body(mock_smtp)
        assert "Bash" in body

    def test_tool_desc_cleared_after_post(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)
        plugin._last_tool_desc["sess-1"] = "`Bash`: `ls`"
        mock_smtp.reset_mock()

        plugin.handle_pty_output("sess-1", "Do you want to allow this?")
        assert "sess-1" not in plugin._last_tool_desc


# ---------------------------------------------------------------------------
# handle_event (async EventBus handler)
# ---------------------------------------------------------------------------


class TestHandleEvent:
    @pytest.mark.anyio
    async def test_tool_use_event_tracks_desc(self, email_env):
        factory, _, _ = email_env
        plugin, _ = factory()
        _register_session(plugin)

        event = ToolUseEvent(name="Bash", tool_use_id="tu1", input={"command": "ls -la"})
        await plugin.handle_event(event)

        assert "sess-1" in plugin._last_tool_desc
        assert "`Bash`" in plugin._last_tool_desc["sess-1"]

    @pytest.mark.anyio
    async def test_tool_use_file_tools_include_path(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)
        mock_smtp.reset_mock()

        event = ToolUseEvent(name="Read", tool_use_id="tu1", input={"file_path": "/foo/bar.py"})
        await plugin.handle_event(event)

        assert "`Read`" in plugin._last_tool_desc["sess-1"]
        assert "/foo/bar.py" in plugin._last_tool_desc["sess-1"]
        mock_smtp.sendmail.assert_not_called()

    @pytest.mark.anyio
    async def test_text_event_resolves_permission(self, email_env):
        factory, _, _ = email_env
        plugin, _ = factory()
        _register_session(plugin)
        plugin._permission_pending["sess-1"] = "<msg-id@example.com>"

        event = TextEvent(text="I'll proceed with the edit")
        await plugin.handle_event(event)

        assert "sess-1" not in plugin._permission_pending

    @pytest.mark.anyio
    async def test_gate_decision_block(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)
        mock_smtp.reset_mock()

        event = GateDecisionEvent(
            tool_use_id="tu1",
            tool_name="Bash",
            action="block",
            reason="Dangerous command",
            tool_input={"command": "rm -rf /"},
        )
        await plugin.handle_event(event)

        mock_smtp.sendmail.assert_called_once()
        msg_text = mock_smtp.sendmail.call_args[0][2]
        assert "BLOCK" in msg_text
        assert "Bash" in msg_text

    @pytest.mark.anyio
    async def test_gate_decision_escalate_with_buttons(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)
        mock_smtp.reset_mock()

        event = GateDecisionEvent(
            tool_use_id="tu1",
            tool_name="Bash",
            action="escalate",
            reason="Needs review",
        )
        await plugin.handle_event(event)

        body = _get_email_body(mock_smtp)
        assert "ESCALATE" in body
        assert "APPROVE" in body
        assert "DENY" in body
        assert "tu1" in plugin._gate_pending

    @pytest.mark.anyio
    async def test_gate_escalation_resolved_allow(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)
        mock_smtp.reset_mock()

        # Post escalation
        event = GateDecisionEvent(tool_use_id="tu1", tool_name="Bash", action="escalate", reason="Review")
        await plugin.handle_event(event)
        assert "tu1" in plugin._gate_pending
        mock_smtp.reset_mock()

        # Resolve
        allow_event = GateDecisionEvent(
            tool_use_id="tu1",
            tool_name="Bash",
            action="allow",
            reason="Approved by human reviewer (@alice)",
        )
        await plugin.handle_event(allow_event)

        mock_smtp.sendmail.assert_called_once()
        body = _get_email_body(mock_smtp)
        assert "Approved" in body
        assert "@alice" in body
        assert "tu1" not in plugin._gate_pending

    @pytest.mark.anyio
    async def test_gate_decision_alert(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)
        mock_smtp.reset_mock()

        event = GateDecisionEvent(tool_use_id="tu1", tool_name="Write", action="alert", reason="Suspicious write")
        await plugin.handle_event(event)

        msg_text = mock_smtp.sendmail.call_args[0][2]
        assert "ALERT" in msg_text

    @pytest.mark.anyio
    async def test_gate_decision_allow_ignored_without_pending(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)
        mock_smtp.reset_mock()

        event = GateDecisionEvent(tool_use_id="tu1", tool_name="Read", action="allow")
        await plugin.handle_event(event)
        mock_smtp.sendmail.assert_not_called()

    @pytest.mark.anyio
    async def test_sentinel_review_alert(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)
        mock_smtp.reset_mock()

        event = SentinelReviewEvent(action="alert", summary="All looks good", concerns="None", event_count=5)
        await plugin.handle_event(event)

        body = _get_email_body(mock_smtp)
        assert "ALERT" in body
        assert "5 events reviewed" in body
        assert "Concerns" not in body  # "None" excluded

    @pytest.mark.anyio
    async def test_sentinel_review_escalate(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)
        mock_smtp.reset_mock()

        event = SentinelReviewEvent(
            action="escalate",
            summary="Suspicious activity",
            concerns="Agent reading credentials",
            event_count=10,
        )
        await plugin.handle_event(event)

        body = _get_email_body(mock_smtp)
        assert "ESCALATE" in body
        assert "Concerns" in body

    @pytest.mark.anyio
    async def test_sentinel_review_allow_ignored(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)
        mock_smtp.reset_mock()

        event = SentinelReviewEvent(action="allow", summary="ok", event_count=3)
        await plugin.handle_event(event)
        mock_smtp.sendmail.assert_not_called()

    @pytest.mark.anyio
    async def test_permission_resolved_from_other_surface(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)
        plugin._permission_pending["sess-1"] = "<perm-msg@example.com>"
        mock_smtp.reset_mock()

        event = PermissionResolvedEvent(session_id="sess-1", action="approve", resolver="slack (@alice)")
        await plugin.handle_event(event)

        mock_smtp.sendmail.assert_called_once()
        body = _get_email_body(mock_smtp)
        assert "Approved" in body
        assert "sess-1" not in plugin._permission_pending

    @pytest.mark.anyio
    async def test_permission_resolved_from_email_ignored(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)
        plugin._permission_pending["sess-1"] = "<perm-msg@example.com>"
        mock_smtp.reset_mock()

        event = PermissionResolvedEvent(session_id="sess-1", action="approve", resolver="email (admin@example.com)")
        await plugin.handle_event(event)

        # Should NOT send email (resolver starts with "email")
        mock_smtp.sendmail.assert_not_called()

    @pytest.mark.anyio
    async def test_error_event(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)
        mock_smtp.reset_mock()

        event = ErrorEvent(status_code=429, body="Rate limited")
        await plugin.handle_event(event)

        msg_text = mock_smtp.sendmail.call_args[0][2]
        assert "429" in msg_text

    @pytest.mark.anyio
    async def test_unhandled_event_type(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)
        mock_smtp.reset_mock()

        await plugin.handle_event(RequestEvent(method="POST", path="/v1/messages"))
        await plugin.handle_event(MessageStartEvent(message_id="m1", model="claude"))
        await plugin.handle_event(MessageDeltaEvent(stop_reason="end_turn"))

        mock_smtp.sendmail.assert_not_called()

    @pytest.mark.anyio
    async def test_most_recent_session_used(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin, session_id="sess-1")
        _register_session(plugin, session_id="sess-2", project="proj2")
        mock_smtp.reset_mock()

        event = ErrorEvent(status_code=500, body="fail")
        await plugin.handle_event(event)

        # Should use sess-2's thread (most recent)
        # Email was sent — the thread is maintained
        mock_smtp.sendmail.assert_called_once()


# ---------------------------------------------------------------------------
# _send_email
# ---------------------------------------------------------------------------


class TestSendEmail:
    def test_basic_send(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        msg_id = plugin._send_email("Test Subject", "Test Body")

        assert msg_id is not None
        assert "@" in msg_id
        mock_smtp.sendmail.assert_called_once()
        mock_smtp.starttls.assert_called_once()
        mock_smtp.login.assert_called_once()
        mock_smtp.quit.assert_called_once()

    def test_ssl_mode(self, email_env):
        factory, settings_mock, mock_smtplib = email_env
        settings_mock.email_smtp_ssl = True
        plugin, _ = factory()

        plugin._send_email("Subject", "Body")
        mock_smtplib.SMTP_SSL.assert_called()

    def test_no_auth(self, email_env):
        factory, settings_mock, _ = email_env
        settings_mock.email_smtp_username = ""
        settings_mock.email_smtp_password = ""
        plugin, mock_smtp = factory()

        plugin._send_email("Subject", "Body")
        mock_smtp.login.assert_not_called()

    def test_in_reply_to_headers(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()

        plugin._send_email("Re: Test", "Reply body", in_reply_to="<original@example.com>")

        msg_text = mock_smtp.sendmail.call_args[0][2]
        assert "In-Reply-To: <original@example.com>" in msg_text
        assert "References: <original@example.com>" in msg_text

    def test_smtp_failure_returns_none(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        mock_smtp.sendmail.side_effect = Exception("Connection refused")

        msg_id = plugin._send_email("Subject", "Body")
        assert msg_id is None

    def test_html_alternative(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()

        plugin._send_email("Subject", "Body with <script>")

        msg_text = mock_smtp.sendmail.call_args[0][2]
        assert "text/html" in msg_text
        assert "&lt;script&gt;" in msg_text  # HTML-escaped


# ---------------------------------------------------------------------------
# IMAP reply processing
# ---------------------------------------------------------------------------


class TestImapProcessing:
    def test_find_session_for_reply(self, email_env):
        factory, _, _ = email_env
        plugin, _ = factory()
        plugin._threads["sess-1"] = "<msg-1@example.com>"

        assert plugin._find_session_for_reply("<msg-1@example.com>") == "sess-1"
        assert plugin._find_session_for_reply("<unknown@example.com>") is None

    def test_find_session_via_permission_pending(self, email_env):
        factory, _, _ = email_env
        plugin, _ = factory()
        plugin._permission_pending["sess-2"] = "<perm-msg@example.com>"

        assert plugin._find_session_for_reply("<perm-msg@example.com>") == "sess-2"

    def test_find_gate_for_reply(self, email_env):
        factory, _, _ = email_env
        plugin, _ = factory()
        plugin._gate_pending["tu1"] = "<gate-msg@example.com>"

        assert plugin._find_gate_for_reply("<gate-msg@example.com>") == "tu1"
        assert plugin._find_gate_for_reply("<unknown@example.com>") is None


# ---------------------------------------------------------------------------
# _inject
# ---------------------------------------------------------------------------


class TestInject:
    def test_inject_with_loop(self, email_env):
        factory, _, _ = email_env
        inject_cb = AsyncMock()
        loop = asyncio.new_event_loop()
        plugin, _ = factory(inject_callback=inject_cb, event_loop=loop)

        plugin._inject("sess-1", "hello")
        loop.run_until_complete(asyncio.sleep(0.05))

        inject_cb.assert_called_once_with("sess-1", "hello")
        loop.close()

    def test_inject_no_callback(self, email_env):
        factory, _, _ = email_env
        plugin, _ = factory(inject_callback=None)
        plugin._inject("sess-1", "hello")  # Should not raise

    def test_inject_no_event_loop(self, email_env):
        factory, _, _ = email_env
        inject_cb = AsyncMock()
        plugin, _ = factory(inject_callback=inject_cb, event_loop=None)
        plugin._inject("sess-1", "hello")
        inject_cb.assert_not_called()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_session_registration(self, email_env):
        factory, _, _ = email_env
        plugin, _ = factory()

        def register(i):
            plugin.session_started_from_data(
                {"session_id": f"sess-{i}", "project": "proj", "agent": "claude", "cwd": "/tmp"}
            )

        threads = [threading.Thread(target=register, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(plugin._sessions) == 10


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_session_end_missing_exit_code(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)
        mock_smtp.reset_mock()

        plugin.session_ended_from_data({"session_id": "sess-1"})
        msg_text = mock_smtp.sendmail.call_args[0][2]
        assert "exit code ?" in msg_text

    @pytest.mark.anyio
    async def test_no_threads_sends_to_default(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        event = ErrorEvent(status_code=500, body="error")
        await plugin.handle_event(event)
        # No threads = no in_reply_to, but still sends
        mock_smtp.sendmail.assert_called_once()

    def test_debounce_resets_after_interval(self, email_env):
        factory, _, _ = email_env
        plugin, mock_smtp = factory()
        _register_session(plugin)
        mock_smtp.reset_mock()

        plugin.handle_pty_output("sess-1", "Do you want to allow this?")
        assert mock_smtp.sendmail.call_count == 1

        # Simulate time passing past the debounce interval
        plugin._last_permission_post["sess-1"] -= 10.0
        mock_smtp.reset_mock()

        plugin.handle_pty_output("sess-1", "Do you want to allow this?")
        assert mock_smtp.sendmail.call_count == 1
