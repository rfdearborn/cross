"""Tests for the Slack relay plugin — channel management, event handling, interactive buttons."""

from __future__ import annotations

import asyncio
import threading
import time
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from cross.events import (
    ErrorEvent,
    GateDecisionEvent,
    MessageDeltaEvent,
    MessageStartEvent,
    RequestEvent,
    SentinelReviewEvent,
    TextEvent,
    ToolUseEvent,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Module-level patches that stay active for all tests.
# The `settings` and `WebClient` are patched at the module level so that
# method calls on the plugin (which access `settings` at call time) work.

_SETTINGS_DEFAULTS = {
    "slack_bot_token": "xoxb-fake-token",
    "slack_app_token": "xapp-fake-token",
    "slack_channel_base": "cross",
    "slack_channel_append_project": False,
    "slack_channel_append_user": True,
}


def _mock_settings(**overrides):
    """Create a mock settings object with sensible defaults for Slack tests."""
    vals = dict(_SETTINGS_DEFAULTS, **overrides)
    mock = MagicMock()
    for k, v in vals.items():
        setattr(mock, k, v)
    return mock


@pytest.fixture()
def slack_env():
    """Yield a factory that creates SlackPlugin instances with mocked SDK.

    The patches remain active for the duration of the test so that methods
    like `_channel_name` and `_ensure_channel` see the mocked settings.
    """
    settings_mock = _mock_settings()

    with (
        patch("cross.plugins.slack.settings", settings_mock),
        patch("cross.plugins.slack.WebClient") as MockWebClient,
    ):
        mock_web = MagicMock()
        MockWebClient.return_value = mock_web

        from cross.plugins.slack import SlackPlugin

        def factory(inject_callback=None, spawn_callback=None, event_loop=None, **kw):
            for k, v in kw.items():
                setattr(settings_mock, k, v)
            p = SlackPlugin(
                inject_callback=inject_callback,
                spawn_callback=spawn_callback,
                event_loop=event_loop,
            )
            return p, mock_web

        yield factory, settings_mock, MockWebClient


def _register_session(plugin, mock_web, session_id="sess-1", project="myproj", agent="claude", cwd="/tmp/myproj"):
    """Helper to register a session (posts to Slack, creates thread)."""
    mock_web.chat_postMessage.return_value = {"ts": "1234567890.000001"}
    mock_web.conversations_list.return_value = {"channels": []}
    mock_web.conversations_create.return_value = {"channel": {"id": "C_CHAN"}}
    mock_web.conversations_members.return_value = {"members": []}
    mock_web.users_list.return_value = {"members": []}

    plugin.session_started_from_data(
        {
            "session_id": session_id,
            "project": project,
            "agent": agent,
            "cwd": cwd,
        }
    )
    return "C_CHAN", "1234567890.000001"


# ---------------------------------------------------------------------------
# Module-level helpers (_slugify, _is_permission_prompt, _extract_allow_all)
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_basic(self, slack_env):
        from cross.plugins.slack import _slugify

        assert _slugify("MyProject") == "myproject"

    def test_special_chars(self, slack_env):
        from cross.plugins.slack import _slugify

        assert _slugify("my_proj/foo") == "my-proj-foo"

    def test_multiple_dashes(self, slack_env):
        from cross.plugins.slack import _slugify

        assert _slugify("a---b") == "a-b"

    def test_leading_trailing_dashes(self, slack_env):
        from cross.plugins.slack import _slugify

        assert _slugify("--hello--") == "hello"

    def test_empty(self, slack_env):
        from cross.plugins.slack import _slugify

        assert _slugify("") == ""


class TestIsPermissionPrompt:
    def test_do_you_want_to(self, slack_env):
        from cross.plugins.slack import _is_permission_prompt

        assert _is_permission_prompt("Do you want to allow this?") is True

    def test_garbled_no_spaces(self, slack_env):
        from cross.plugins.slack import _is_permission_prompt

        assert _is_permission_prompt("Doyouwant to proceed") is True

    def test_partial_match(self, slack_env):
        from cross.plugins.slack import _is_permission_prompt

        assert _is_permission_prompt("Do you want t") is True

    def test_numbered_options(self, slack_env):
        from cross.plugins.slack import _is_permission_prompt

        assert _is_permission_prompt("1. Yes, allow\n2. Allow all\n3. No, deny") is True

    def test_not_a_prompt(self, slack_env):
        from cross.plugins.slack import _is_permission_prompt

        assert _is_permission_prompt("Compiling project...") is False

    def test_partial_numbered_only_yes(self, slack_env):
        from cross.plugins.slack import _is_permission_prompt

        # Only "1. Yes" without "3. No" should not match the numbered pattern
        assert _is_permission_prompt("1. Yes but no 3") is False


class TestExtractAllowAll:
    def test_clean_text(self, slack_env):
        from cross.plugins.slack import _extract_allow_all

        result = _extract_allow_all("2. allow all edits in Downloads/")
        assert result == "Allow all edits in Downloads/"

    def test_garbled_text(self, slack_env):
        from cross.plugins.slack import _extract_allow_all

        result = _extract_allow_all("allowalleditsinDownloads/")
        assert result == "Allow all edits in Downloads/"

    def test_bash_commands(self, slack_env):
        from cross.plugins.slack import _extract_allow_all

        result = _extract_allow_all("allow all bash")
        assert result == "Allow all Bash"

    def test_bash_spaced(self, slack_env):
        from cross.plugins.slack import _extract_allow_all

        result = _extract_allow_all("Allow all Bash commands")
        assert result == "Allow all Bash"

    def test_no_match(self, slack_env):
        from cross.plugins.slack import _extract_allow_all

        result = _extract_allow_all("some random text")
        assert result is None


# ---------------------------------------------------------------------------
# SlackPlugin.__init__
# ---------------------------------------------------------------------------


class TestSlackPluginInit:
    def test_init_creates_plugin(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        # Verify it's usable — can register a session without error
        assert plugin._socket is None
        assert plugin._bot_user_id is None


# ---------------------------------------------------------------------------
# start / stop
# ---------------------------------------------------------------------------


class TestStartStop:
    def test_start_resolves_bot_id(self, slack_env):
        factory, _, MockWebClient = slack_env
        plugin, mock_web = factory()
        mock_web.auth_test.return_value = {"user_id": "U_BOT", "user": "crossbot"}
        mock_web.users_list.return_value = {"members": []}

        with patch("cross.plugins.slack.SocketModeClient") as MockSocket:
            mock_socket = MagicMock()
            mock_socket.socket_mode_request_listeners = []
            MockSocket.return_value = mock_socket
            plugin.start()

        assert plugin._bot_user_id == "U_BOT"
        assert plugin._username is None
        mock_socket.connect.assert_called_once()
        assert plugin._on_socket_event in mock_socket.socket_mode_request_listeners

    def test_start_resolves_username(self, slack_env):
        factory, _, MockWebClient = slack_env
        plugin, mock_web = factory()
        mock_web.auth_test.return_value = {"user_id": "U_BOT", "user": "crossbot"}
        mock_web.users_list.return_value = {
            "members": [
                {"id": "USLACKBOT", "is_bot": False, "deleted": False, "name": "slackbot"},
                {"id": "U_BOT", "is_bot": True, "deleted": False, "name": "crossbot"},
                {"id": "U_HUMAN", "is_bot": False, "deleted": False, "name": "rob"},
            ]
        }

        with patch("cross.plugins.slack.SocketModeClient") as MockSocket:
            mock_socket = MagicMock()
            mock_socket.socket_mode_request_listeners = []
            MockSocket.return_value = mock_socket
            plugin.start()

        assert plugin._username == "rob"
        assert plugin._user_ids == ["U_HUMAN"]

    def test_stop_disconnects(self, slack_env):
        factory, _, _ = slack_env
        plugin, _ = factory()
        mock_socket = MagicMock()
        plugin._socket = mock_socket

        plugin.stop()
        mock_socket.disconnect.assert_called_once()

    def test_stop_no_socket(self, slack_env):
        factory, _, _ = slack_env
        plugin, _ = factory()
        plugin._socket = None
        plugin.stop()
        assert plugin._socket is None  # no socket created

    def test_stop_disconnect_error_swallowed(self, slack_env):
        factory, _, _ = slack_env
        plugin, _ = factory()
        mock_socket = MagicMock()
        mock_socket.disconnect.side_effect = RuntimeError("gone")
        plugin._socket = mock_socket
        plugin.stop()
        mock_socket.disconnect.assert_called_once()  # attempted despite error


# ---------------------------------------------------------------------------
# _channel_name
# ---------------------------------------------------------------------------


class TestChannelName:
    def test_default(self, slack_env):
        factory, _, _ = slack_env
        plugin, _ = factory()
        assert plugin._channel_name("myproj") == "cross"

    def test_with_project(self, slack_env):
        factory, settings_mock, _ = slack_env
        settings_mock.slack_channel_append_project = True
        plugin, _ = factory()
        assert plugin._channel_name("myproj") == "cross-myproj"

    def test_custom_base(self, slack_env):
        factory, settings_mock, _ = slack_env
        settings_mock.slack_channel_base = "acme-agents"
        settings_mock.slack_channel_append_project = True
        plugin, _ = factory()
        assert plugin._channel_name("myproj") == "acme-agents-myproj"

    def test_with_username(self, slack_env):
        factory, settings_mock, _ = slack_env
        settings_mock.slack_channel_append_project = True
        plugin, _ = factory()
        plugin._username = "rob"
        assert plugin._channel_name("myproj") == "cross-myproj-rob"

    def test_default_with_username(self, slack_env):
        factory, _, _ = slack_env
        plugin, _ = factory()
        plugin._username = "rob"
        assert plugin._channel_name("myproj") == "cross-rob"

    def test_no_user_config(self, slack_env):
        factory, settings_mock, _ = slack_env
        settings_mock.slack_channel_append_project = True
        settings_mock.slack_channel_append_user = False
        plugin, _ = factory()
        plugin._username = "rob"
        assert plugin._channel_name("myproj") == "cross-myproj"

    def test_truncation(self, slack_env):
        factory, settings_mock, _ = slack_env
        settings_mock.slack_channel_append_project = True
        plugin, _ = factory()
        long_name = "a" * 100
        name = plugin._channel_name(long_name)
        assert len(name) <= 80


# ---------------------------------------------------------------------------
# _ensure_channel
# ---------------------------------------------------------------------------


class TestEnsureChannel:
    def test_cached(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        plugin._channels["cross"] = "C_CACHED"
        result = plugin._ensure_channel("myproj")
        assert result == "C_CACHED"
        mock_web.conversations_list.assert_not_called()

    def test_found_existing(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        mock_web.conversations_list.side_effect = [
            {"channels": [{"name": "cross", "id": "C_EXISTING"}]},
        ]
        result = plugin._ensure_channel("myproj")
        assert result == "C_EXISTING"
        assert plugin._channels["cross"] == "C_EXISTING"

    def test_creates_private(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        mock_web.conversations_list.return_value = {"channels": []}
        mock_web.conversations_create.return_value = {"channel": {"id": "C_NEW"}}

        result = plugin._ensure_channel("myproj")
        assert result == "C_NEW"
        mock_web.conversations_create.assert_called_once_with(name="cross", is_private=True)

    def test_fallback_to_public(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        mock_web.conversations_list.return_value = {"channels": []}
        mock_web.conversations_create.side_effect = [
            Exception("not_allowed"),
            {"channel": {"id": "C_PUBLIC"}},
        ]

        result = plugin._ensure_channel("myproj")
        assert result == "C_PUBLIC"
        assert mock_web.conversations_create.call_count == 2
        # Second call should NOT have is_private
        second_call = mock_web.conversations_create.call_args_list[1]
        assert "is_private" not in second_call.kwargs


# ---------------------------------------------------------------------------
# _ensure_users_invited
# ---------------------------------------------------------------------------


class TestEnsureUsersInvited:
    def test_invites_missing_users(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        plugin._user_ids = ["U_HUMAN"]
        mock_web.conversations_members.return_value = {"members": ["U_BOT"]}

        plugin._ensure_users_invited("C_CHAN")
        mock_web.conversations_invite.assert_called_once_with(channel="C_CHAN", users="U_HUMAN")

    def test_skips_already_members(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        plugin._user_ids = ["U_HUMAN"]
        mock_web.conversations_members.return_value = {"members": ["U_BOT", "U_HUMAN"]}

        plugin._ensure_users_invited("C_CHAN")
        mock_web.conversations_invite.assert_not_called()

    def test_no_cached_users(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        plugin._user_ids = []

        plugin._ensure_users_invited("C_CHAN")
        mock_web.conversations_members.assert_not_called()
        mock_web.conversations_invite.assert_not_called()

    def test_error_handled(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        plugin._user_ids = ["U_HUMAN"]
        mock_web.conversations_members.side_effect = Exception("api error")
        # Should not raise
        plugin._ensure_users_invited("C_CHAN")


# ---------------------------------------------------------------------------
# _find_channel
# ---------------------------------------------------------------------------


class TestFindChannel:
    def test_finds_private(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        mock_web.conversations_list.side_effect = [
            {"channels": [{"name": "test-cross-myproj", "id": "C_PRIV"}]},
        ]
        result = plugin._find_channel("test-cross-myproj")
        assert result == "C_PRIV"

    def test_finds_public(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        mock_web.conversations_list.side_effect = [
            {"channels": []},  # private
            {"channels": [{"name": "test-cross-myproj", "id": "C_PUB"}]},  # public
        ]
        result = plugin._find_channel("test-cross-myproj")
        assert result == "C_PUB"

    def test_not_found(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        mock_web.conversations_list.return_value = {"channels": []}
        result = plugin._find_channel("test-cross-nope")
        assert result is None

    def test_api_error(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        mock_web.conversations_list.side_effect = Exception("network error")
        result = plugin._find_channel("test-cross-myproj")
        assert result is None


# ---------------------------------------------------------------------------
# _project_for_channel
# ---------------------------------------------------------------------------


class TestProjectForChannel:
    def test_cached_lookup(self, slack_env):
        factory, _, _ = slack_env
        plugin, _ = factory()
        plugin._channels["cross-myproj"] = "C_CHAN"
        result = plugin._project_for_channel("C_CHAN")
        assert result == "myproj"

    def test_api_lookup(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        mock_web.conversations_info.return_value = {"channel": {"name": "cross-myproj"}}
        result = plugin._project_for_channel("C_CHAN")
        assert result == "myproj"
        # Should be cached now
        assert plugin._channels["cross-myproj"] == "C_CHAN"

    def test_api_error(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        mock_web.conversations_info.side_effect = Exception("not found")
        result = plugin._project_for_channel("C_UNKNOWN")
        assert result is None

    def test_no_base_in_name(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        mock_web.conversations_info.return_value = {"channel": {"name": "random-channel"}}
        result = plugin._project_for_channel("C_RANDOM")
        assert result is None

    def test_custom_base(self, slack_env):
        factory, settings_mock, _ = slack_env
        settings_mock.slack_channel_base = "acme-agents"
        plugin, _ = factory()
        plugin._channels["acme-agents-myproj"] = "C_CHAN"
        result = plugin._project_for_channel("C_CHAN")
        assert result == "myproj"

    def test_multi_part_project(self, slack_env):
        factory, _, _ = slack_env
        plugin, _ = factory()
        plugin._channels["cross-my-cool-proj"] = "C_CHAN"
        result = plugin._project_for_channel("C_CHAN")
        assert result == "my-cool-proj"

    def test_strips_username_suffix(self, slack_env):
        factory, _, _ = slack_env
        plugin, _ = factory()
        plugin._username = "rob"
        plugin._channels["cross-myproj-rob"] = "C_CHAN"
        result = plugin._project_for_channel("C_CHAN")
        assert result == "myproj"

    def test_strips_username_multi_part_project(self, slack_env):
        factory, _, _ = slack_env
        plugin, _ = factory()
        plugin._username = "rob"
        plugin._channels["cross-my-cool-proj-rob"] = "C_CHAN"
        result = plugin._project_for_channel("C_CHAN")
        assert result == "my-cool-proj"

    def test_base_only_no_project(self, slack_env):
        factory, _, _ = slack_env
        plugin, _ = factory()
        plugin._channels["cross"] = "C_CHAN"
        result = plugin._project_for_channel("C_CHAN")
        assert result is None


# ---------------------------------------------------------------------------
# session_started_from_data
# ---------------------------------------------------------------------------


class TestSessionStartedFromData:
    def test_basic_session_start(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)

        assert "sess-1" in plugin._threads
        ch, ts = plugin._threads["sess-1"]
        assert ch == "C_CHAN"
        assert ts == "1234567890.000001"
        assert "sess-1" in plugin._sessions

    def test_slack_initiated_session(self, slack_env):
        """When a session was initiated via Slack @mention, thread under the original message."""
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        mock_web.chat_postMessage.return_value = {"ts": "orig_msg_ts"}

        # Simulate pending thread from a Slack @mention
        plugin._pending_thread_ts["myproj"] = ("C_MENTION", "mention_ts")

        plugin.session_started_from_data(
            {
                "session_id": "sess-2",
                "project": "myproj",
                "agent": "claude",
                "cwd": "/tmp/myproj",
            }
        )

        # Should post in the mention thread, not create a new channel
        mock_web.chat_postMessage.assert_called_once_with(
            channel="C_MENTION",
            thread_ts="mention_ts",
            text="*claude* session started in `myproj` (`/tmp/myproj`)",
        )
        assert plugin._threads["sess-2"] == ("C_MENTION", "mention_ts")
        # Pending should be consumed
        assert "myproj" not in plugin._pending_thread_ts

    def test_default_values(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        mock_web.chat_postMessage.return_value = {"ts": "ts1"}
        mock_web.conversations_list.return_value = {"channels": []}
        mock_web.conversations_create.return_value = {"channel": {"id": "C1"}}
        mock_web.users_list.return_value = {"members": []}

        plugin.session_started_from_data({"session_id": "s1"})

        # Defaults: project=unknown, agent=agent, cwd=""
        msg_call = mock_web.chat_postMessage.call_args
        assert "agent" in msg_call.kwargs["text"]
        assert "unknown" in msg_call.kwargs["text"]


# ---------------------------------------------------------------------------
# session_ended_from_data
# ---------------------------------------------------------------------------


class TestSessionEndedFromData:
    def test_basic_end(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        plugin.session_ended_from_data(
            {
                "session_id": "sess-1",
                "exit_code": 0,
            }
        )

        mock_web.chat_postMessage.assert_called_once()
        msg = mock_web.chat_postMessage.call_args.kwargs["text"]
        assert "exit code 0" in msg

    def test_with_duration(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        now = time.time()
        plugin.session_ended_from_data(
            {
                "session_id": "sess-1",
                "exit_code": 0,
                "started_at": now - 300,
                "ended_at": now,
            }
        )

        msg = mock_web.chat_postMessage.call_args.kwargs["text"]
        assert "5m" in msg

    def test_short_duration(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        now = time.time()
        plugin.session_ended_from_data(
            {
                "session_id": "sess-1",
                "exit_code": 0,
                "started_at": now - 30,
                "ended_at": now,
            }
        )

        msg = mock_web.chat_postMessage.call_args.kwargs["text"]
        # Duration < 1 min should not show "after Xm"
        assert "after" not in msg

    def test_unknown_session(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        # No session registered - should not post
        plugin.session_ended_from_data({"session_id": "unknown", "exit_code": 1})
        mock_web.chat_postMessage.assert_not_called()

    def test_cleans_up_sessions(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        assert "sess-1" in plugin._sessions

        plugin.session_ended_from_data({"session_id": "sess-1", "exit_code": 0})
        assert "sess-1" not in plugin._sessions


# ---------------------------------------------------------------------------
# handle_pty_output
# ---------------------------------------------------------------------------


class TestHandlePtyOutput:
    def test_permission_prompt_posts(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()
        mock_web.chat_postMessage.return_value = {"ts": "perm_ts"}

        plugin.handle_pty_output("sess-1", "Do you want to allow this? 1. Yes 2. Allow all 3. No")

        mock_web.chat_postMessage.assert_called_once()
        call_kwargs = mock_web.chat_postMessage.call_args.kwargs
        assert call_kwargs["reply_broadcast"] is True
        assert "blocks" in call_kwargs
        assert "sess-1" in plugin._permission_pending

    def test_non_permission_text_ignored(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        plugin.handle_pty_output("sess-1", "Compiling project...")
        mock_web.chat_postMessage.assert_not_called()

    def test_no_thread_returns(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        plugin.handle_pty_output("unknown", "Do you want to proceed?")
        mock_web.chat_postMessage.assert_not_called()

    def test_debounce(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()
        mock_web.chat_postMessage.return_value = {"ts": "perm_ts"}

        plugin.handle_pty_output("sess-1", "Do you want to allow this?")
        plugin.handle_pty_output("sess-1", "Do you want to allow this?")

        # Second call should be debounced
        assert mock_web.chat_postMessage.call_count == 1

    def test_tool_desc_included(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        plugin._last_tool_desc["sess-1"] = "`Bash`: `rm -rf /`"
        mock_web.reset_mock()
        mock_web.chat_postMessage.return_value = {"ts": "perm_ts"}

        plugin.handle_pty_output("sess-1", "Do you want to allow this?")

        call_kwargs = mock_web.chat_postMessage.call_args.kwargs
        assert "`Bash`: `rm -rf /`" in call_kwargs["text"]

    def test_tool_desc_cleared_after_post(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        plugin._last_tool_desc["sess-1"] = "`Bash`: `ls`"
        mock_web.reset_mock()
        mock_web.chat_postMessage.return_value = {"ts": "perm_ts"}

        plugin.handle_pty_output("sess-1", "Do you want to allow this?")

        assert "sess-1" not in plugin._last_tool_desc

    def test_allow_all_label_extracted(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()
        mock_web.chat_postMessage.return_value = {"ts": "perm_ts"}

        plugin.handle_pty_output("sess-1", "Do you want to allow all edits in Downloads/?")

        call_kwargs = mock_web.chat_postMessage.call_args.kwargs
        blocks = call_kwargs["blocks"]
        # Find the actions block
        actions = [b for b in blocks if b["type"] == "actions"][0]
        allow_all_btn = [e for e in actions["elements"] if e["action_id"] == "permission_allow_all"][0]
        assert allow_all_btn["text"]["text"] == "Allow all edits in Downloads/"


# ---------------------------------------------------------------------------
# handle_event (async EventBus handler)
# ---------------------------------------------------------------------------


class TestHandleEvent:
    @pytest.mark.anyio
    async def test_no_threads_posts_to_fallback_channel(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        event = ErrorEvent(status_code=500, body="error")
        await plugin.handle_event(event)
        mock_web.chat_postMessage.assert_called_once()
        kwargs = mock_web.chat_postMessage.call_args.kwargs
        assert kwargs["thread_ts"] is None  # top-level message, no thread

    @pytest.mark.anyio
    async def test_tool_use_event_tracks_desc(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)

        event = ToolUseEvent(name="Bash", tool_use_id="tu1", input={"command": "ls -la"})
        await plugin.handle_event(event)

        # Should track tool description, but NOT post to Slack
        assert "sess-1" in plugin._last_tool_desc
        assert "`Bash`" in plugin._last_tool_desc["sess-1"]
        assert "ls -la" in plugin._last_tool_desc["sess-1"]

    @pytest.mark.anyio
    async def test_tool_use_file_tools_include_path(self, slack_env):
        """Read/Write/Edit tools should include file_path in description."""
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)

        mock_web.chat_postMessage.reset_mock()

        event = ToolUseEvent(name="Read", tool_use_id="tu1", input={"file_path": "/foo/bar.py"})
        await plugin.handle_event(event)

        assert "`Read`" in plugin._last_tool_desc["sess-1"]
        assert "/foo/bar.py" in plugin._last_tool_desc["sess-1"]
        # File tool events are tracked internally but NOT posted to Slack
        mock_web.chat_postMessage.assert_not_called()

    @pytest.mark.anyio
    async def test_tool_use_generic(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)

        event = ToolUseEvent(name="WebSearch", tool_use_id="tu1", input={"query": "test"})
        await plugin.handle_event(event)

        assert plugin._last_tool_desc["sess-1"] == "`WebSearch`"

    @pytest.mark.anyio
    async def test_tool_use_bash_truncates(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)

        long_cmd = "x" * 200
        event = ToolUseEvent(name="Bash", tool_use_id="tu1", input={"command": long_cmd})
        await plugin.handle_event(event)

        # Command should be truncated to 80 chars
        desc = plugin._last_tool_desc["sess-1"]
        assert len(long_cmd) > 80
        assert long_cmd[:80] in desc

    @pytest.mark.anyio
    async def test_text_event_resolves_permission(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)

        # Set up pending permission
        plugin._permission_pending["sess-1"] = ("C_CHAN", "perm_ts")

        event = TextEvent(text="I'll proceed with the edit")
        await plugin.handle_event(event)

        # Permission should be resolved
        assert "sess-1" not in plugin._permission_pending
        mock_web.chat_update.assert_called_once()
        update_kwargs = mock_web.chat_update.call_args.kwargs
        assert update_kwargs["ts"] == "perm_ts"
        assert "Approved" in update_kwargs["text"]

    @pytest.mark.anyio
    async def test_text_event_no_pending(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        event = TextEvent(text="Hello world")
        await plugin.handle_event(event)

        # No pending permission -> no update
        mock_web.chat_update.assert_not_called()

    @pytest.mark.anyio
    async def test_text_event_update_fails(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        plugin._permission_pending["sess-1"] = ("C_CHAN", "perm_ts")
        mock_web.chat_update.side_effect = Exception("api error")

        event = TextEvent(text="ok")
        # Should not raise
        await plugin.handle_event(event)
        assert "sess-1" not in plugin._permission_pending

    @pytest.mark.anyio
    async def test_gate_decision_block(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        event = GateDecisionEvent(
            tool_use_id="tu1",
            tool_name="Bash",
            action="block",
            reason="Dangerous command",
            tool_input={"command": "rm -rf /"},
        )
        await plugin.handle_event(event)

        mock_web.chat_postMessage.assert_called_once()
        kwargs = mock_web.chat_postMessage.call_args.kwargs
        assert "BLOCK" in kwargs["text"]
        assert "`Bash`" in kwargs["text"]
        assert "Dangerous command" in kwargs["text"]
        assert kwargs["reply_broadcast"] is True

    @pytest.mark.anyio
    async def test_gate_decision_escalate(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        event = GateDecisionEvent(
            tool_use_id="tu1",
            tool_name="Bash",
            action="escalate",
            reason="Needs review",
        )
        await plugin.handle_event(event)

        kwargs = mock_web.chat_postMessage.call_args.kwargs
        assert "ESCALATE" in kwargs["text"]
        assert kwargs["reply_broadcast"] is True

    @pytest.mark.anyio
    async def test_gate_decision_alert(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        event = GateDecisionEvent(
            tool_use_id="tu1",
            tool_name="Write",
            action="alert",
            reason="Suspicious write",
        )
        await plugin.handle_event(event)

        kwargs = mock_web.chat_postMessage.call_args.kwargs
        assert "ALERT" in kwargs["text"]
        # alert should NOT reply_broadcast
        assert kwargs["reply_broadcast"] is False

    @pytest.mark.anyio
    async def test_gate_decision_allow_ignored_without_pending(self, slack_env):
        """Allow events with no pending escalation should not post or update."""
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        event = GateDecisionEvent(
            tool_use_id="tu1",
            tool_name="Read",
            action="allow",
        )
        await plugin.handle_event(event)
        mock_web.chat_postMessage.assert_not_called()
        mock_web.chat_update.assert_not_called()

    @pytest.mark.anyio
    async def test_gate_escalation_resolved_allow_updates_slack(self, slack_env):
        """When an escalation is approved externally (dashboard), Slack message should update."""
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        # Post escalation message (stores pending gate)
        mock_web.chat_postMessage.return_value = {"ok": True, "ts": "9999.0001"}
        escalate_event = GateDecisionEvent(
            tool_use_id="tu1",
            tool_name="Bash",
            action="escalate",
            reason="Needs review",
        )
        await plugin.handle_event(escalate_event)
        assert "tu1" in plugin._gate_pending

        mock_web.reset_mock()

        # Resolve from dashboard
        allow_event = GateDecisionEvent(
            tool_use_id="tu1",
            tool_name="Bash",
            action="allow",
            reason="Approved by human reviewer (@alice)",
            evaluator="human",
        )
        await plugin.handle_event(allow_event)

        mock_web.chat_update.assert_called_once()
        kwargs = mock_web.chat_update.call_args.kwargs
        assert kwargs["ts"] == "9999.0001"
        assert "Approved" in kwargs["text"]
        assert "@alice" in kwargs["text"]
        assert "tu1" not in plugin._gate_pending

    @pytest.mark.anyio
    async def test_gate_escalation_resolved_deny_updates_slack(self, slack_env):
        """When an escalation is denied externally, Slack message should update."""
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        mock_web.chat_postMessage.return_value = {"ok": True, "ts": "9999.0002"}
        escalate_event = GateDecisionEvent(
            tool_use_id="tu2",
            tool_name="Write",
            action="escalate",
            reason="Dangerous write",
        )
        await plugin.handle_event(escalate_event)
        mock_web.reset_mock()

        deny_event = GateDecisionEvent(
            tool_use_id="tu2",
            tool_name="Write",
            action="halt_session",
            reason="Denied by human reviewer (@bob)",
            evaluator="human",
        )
        await plugin.handle_event(deny_event)

        mock_web.chat_update.assert_called_once()
        kwargs = mock_web.chat_update.call_args.kwargs
        assert "Denied" in kwargs["text"]
        assert "@bob" in kwargs["text"]
        assert "tu2" not in plugin._gate_pending

    @pytest.mark.anyio
    async def test_gate_slack_button_clears_pending(self, slack_env):
        """When Slack's own button resolves an escalation, _gate_pending is cleaned up."""
        factory, _, _ = slack_env
        loop = asyncio.get_event_loop()
        plugin, mock_web = factory(event_loop=loop)
        plugin._resolve_approval_callback = MagicMock()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        # Simulate an escalation message being posted
        plugin._gate_pending["tu3"] = ("C_CHAN", "8888.0001")

        # Simulate Slack button click
        payload = {
            "type": "block_actions",
            "actions": [{"action_id": "gate_approve", "value": "tu3"}],
            "user": {"username": "carol"},
            "channel": {"id": "C_CHAN"},
            "message": {"ts": "8888.0001"},
        }
        plugin._handle_interactive(payload)

        # _gate_pending should be cleared
        assert "tu3" not in plugin._gate_pending

    @pytest.mark.anyio
    async def test_gate_decision_abstain_ignored(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        event = GateDecisionEvent(
            tool_use_id="tu1",
            tool_name="Read",
            action="abstain",
        )
        await plugin.handle_event(event)
        mock_web.chat_postMessage.assert_not_called()

    @pytest.mark.anyio
    async def test_gate_decision_no_reason(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        event = GateDecisionEvent(
            tool_use_id="tu1",
            tool_name="Bash",
            action="block",
            reason="",
            tool_input=None,
        )
        await plugin.handle_event(event)

        kwargs = mock_web.chat_postMessage.call_args.kwargs
        assert "BLOCK" in kwargs["text"]
        # No reason or tool_input lines
        assert ">" not in kwargs["text"]
        assert "```" not in kwargs["text"]

    @pytest.mark.anyio
    async def test_gate_decision_large_tool_input(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        event = GateDecisionEvent(
            tool_use_id="tu1",
            tool_name="Bash",
            action="block",
            reason="bad",
            tool_input={"command": "x" * 1000},
        )
        await plugin.handle_event(event)

        kwargs = mock_web.chat_postMessage.call_args.kwargs
        # Tool input should be truncated
        assert "..." in kwargs["text"]

    @pytest.mark.anyio
    async def test_sentinel_review_alert(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        event = SentinelReviewEvent(
            action="alert",
            summary="All looks good",
            concerns="None",
            event_count=5,
        )
        await plugin.handle_event(event)

        kwargs = mock_web.chat_postMessage.call_args.kwargs
        assert "ALERT" in kwargs["text"]
        assert "5 events reviewed" in kwargs["text"]
        assert "All looks good" in kwargs["text"]
        # "None" concerns should be excluded
        assert "Concerns" not in kwargs["text"]
        # alert should NOT reply_broadcast
        assert kwargs["reply_broadcast"] is False

    @pytest.mark.anyio
    async def test_sentinel_review_escalate(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        event = SentinelReviewEvent(
            action="escalate",
            summary="Suspicious activity",
            concerns="Agent reading credentials",
            event_count=10,
        )
        await plugin.handle_event(event)

        kwargs = mock_web.chat_postMessage.call_args.kwargs
        assert "ESCALATE" in kwargs["text"]
        assert "Concerns" in kwargs["text"]
        assert kwargs["reply_broadcast"] is True

    @pytest.mark.anyio
    async def test_sentinel_review_halt_session(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        event = SentinelReviewEvent(
            action="halt_session",
            summary="Critical issue",
            concerns="Data exfiltration attempt",
            event_count=20,
        )
        await plugin.handle_event(event)

        kwargs = mock_web.chat_postMessage.call_args.kwargs
        assert "HALT_SESSION" in kwargs["text"]
        assert kwargs["reply_broadcast"] is True

    @pytest.mark.anyio
    async def test_sentinel_review_allow_ignored(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        event = SentinelReviewEvent(action="allow", summary="ok", event_count=3)
        await plugin.handle_event(event)
        mock_web.chat_postMessage.assert_not_called()

    @pytest.mark.anyio
    async def test_sentinel_review_no_summary(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        event = SentinelReviewEvent(action="alert", summary="", concerns="", event_count=2)
        await plugin.handle_event(event)

        kwargs = mock_web.chat_postMessage.call_args.kwargs
        assert "Summary" not in kwargs["text"]

    @pytest.mark.anyio
    async def test_error_event(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        event = ErrorEvent(status_code=429, body="Rate limited")
        await plugin.handle_event(event)

        kwargs = mock_web.chat_postMessage.call_args.kwargs
        assert "429" in kwargs["text"]
        assert "Rate limited" in kwargs["text"]

    @pytest.mark.anyio
    async def test_unhandled_event_type(self, slack_env):
        """Events like RequestEvent, MessageStartEvent, MessageDeltaEvent should be ignored."""
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        await plugin.handle_event(RequestEvent(method="POST", path="/v1/messages"))
        await plugin.handle_event(MessageStartEvent(message_id="m1", model="claude"))
        await plugin.handle_event(MessageDeltaEvent(stop_reason="end_turn"))

        mock_web.chat_postMessage.assert_not_called()
        mock_web.chat_update.assert_not_called()

    @pytest.mark.anyio
    async def test_most_recent_session_used(self, slack_env):
        """handle_event should post to the most recently registered session's thread."""
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web, session_id="sess-1")
        # Register a second session
        mock_web.chat_postMessage.return_value = {"ts": "second_ts"}
        mock_web.conversations_list.return_value = {"channels": []}
        mock_web.conversations_create.return_value = {"channel": {"id": "C_CHAN2"}}
        mock_web.users_list.return_value = {"members": []}
        plugin.session_started_from_data(
            {"session_id": "sess-2", "project": "proj2", "agent": "codex", "cwd": "/tmp/proj2"}
        )
        mock_web.reset_mock()

        event = ErrorEvent(status_code=500, body="fail")
        await plugin.handle_event(event)

        # Should post to sess-2's thread (most recent)
        kwargs = mock_web.chat_postMessage.call_args.kwargs
        assert kwargs["thread_ts"] == "second_ts"

    @pytest.mark.anyio
    async def test_sentinel_concerns_none_lowercase(self, slack_env):
        """Concerns that are literally 'none' (case-insensitive) should be excluded."""
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        event = SentinelReviewEvent(action="alert", concerns="none", event_count=1)
        await plugin.handle_event(event)

        kwargs = mock_web.chat_postMessage.call_args.kwargs
        assert "Concerns" not in kwargs["text"]


# ---------------------------------------------------------------------------
# _on_socket_event
# ---------------------------------------------------------------------------


class TestOnSocketEvent:
    def _make_request(self, req_type="events_api", event=None, envelope_id="env1"):
        req = MagicMock()
        req.type = req_type
        req.envelope_id = envelope_id
        req.payload = {"event": event} if event else {}
        return req

    def test_acks_every_event(self, slack_env):
        factory, _, _ = slack_env
        plugin, _ = factory()
        client = MagicMock()
        req = self._make_request(req_type="unknown_type")
        plugin._on_socket_event(client, req)
        client.send_socket_mode_response.assert_called_once()

    def test_ignores_non_events_api(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        client = MagicMock()
        req = self._make_request(req_type="slash_commands")
        plugin._on_socket_event(client, req)
        mock_web.chat_postMessage.assert_not_called()

    def test_ignores_non_message_events(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        client = MagicMock()
        req = self._make_request(event={"type": "reaction_added"})
        plugin._on_socket_event(client, req)
        mock_web.chat_postMessage.assert_not_called()

    def test_ignores_bot_messages(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        plugin._bot_user_id = "U_BOT"
        client = MagicMock()
        req = self._make_request(event={"type": "message", "user": "U_BOT", "text": "hello"})
        plugin._on_socket_event(client, req)
        mock_web.chat_postMessage.assert_not_called()

    def test_ignores_bot_id_messages(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        plugin._bot_user_id = "U_BOT"
        client = MagicMock()
        # Has a user that is NOT the bot, but does have bot_id
        req = self._make_request(event={"type": "message", "user": "U_OTHER_BOT", "bot_id": "B123", "text": "hello"})
        plugin._on_socket_event(client, req)
        mock_web.chat_postMessage.assert_not_called()

    def test_ignores_empty_text(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        client = MagicMock()
        req = self._make_request(event={"type": "message", "user": "U_HUMAN", "text": "  "})
        plugin._on_socket_event(client, req)
        mock_web.chat_postMessage.assert_not_called()

    def test_threaded_message_injects(self, slack_env):
        factory, _, _ = slack_env
        inject_cb = AsyncMock()
        loop = asyncio.new_event_loop()
        plugin, mock_web = factory(inject_callback=inject_cb, event_loop=loop)
        _register_session(plugin, mock_web)

        client = MagicMock()
        req = self._make_request(
            event={
                "type": "message",
                "user": "U_HUMAN",
                "text": "fix the bug",
                "thread_ts": "1234567890.000001",
                "channel": "C_CHAN",
            }
        )
        plugin._on_socket_event(client, req)

        # Give the future a moment to be scheduled
        loop.run_until_complete(asyncio.sleep(0.05))
        inject_cb.assert_called_once_with("sess-1", "fix the bug\r")
        loop.close()

    def test_threaded_message_no_match(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        client = MagicMock()
        req = self._make_request(
            event={
                "type": "message",
                "user": "U_HUMAN",
                "text": "hello",
                "thread_ts": "nonexistent_ts",
                "channel": "C_CHAN",
            }
        )
        plugin._on_socket_event(client, req)
        # No matching session — nothing should be posted or injected
        mock_web.chat_postMessage.assert_not_called()

    def test_threaded_message_no_inject_callback(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory(inject_callback=None)
        _register_session(plugin, mock_web)

        client = MagicMock()
        req = self._make_request(
            event={
                "type": "message",
                "user": "U_HUMAN",
                "text": "hello",
                "thread_ts": "1234567890.000001",
                "channel": "C_CHAN",
            }
        )
        mock_web.chat_postMessage.reset_mock()
        plugin._on_socket_event(client, req)
        # No inject callback — message acknowledged but not injected
        mock_web.chat_postMessage.assert_not_called()

    def test_threaded_permission_pending_deny_feedback(self, slack_env):
        factory, _, _ = slack_env
        inject_cb = AsyncMock()
        loop = asyncio.new_event_loop()
        plugin, mock_web = factory(inject_callback=inject_cb, event_loop=loop)
        _register_session(plugin, mock_web)
        plugin._permission_pending["sess-1"] = ("C_CHAN", "perm_ts")

        client = MagicMock()
        req = self._make_request(
            event={
                "type": "message",
                "user": "U_HUMAN",
                "text": "no, use a safer approach",
                "thread_ts": "1234567890.000001",
                "channel": "C_CHAN",
            }
        )

        # The module does `import time` inline, then calls `time.sleep(0.5)`.
        # We patch `time.sleep` at the module level to avoid actual sleeping.
        with patch("time.sleep"):
            plugin._on_socket_event(client, req)

        # Should inject "3" (deny) then feedback
        loop.run_until_complete(asyncio.sleep(0.05))
        calls = inject_cb.call_args_list
        assert len(calls) == 2
        assert calls[0] == call("sess-1", "3")
        assert calls[1] == call("sess-1", "no, use a safer approach\r")

        # Permission should be cleared
        assert "sess-1" not in plugin._permission_pending
        # Should post the denial feedback message
        mock_web.chat_postMessage.assert_called()
        loop.close()

    def test_non_threaded_without_mention_ignored(self, slack_env):
        factory, _, _ = slack_env
        spawn_cb = AsyncMock()
        plugin, mock_web = factory(spawn_callback=spawn_cb)
        plugin._bot_user_id = "U_BOT"
        plugin._channels["cross-myproj"] = "C_CHAN"

        client = MagicMock()
        req = self._make_request(
            event={
                "type": "message",
                "user": "U_HUMAN",
                "text": "hello there",
                "channel": "C_CHAN",
            }
        )
        plugin._on_socket_event(client, req)
        spawn_cb.assert_not_called()

    def test_non_threaded_with_mention_spawns(self, slack_env):
        factory, _, _ = slack_env
        spawn_cb = AsyncMock()
        loop = asyncio.new_event_loop()
        plugin, mock_web = factory(spawn_callback=spawn_cb, event_loop=loop)
        plugin._bot_user_id = "U_BOT"
        plugin._channels["cross-myproj"] = "C_CHAN"

        client = MagicMock()
        req = self._make_request(
            event={
                "type": "message",
                "user": "U_HUMAN",
                "text": "<@U_BOT> run the tests",
                "channel": "C_CHAN",
                "ts": "msg_ts",
            }
        )
        plugin._on_socket_event(client, req)

        loop.run_until_complete(asyncio.sleep(0.05))
        spawn_cb.assert_called_once_with("myproj", "run the tests")
        mock_web.reactions_add.assert_called_once()
        assert "myproj" in plugin._pending_thread_ts
        loop.close()

    def test_non_threaded_mention_empty_after_strip(self, slack_env):
        factory, _, _ = slack_env
        spawn_cb = AsyncMock()
        plugin, mock_web = factory(spawn_callback=spawn_cb)
        plugin._bot_user_id = "U_BOT"
        plugin._channels["cross-myproj"] = "C_CHAN"

        client = MagicMock()
        req = self._make_request(
            event={
                "type": "message",
                "user": "U_HUMAN",
                "text": "<@U_BOT>",
                "channel": "C_CHAN",
            }
        )
        plugin._on_socket_event(client, req)
        spawn_cb.assert_not_called()

    def test_non_threaded_no_spawn_callback(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory(spawn_callback=None)
        plugin._bot_user_id = "U_BOT"

        client = MagicMock()
        req = self._make_request(
            event={
                "type": "message",
                "user": "U_HUMAN",
                "text": "<@U_BOT> hello",
                "channel": "C_CHAN",
            }
        )
        plugin._on_socket_event(client, req)
        # No spawn callback — mention acknowledged but no session spawned
        mock_web.chat_postMessage.assert_not_called()

    def test_non_threaded_no_bot_user_id(self, slack_env):
        factory, _, _ = slack_env
        spawn_cb = AsyncMock()
        plugin, _ = factory(spawn_callback=spawn_cb)
        plugin._bot_user_id = None

        client = MagicMock()
        req = self._make_request(
            event={
                "type": "message",
                "user": "U_HUMAN",
                "text": "hello",
                "channel": "C_CHAN",
            }
        )
        plugin._on_socket_event(client, req)
        # No bot_user_id — can't detect mentions, so no spawn
        spawn_cb.assert_not_called()

    def test_non_threaded_no_project_for_channel(self, slack_env):
        factory, _, _ = slack_env
        spawn_cb = AsyncMock()
        plugin, mock_web = factory(spawn_callback=spawn_cb)
        plugin._bot_user_id = "U_BOT"
        # No channels cached, API lookup returns no "cross" in name
        mock_web.conversations_info.return_value = {"channel": {"name": "random-channel"}}

        client = MagicMock()
        req = self._make_request(
            event={
                "type": "message",
                "user": "U_HUMAN",
                "text": "<@U_BOT> hello",
                "channel": "C_UNKNOWN",
            }
        )
        plugin._on_socket_event(client, req)
        spawn_cb.assert_not_called()

    def test_interactive_routes_to_handler(self, slack_env):
        factory, _, _ = slack_env
        inject_cb = AsyncMock()
        loop = asyncio.new_event_loop()
        plugin, mock_web = factory(inject_callback=inject_cb, event_loop=loop)

        client = MagicMock()
        req = MagicMock()
        req.type = "interactive"
        req.envelope_id = "env1"
        req.payload = {
            "type": "block_actions",
            "actions": [{"action_id": "permission_approve", "value": "sess-1"}],
            "user": {"username": "testuser"},
            "channel": {"id": "C_CHAN"},
            "message": {"ts": "msg_ts"},
        }

        plugin._on_socket_event(client, req)

        loop.run_until_complete(asyncio.sleep(0.05))
        inject_cb.assert_called_once_with("sess-1", "1")
        loop.close()


# ---------------------------------------------------------------------------
# _handle_interactive
# ---------------------------------------------------------------------------


class TestHandleInteractive:
    def test_approve(self, slack_env):
        factory, _, _ = slack_env
        inject_cb = AsyncMock()
        loop = asyncio.new_event_loop()
        plugin, mock_web = factory(inject_callback=inject_cb, event_loop=loop)
        plugin._permission_pending["sess-1"] = ("C_CHAN", "perm_ts")

        payload = {
            "type": "block_actions",
            "actions": [{"action_id": "permission_approve", "value": "sess-1"}],
            "user": {"username": "alice"},
            "channel": {"id": "C_CHAN"},
            "message": {"ts": "msg_ts"},
        }

        plugin._handle_interactive(payload)

        loop.run_until_complete(asyncio.sleep(0.05))
        inject_cb.assert_called_once_with("sess-1", "1")
        assert "sess-1" not in plugin._permission_pending
        mock_web.chat_update.assert_called_once()
        update_text = mock_web.chat_update.call_args.kwargs["text"]
        assert "Approved" in update_text
        assert "alice" in update_text
        loop.close()

    def test_allow_all(self, slack_env):
        factory, _, _ = slack_env
        inject_cb = AsyncMock()
        loop = asyncio.new_event_loop()
        plugin, mock_web = factory(inject_callback=inject_cb, event_loop=loop)

        payload = {
            "type": "block_actions",
            "actions": [{"action_id": "permission_allow_all", "value": "sess-2"}],
            "user": {"username": "bob"},
            "channel": {"id": "C_CHAN"},
            "message": {"ts": "msg_ts"},
        }

        plugin._handle_interactive(payload)

        loop.run_until_complete(asyncio.sleep(0.05))
        inject_cb.assert_called_once_with("sess-2", "2")
        loop.close()

    def test_deny(self, slack_env):
        factory, _, _ = slack_env
        inject_cb = AsyncMock()
        loop = asyncio.new_event_loop()
        plugin, mock_web = factory(inject_callback=inject_cb, event_loop=loop)

        payload = {
            "type": "block_actions",
            "actions": [{"action_id": "permission_deny", "value": "sess-3"}],
            "user": {"username": "carol"},
            "channel": {"id": "C_CHAN"},
            "message": {"ts": "msg_ts"},
        }

        plugin._handle_interactive(payload)

        loop.run_until_complete(asyncio.sleep(0.05))
        inject_cb.assert_called_once_with("sess-3", "3")
        update_text = mock_web.chat_update.call_args.kwargs["text"]
        assert "Denied" in update_text
        assert "carol" in update_text
        loop.close()

    def test_unknown_action_ignored(self, slack_env):
        factory, _, _ = slack_env
        inject_cb = AsyncMock()
        plugin, mock_web = factory(inject_callback=inject_cb)

        payload = {
            "type": "block_actions",
            "actions": [{"action_id": "some_other_action", "value": "sess-1"}],
            "user": {"username": "dave"},
            "channel": {"id": "C_CHAN"},
            "message": {"ts": "msg_ts"},
        }

        plugin._handle_interactive(payload)
        inject_cb.assert_not_called()
        mock_web.chat_update.assert_not_called()

    def test_not_block_actions_ignored(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        payload = {"type": "view_submission", "actions": []}
        plugin._handle_interactive(payload)
        mock_web.chat_update.assert_not_called()

    def test_empty_actions_ignored(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        payload = {"type": "block_actions", "actions": []}
        plugin._handle_interactive(payload)
        mock_web.chat_update.assert_not_called()

    def test_update_fails_gracefully(self, slack_env):
        factory, _, _ = slack_env
        inject_cb = AsyncMock()
        loop = asyncio.new_event_loop()
        plugin, mock_web = factory(inject_callback=inject_cb, event_loop=loop)
        mock_web.chat_update.side_effect = Exception("update failed")

        payload = {
            "type": "block_actions",
            "actions": [{"action_id": "permission_approve", "value": "sess-1"}],
            "user": {"username": "alice"},
            "channel": {"id": "C_CHAN"},
            "message": {"ts": "msg_ts"},
        }

        # Should not raise
        plugin._handle_interactive(payload)
        loop.run_until_complete(asyncio.sleep(0.05))
        inject_cb.assert_called_once_with("sess-1", "1")
        loop.close()

    def test_no_channel_or_message(self, slack_env):
        factory, _, _ = slack_env
        inject_cb = AsyncMock()
        loop = asyncio.new_event_loop()
        plugin, mock_web = factory(inject_callback=inject_cb, event_loop=loop)

        payload = {
            "type": "block_actions",
            "actions": [{"action_id": "permission_approve", "value": "sess-1"}],
            "user": {"username": "alice"},
            "channel": {},
            "message": {},
        }

        plugin._handle_interactive(payload)
        loop.run_until_complete(asyncio.sleep(0.05))
        # Should inject but not try to update (no channel_id / message_ts)
        inject_cb.assert_called_once()
        mock_web.chat_update.assert_not_called()
        loop.close()


# ---------------------------------------------------------------------------
# _inject
# ---------------------------------------------------------------------------


class TestInject:
    def test_inject_with_loop(self, slack_env):
        factory, _, _ = slack_env
        inject_cb = AsyncMock()
        loop = asyncio.new_event_loop()
        plugin, _ = factory(inject_callback=inject_cb, event_loop=loop)

        plugin._inject("sess-1", "hello")
        loop.run_until_complete(asyncio.sleep(0.05))

        inject_cb.assert_called_once_with("sess-1", "hello")
        loop.close()

    def test_inject_no_callback(self, slack_env):
        factory, _, _ = slack_env
        plugin, _ = factory(inject_callback=None)
        # Should warn, not raise
        plugin._inject("sess-1", "hello")

    def test_inject_no_event_loop(self, slack_env):
        factory, _, _ = slack_env
        inject_cb = AsyncMock()
        plugin, _ = factory(inject_callback=inject_cb, event_loop=None)
        # Should warn, not raise
        plugin._inject("sess-1", "hello")
        inject_cb.assert_not_called()


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_session_registration(self, slack_env):
        """Multiple threads registering sessions should not corrupt state."""
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        mock_web.conversations_list.return_value = {"channels": []}
        mock_web.conversations_create.return_value = {"channel": {"id": "C_CHAN"}}
        mock_web.users_list.return_value = {"members": []}
        mock_web.chat_postMessage.return_value = {"ts": "ts1"}

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
        assert len(plugin._threads) == 10

    def test_concurrent_session_end(self, slack_env):
        """Multiple threads ending sessions should not corrupt state."""
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        mock_web.conversations_list.return_value = {"channels": []}
        mock_web.conversations_create.return_value = {"channel": {"id": "C_CHAN"}}
        mock_web.users_list.return_value = {"members": []}
        mock_web.chat_postMessage.return_value = {"ts": "ts1"}

        for i in range(10):
            plugin.session_started_from_data(
                {"session_id": f"sess-{i}", "project": "proj", "agent": "claude", "cwd": "/tmp"}
            )

        def end_session(i):
            plugin.session_ended_from_data({"session_id": f"sess-{i}", "exit_code": 0})

        threads = [threading.Thread(target=end_session, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(plugin._sessions) == 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_session_end_missing_exit_code(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        plugin.session_ended_from_data({"session_id": "sess-1"})
        msg = mock_web.chat_postMessage.call_args.kwargs["text"]
        assert "exit code ?" in msg

    def test_session_end_no_timestamps(self, slack_env):
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        plugin.session_ended_from_data({"session_id": "sess-1", "exit_code": 0})
        msg = mock_web.chat_postMessage.call_args.kwargs["text"]
        assert "after" not in msg

    @pytest.mark.anyio
    async def test_handle_event_with_single_thread(self, slack_env):
        """Verify handle_event works correctly with a single registered thread."""
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        plugin._threads["sess-1"] = ("C_CHAN", "ts1")

        event = ErrorEvent(status_code=500, body="error")
        await plugin.handle_event(event)

        mock_web.chat_postMessage.assert_called_once()

    @pytest.mark.anyio
    async def test_handle_event_thread_info_none_falls_back(self, slack_env):
        """When thread_info is None (race condition), falls back to default channel."""
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        # Store None as the value to simulate the race condition
        plugin._threads["sess-1"] = None

        event = ErrorEvent(status_code=500, body="error")
        await plugin.handle_event(event)

        # Should post to fallback channel with no thread
        mock_web.chat_postMessage.assert_called_once()
        kwargs = mock_web.chat_postMessage.call_args.kwargs
        assert kwargs["thread_ts"] is None

    @pytest.mark.anyio
    async def test_gate_decision_small_tool_input(self, slack_env):
        """Tool input that fits within 500 chars should not have ellipsis."""
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        event = GateDecisionEvent(
            tool_use_id="tu1",
            tool_name="Bash",
            action="block",
            reason="bad",
            tool_input={"command": "ls"},
        )
        await plugin.handle_event(event)

        kwargs = mock_web.chat_postMessage.call_args.kwargs
        assert "..." not in kwargs["text"]

    @pytest.mark.anyio
    async def test_error_event_long_body(self, slack_env):
        """ErrorEvent body is sliced to 500 chars in the Slack message."""
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()

        long_body = "x" * 1000
        event = ErrorEvent(status_code=500, body=long_body)
        await plugin.handle_event(event)

        kwargs = mock_web.chat_postMessage.call_args.kwargs
        # The body[:500] slice means only first 500 chars appear
        assert "x" * 500 in kwargs["text"]
        # Full 1000-char body should not appear
        assert "x" * 501 not in kwargs["text"]

    def test_debounce_resets_after_interval(self, slack_env):
        """After the debounce interval, a new permission prompt should post."""
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()
        mock_web.chat_postMessage.return_value = {"ts": "perm_ts"}

        plugin.handle_pty_output("sess-1", "Do you want to allow this?")
        assert mock_web.chat_postMessage.call_count == 1

        # Simulate time passing past the debounce interval
        plugin._last_permission_post["sess-1"] -= 10.0
        mock_web.reset_mock()
        mock_web.chat_postMessage.return_value = {"ts": "perm_ts2"}

        plugin.handle_pty_output("sess-1", "Do you want to allow this?")
        assert mock_web.chat_postMessage.call_count == 1

    def test_handle_pty_output_default_allow_all_label(self, slack_env):
        """When _extract_allow_all returns None, the default label is used."""
        factory, _, _ = slack_env
        plugin, mock_web = factory()
        _register_session(plugin, mock_web)
        mock_web.reset_mock()
        mock_web.chat_postMessage.return_value = {"ts": "perm_ts"}

        # "Do you want to" triggers prompt, but no "allow all" pattern
        plugin.handle_pty_output("sess-1", "Do you want to proceed?")

        call_kwargs = mock_web.chat_postMessage.call_args.kwargs
        blocks = call_kwargs["blocks"]
        actions = [b for b in blocks if b["type"] == "actions"][0]
        allow_all_btn = [e for e in actions["elements"] if e["action_id"] == "permission_allow_all"][0]
        assert allow_all_btn["text"]["text"] == "Allow all (session)"
