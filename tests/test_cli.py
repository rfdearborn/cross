"""Tests for the CLI entry point module."""

from __future__ import annotations

import logging
import os
import queue
from unittest.mock import MagicMock, patch

import pytest

from cross.cli import (
    _end_session,
    _get_installed_version,
    _on_pty_output,
    _parse_agent_argv,
    _register_session,
    _run_update,
    main,
)


class TestParseAgentArgv:
    def test_strips_leading_double_dash(self):
        assert _parse_agent_argv(["--", "claude", "--help"]) == ["claude", "--help"]

    def test_returns_remainder_without_dash(self):
        assert _parse_agent_argv(["claude", "--help"]) == ["claude", "--help"]

    def test_empty_list(self):
        assert _parse_agent_argv([]) == []

    def test_only_double_dash(self):
        assert _parse_agent_argv(["--"]) == []

    def test_multiple_double_dashes(self):
        # Only the leading -- is stripped
        assert _parse_agent_argv(["--", "--", "claude"]) == ["--", "claude"]


class TestOnPtyOutput:
    def test_queues_cleaned_output(self):
        q = queue.Queue()
        _on_pty_output(b"Hello, world!", q)
        msg = q.get_nowait()
        assert msg["type"] == "pty_output"
        assert "Hello, world!" in msg["text"]

    def test_skips_empty_output(self):
        q = queue.Queue()
        _on_pty_output(b"", q)
        assert q.empty()

    def test_skips_whitespace_only_output(self):
        q = queue.Queue()
        _on_pty_output(b"   \n\t  ", q)
        assert q.empty()

    def test_strips_ansi_codes(self):
        q = queue.Queue()
        _on_pty_output(b"\x1b[32mGreen text\x1b[0m", q)
        msg = q.get_nowait()
        assert "Green text" in msg["text"]
        assert "\x1b" not in msg["text"]


class TestMainCommandRouting:
    @patch("cross.cli._run_daemon")
    def test_daemon_command(self, mock_run_daemon):
        with patch("sys.argv", ["cross", "daemon"]):
            main()
        mock_run_daemon.assert_called_once()

    @patch("cross.cli._run_proxy")
    def test_proxy_command(self, mock_run_proxy):
        with patch("sys.argv", ["cross", "proxy"]):
            main()
        mock_run_proxy.assert_called_once()

    @patch("cross.cli._run_wrap", return_value=0)
    def test_wrap_command_with_agent(self, mock_run_wrap):
        with patch("sys.argv", ["cross", "wrap", "--", "claude"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        mock_run_wrap.assert_called_once()
        # First arg should be the argv list
        argv_arg = mock_run_wrap.call_args[0][0]
        assert argv_arg == ["claude"]

    def test_wrap_command_without_agent_errors(self):
        with patch("sys.argv", ["cross", "wrap", "--"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # argparse error exits with code 2
            assert exc_info.value.code == 2

    def test_no_command_prints_help(self):
        with patch("sys.argv", ["cross"]):
            with patch("argparse.ArgumentParser.print_help") as mock_help:
                main()
            mock_help.assert_called_once()


class TestMainLoggingSetup:
    @patch("cross.cli._run_daemon")
    @patch("logging.basicConfig")
    def test_logging_configured(self, mock_basic_config, mock_run_daemon):
        with patch("sys.argv", ["cross", "daemon"]):
            main()
        mock_basic_config.assert_called_once()
        kwargs = mock_basic_config.call_args[1]
        assert kwargs["level"] == logging.INFO


class TestRunDaemon:
    @patch("uvicorn.run")
    def test_starts_uvicorn_with_daemon_app(self, mock_uvicorn_run):
        from cross.cli import _run_daemon

        _run_daemon()
        mock_uvicorn_run.assert_called_once()
        args, kwargs = mock_uvicorn_run.call_args
        assert args[0] == "cross.daemon:app"
        assert kwargs["log_level"] == "warning"


class TestRunProxy:
    @patch("uvicorn.run")
    def test_starts_uvicorn_with_starlette_app(self, mock_uvicorn_run):
        from cross.cli import _run_proxy

        _run_proxy()
        mock_uvicorn_run.assert_called_once()
        args, kwargs = mock_uvicorn_run.call_args
        # First arg is a Starlette app instance (not a string import path)
        from starlette.applications import Starlette

        assert isinstance(args[0], Starlette)
        assert kwargs["log_level"] == "warning"


class TestRunWrap:
    @patch("cross.cli._end_session")
    @patch("cross.cli._start_ws_relay")
    @patch("cross.cli._register_session")
    @patch("shutil.which", return_value="/usr/bin/claude")
    def test_agent_not_found(self, mock_which, mock_register, mock_ws, mock_end):
        from cross.cli import _run_wrap

        mock_which.return_value = None
        log = logging.getLogger("test")
        exit_code = _run_wrap(["nonexistent"], log)
        assert exit_code == 127

    @patch("cross.cli._end_session")
    @patch("cross.cli._start_ws_relay")
    @patch("cross.cli._register_session")
    @patch("shutil.which", return_value="/usr/bin/claude")
    def test_successful_wrap(self, mock_which, mock_register, mock_ws, mock_end):
        from cross.cli import _run_wrap
        from cross.session import registry

        log = logging.getLogger("test")

        # Mock the PTY session spawn to return an exit code without actually forking
        with patch.object(registry, "create") as mock_create:
            mock_info = MagicMock()
            mock_info.session_id = "abc123"
            mock_info.agent = "claude"
            mock_info.project = "test-project"
            mock_info.cwd = "/tmp/test"
            mock_info.started_at = 1234567890.0
            mock_info.pty_session = MagicMock()
            mock_info.pty_session.spawn.return_value = 0
            mock_create.return_value = mock_info

            with patch.object(registry, "complete") as mock_complete:
                exit_code = _run_wrap(["claude"], log)

            assert exit_code == 0
            mock_info.pty_session.on_output.assert_called_once()
            mock_info.pty_session.spawn.assert_called_once()
            mock_complete.assert_called_once_with("abc123", 0)
            mock_end.assert_called_once()

    @patch("cross.cli._end_session")
    @patch("cross.cli._start_ws_relay")
    @patch("cross.cli._register_session")
    @patch("shutil.which", return_value="/usr/bin/claude")
    def test_wrap_passes_env_with_proxy_url(self, mock_which, mock_register, mock_ws, mock_end):
        from cross.cli import _run_wrap
        from cross.config import settings
        from cross.session import registry

        log = logging.getLogger("test")

        with patch.object(registry, "create") as mock_create:
            mock_info = MagicMock()
            mock_info.session_id = "abc123"
            mock_info.agent = "claude"
            mock_info.project = "test-project"
            mock_info.cwd = "/tmp/test"
            mock_info.started_at = 1234567890.0
            mock_info.pty_session = MagicMock()
            mock_info.pty_session.spawn.return_value = 0
            mock_create.return_value = mock_info

            with patch.object(registry, "complete"):
                _run_wrap(["claude"], log)

            # Verify spawn was called with env containing the proxy URL
            spawn_call = mock_info.pty_session.spawn.call_args
            env_arg = spawn_call[1].get("env") or spawn_call[0][1] if len(spawn_call[0]) > 1 else spawn_call[1]["env"]
            assert "ANTHROPIC_BASE_URL" in env_arg
            assert str(settings.listen_port) in env_arg["ANTHROPIC_BASE_URL"]

    @patch("cross.cli._end_session")
    @patch("cross.cli._start_ws_relay")
    @patch("cross.cli._register_session")
    @patch("shutil.which", return_value="/usr/bin/claude")
    def test_wrap_resolves_binary_path(self, mock_which, mock_register, mock_ws, mock_end):
        from cross.cli import _run_wrap
        from cross.session import registry

        log = logging.getLogger("test")

        with patch.object(registry, "create") as mock_create:
            mock_info = MagicMock()
            mock_info.session_id = "abc123"
            mock_info.agent = "claude"
            mock_info.project = "test-project"
            mock_info.cwd = "/tmp/test"
            mock_info.started_at = 1234567890.0
            mock_info.pty_session = MagicMock()
            mock_info.pty_session.spawn.return_value = 0
            mock_create.return_value = mock_info

            with patch.object(registry, "complete"):
                _run_wrap(["claude", "--help"], log)

            # The argv passed to spawn should have the resolved path
            spawn_argv = mock_info.pty_session.spawn.call_args[0][0]
            assert spawn_argv[0] == "/usr/bin/claude"
            assert spawn_argv[1] == "--help"


class TestRegisterSession:
    @patch("httpx.post")
    def test_successful_registration(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_post.return_value = mock_response

        info = MagicMock()
        info.session_id = "abc123"
        info.agent = "claude"
        info.project = "test"
        info.cwd = "/tmp"
        info.started_at = 1234567890.0

        log = logging.getLogger("test")
        _register_session("http://localhost:8080", info, log)

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "/cross/sessions" in call_args[0][0]

    @patch("httpx.post")
    def test_registration_failure_logs_warning(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_post.return_value = mock_response

        info = MagicMock()
        info.session_id = "abc123"
        info.agent = "claude"
        info.project = "test"
        info.cwd = "/tmp"
        info.started_at = 1234567890.0

        log = MagicMock()
        _register_session("http://localhost:8080", info, log)
        log.warning.assert_called_once()

    @patch("httpx.post", side_effect=__import__("httpx").ConnectError("refused"))
    def test_connect_error_logs_warning(self, mock_post):
        info = MagicMock()
        info.session_id = "abc123"
        info.agent = "claude"
        info.project = "test"
        info.cwd = "/tmp"
        info.started_at = 1234567890.0

        log = MagicMock()
        _register_session("http://localhost:8080", info, log)
        log.warning.assert_called_once()
        assert "not running" in log.warning.call_args[0][0]


class TestEndSession:
    @patch("httpx.post")
    def test_sends_end_notification(self, mock_post):
        mock_post.return_value = MagicMock(status_code=200)

        info = MagicMock()
        info.session_id = "abc123"
        info.exit_code = 0
        info.started_at = 1234567890.0
        info.ended_at = 1234567900.0

        log = logging.getLogger("test")
        _end_session("http://localhost:8080", info, log)

        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "abc123/end" in call_args[0][0]

    @patch("httpx.post", side_effect=Exception("network error"))
    def test_end_session_swallows_exceptions(self, mock_post):
        info = MagicMock()
        info.session_id = "abc123"
        info.exit_code = 1
        info.started_at = 1234567890.0
        info.ended_at = 1234567900.0

        log = logging.getLogger("test")
        # Should not raise
        _end_session("http://localhost:8080", info, log)


class TestStartWsRelay:
    def test_returns_thread(self):
        from cross.cli import _start_ws_relay

        info = MagicMock()
        info.session_id = "abc123"
        output_queue = queue.Queue()
        log = logging.getLogger("test")

        # The thread will fail to connect but that's fine -- we just test it starts
        with patch("websockets.sync.client.connect", side_effect=Exception("no server")):
            thread = _start_ws_relay("http://localhost:8080", info, output_queue, log)

        assert thread is not None
        assert thread.daemon is True
        # Wait for thread to finish (it should fail quickly)
        thread.join(timeout=2.0)

    def test_relay_sends_queued_output(self):
        """The relay loop should send queued messages over the WebSocket."""
        from cross.cli import _start_ws_relay

        info = MagicMock()
        info.session_id = "abc123"
        output_queue = queue.Queue()
        log = logging.getLogger("test")

        mock_ws = MagicMock()
        # First recv times out, second recv raises to break the loop
        mock_ws.recv.side_effect = [TimeoutError, Exception("done")]

        output_queue.put({"type": "pty_output", "text": "hello"})

        with patch("websockets.sync.client.connect") as mock_connect:
            mock_connect.return_value.__enter__ = MagicMock(return_value=mock_ws)
            mock_connect.return_value.__exit__ = MagicMock(return_value=False)
            thread = _start_ws_relay("http://localhost:8080", info, output_queue, log)
            thread.join(timeout=2.0)

        mock_ws.send.assert_called_once()
        import json

        sent_data = json.loads(mock_ws.send.call_args[0][0])
        assert sent_data["type"] == "pty_output"
        assert sent_data["text"] == "hello"

    def test_relay_injects_received_messages(self):
        """The relay loop should inject received 'inject' messages into the PTY."""
        import json

        from cross.cli import _start_ws_relay

        info = MagicMock()
        info.session_id = "abc123"
        info.pty_session = MagicMock()
        output_queue = queue.Queue()
        log = logging.getLogger("test")

        inject_msg = json.dumps({"type": "inject", "text": "injected text"})
        mock_ws = MagicMock()
        # First recv returns an inject message, second raises to break
        mock_ws.recv.side_effect = [inject_msg, Exception("done")]

        with patch("websockets.sync.client.connect") as mock_connect:
            mock_connect.return_value.__enter__ = MagicMock(return_value=mock_ws)
            mock_connect.return_value.__exit__ = MagicMock(return_value=False)
            thread = _start_ws_relay("http://localhost:8080", info, output_queue, log)
            thread.join(timeout=2.0)

        info.pty_session.inject_input.assert_called_once_with(b"injected text")

    def test_relay_ws_url_conversion(self):
        """The relay should convert http:// to ws:// in the daemon URL."""
        from cross.cli import _start_ws_relay

        info = MagicMock()
        info.session_id = "session42"
        output_queue = queue.Queue()
        log = logging.getLogger("test")

        with patch("websockets.sync.client.connect", side_effect=Exception("no server")) as mock_connect:
            thread = _start_ws_relay("http://localhost:9090", info, output_queue, log)
            thread.join(timeout=2.0)

        mock_connect.assert_called_once_with("ws://localhost:9090/cross/sessions/session42/io")


class TestRunWrapExitCodePropagation:
    @patch("cross.cli._end_session")
    @patch("cross.cli._start_ws_relay")
    @patch("cross.cli._register_session")
    @patch("shutil.which", return_value="/usr/bin/claude")
    def test_wrap_propagates_nonzero_exit_code(self, mock_which, mock_register, mock_ws, mock_end):
        from cross.cli import _run_wrap
        from cross.session import registry

        log = logging.getLogger("test")

        with patch.object(registry, "create") as mock_create:
            mock_info = MagicMock()
            mock_info.session_id = "abc123"
            mock_info.agent = "claude"
            mock_info.project = "test-project"
            mock_info.cwd = "/tmp/test"
            mock_info.started_at = 1234567890.0
            mock_info.pty_session = MagicMock()
            mock_info.pty_session.spawn.return_value = 42
            mock_create.return_value = mock_info

            with patch.object(registry, "complete") as mock_complete:
                exit_code = _run_wrap(["claude"], log)

            assert exit_code == 42
            mock_complete.assert_called_once_with("abc123", 42)


class TestRunWrapOpenClaw:
    """Test OpenClaw-specific behavior in _run_wrap."""

    @patch("cross.cli._end_session")
    @patch("cross.cli._start_ws_relay")
    @patch("cross.cli._register_session")
    @patch("shutil.which", return_value="/usr/local/bin/openclaw")
    def test_openclaw_sets_node_options(self, mock_which, mock_register, mock_ws, mock_end):
        from cross.cli import _run_wrap
        from cross.session import registry

        log = logging.getLogger("test")

        with patch.object(registry, "create") as mock_create:
            mock_info = MagicMock()
            mock_info.session_id = "oc_sess_1"
            mock_info.agent = "openclaw"
            mock_info.project = "test-project"
            mock_info.cwd = "/tmp/test"
            mock_info.started_at = 1234567890.0
            mock_info.pty_session = MagicMock()
            mock_info.pty_session.spawn.return_value = 0
            mock_create.return_value = mock_info

            with patch.object(registry, "complete"):
                _run_wrap(["openclaw"], log)

            spawn_call = mock_info.pty_session.spawn.call_args
            env_arg = spawn_call[1].get("env") or spawn_call[0][1] if len(spawn_call[0]) > 1 else spawn_call[1]["env"]
            assert "NODE_OPTIONS" in env_arg
            assert "--import" in env_arg["NODE_OPTIONS"]
            assert "openclaw_hook.mjs" in env_arg["NODE_OPTIONS"]

    @patch("cross.cli._end_session")
    @patch("cross.cli._start_ws_relay")
    @patch("cross.cli._register_session")
    @patch("shutil.which", return_value="/usr/local/bin/openclaw")
    def test_openclaw_sets_cross_listen_port(self, mock_which, mock_register, mock_ws, mock_end):
        from cross.cli import _run_wrap
        from cross.config import settings
        from cross.session import registry

        log = logging.getLogger("test")

        with patch.object(registry, "create") as mock_create:
            mock_info = MagicMock()
            mock_info.session_id = "oc_sess_2"
            mock_info.agent = "openclaw"
            mock_info.project = "test-project"
            mock_info.cwd = "/tmp/test"
            mock_info.started_at = 1234567890.0
            mock_info.pty_session = MagicMock()
            mock_info.pty_session.spawn.return_value = 0
            mock_create.return_value = mock_info

            with patch.object(registry, "complete"):
                _run_wrap(["openclaw"], log)

            spawn_call = mock_info.pty_session.spawn.call_args
            env_arg = spawn_call[1].get("env") or spawn_call[0][1] if len(spawn_call[0]) > 1 else spawn_call[1]["env"]
            assert "CROSS_LISTEN_PORT" in env_arg
            assert env_arg["CROSS_LISTEN_PORT"] == str(settings.listen_port)

    @patch("cross.cli._end_session")
    @patch("cross.cli._start_ws_relay")
    @patch("cross.cli._register_session")
    @patch("shutil.which", return_value="/usr/local/bin/openclaw")
    def test_openclaw_sets_cross_session_id(self, mock_which, mock_register, mock_ws, mock_end):
        from cross.cli import _run_wrap
        from cross.session import registry

        log = logging.getLogger("test")

        with patch.object(registry, "create") as mock_create:
            mock_info = MagicMock()
            mock_info.session_id = "oc_sess_3"
            mock_info.agent = "openclaw"
            mock_info.project = "test-project"
            mock_info.cwd = "/tmp/test"
            mock_info.started_at = 1234567890.0
            mock_info.pty_session = MagicMock()
            mock_info.pty_session.spawn.return_value = 0
            mock_create.return_value = mock_info

            with patch.object(registry, "complete"):
                _run_wrap(["openclaw"], log)

            spawn_call = mock_info.pty_session.spawn.call_args
            env_arg = spawn_call[1].get("env") or spawn_call[0][1] if len(spawn_call[0]) > 1 else spawn_call[1]["env"]
            assert "CROSS_SESSION_ID" in env_arg
            assert env_arg["CROSS_SESSION_ID"] == "oc_sess_3"

    @patch("cross.cli._end_session")
    @patch("cross.cli._start_ws_relay")
    @patch("cross.cli._register_session")
    @patch("shutil.which", return_value="/usr/bin/claude")
    def test_non_openclaw_no_node_options(self, mock_which, mock_register, mock_ws, mock_end):
        """Non-OpenClaw agents should not get NODE_OPTIONS."""
        from cross.cli import _run_wrap
        from cross.session import registry

        log = logging.getLogger("test")

        with patch.object(registry, "create") as mock_create:
            mock_info = MagicMock()
            mock_info.session_id = "claude_sess_1"
            mock_info.agent = "claude"
            mock_info.project = "test-project"
            mock_info.cwd = "/tmp/test"
            mock_info.started_at = 1234567890.0
            mock_info.pty_session = MagicMock()
            mock_info.pty_session.spawn.return_value = 0
            mock_create.return_value = mock_info

            with patch.object(registry, "complete"):
                _run_wrap(["claude"], log)

            spawn_call = mock_info.pty_session.spawn.call_args
            env_arg = spawn_call[1].get("env") or spawn_call[0][1] if len(spawn_call[0]) > 1 else spawn_call[1]["env"]
            assert "NODE_OPTIONS" not in env_arg
            assert "CROSS_LISTEN_PORT" not in env_arg
            assert "CROSS_SESSION_ID" not in env_arg

    @patch("cross.cli._end_session")
    @patch("cross.cli._start_ws_relay")
    @patch("cross.cli._register_session")
    @patch("shutil.which", return_value="/usr/local/bin/openclaw")
    def test_openclaw_preserves_existing_node_options(self, mock_which, mock_register, mock_ws, mock_end):
        """Existing NODE_OPTIONS should be preserved when wrapping OpenClaw."""
        from cross.cli import _run_wrap
        from cross.session import registry

        log = logging.getLogger("test")

        with (
            patch.object(registry, "create") as mock_create,
            patch.dict(os.environ, {"NODE_OPTIONS": "--max-old-space-size=4096"}, clear=False),
        ):
            mock_info = MagicMock()
            mock_info.session_id = "oc_sess_4"
            mock_info.agent = "openclaw"
            mock_info.project = "test-project"
            mock_info.cwd = "/tmp/test"
            mock_info.started_at = 1234567890.0
            mock_info.pty_session = MagicMock()
            mock_info.pty_session.spawn.return_value = 0
            mock_create.return_value = mock_info

            with patch.object(registry, "complete"):
                _run_wrap(["openclaw"], log)

            spawn_call = mock_info.pty_session.spawn.call_args
            env_arg = spawn_call[1].get("env") or spawn_call[0][1] if len(spawn_call[0]) > 1 else spawn_call[1]["env"]
            assert "--import" in env_arg["NODE_OPTIONS"]
            assert "--max-old-space-size=4096" in env_arg["NODE_OPTIONS"]


class TestGetInstalledVersion:
    @patch("importlib.metadata.version", return_value="0.1.0")
    def test_finds_cross_ai(self, mock_version):
        assert _get_installed_version() == "0.1.0"
        mock_version.assert_called_with("cross-ai")

    @patch(
        "importlib.metadata.version",
        side_effect=[__import__("importlib").metadata.PackageNotFoundError, "0.1.0"],
    )
    def test_falls_back_to_cross(self, mock_version):
        assert _get_installed_version() == "0.1.0"
        assert mock_version.call_count == 2

    @patch(
        "importlib.metadata.version",
        side_effect=__import__("importlib").metadata.PackageNotFoundError,
    )
    def test_returns_none_when_not_found(self, mock_version):
        assert _get_installed_version() is None


class TestRunUpdate:
    @patch("subprocess.run")
    @patch("cross.cli._get_installed_version", return_value="0.1.0")
    def test_successful_update(self, mock_get_version, mock_run):
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),  # pip install
            MagicMock(returncode=0, stdout="0.2.0\n", stderr=""),  # version check
        ]
        exit_code = _run_update()
        assert exit_code == 0
        # First subprocess.run call should be pip install --upgrade cross-ai
        pip_call_args = mock_run.call_args_list[0][0][0]
        assert "pip" in pip_call_args
        assert "--upgrade" in pip_call_args
        assert "cross-ai" in pip_call_args

    @patch("subprocess.run")
    @patch("cross.cli._get_installed_version", return_value="0.1.0")
    def test_update_failure(self, mock_get_version, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="some error")
        exit_code = _run_update()
        assert exit_code == 1

    @patch("subprocess.run")
    @patch("cross.cli._get_installed_version", return_value="0.1.0")
    def test_already_up_to_date(self, mock_get_version, mock_run):
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),  # pip install
            MagicMock(returncode=0, stdout="0.1.0\n", stderr=""),  # version check
        ]
        exit_code = _run_update()
        assert exit_code == 0

    @patch("subprocess.run")
    @patch("cross.cli._get_installed_version", return_value=None)
    def test_unknown_current_version(self, mock_get_version, mock_run):
        mock_run.side_effect = [
            MagicMock(returncode=0, stdout="", stderr=""),  # pip install
            MagicMock(returncode=0, stdout="0.1.0\n", stderr=""),  # version check
        ]
        exit_code = _run_update()
        assert exit_code == 0

    @patch("cross.cli._run_update", return_value=0)
    def test_update_command_routing(self, mock_run_update):
        with patch("sys.argv", ["cross", "update"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        mock_run_update.assert_called_once()
