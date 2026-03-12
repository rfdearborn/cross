"""Tests for the PTY wrapper module."""

from __future__ import annotations

import errno
import os
import select
import signal
import sys
from unittest.mock import MagicMock, patch

import pytest

from cross.pty_wrapper import PTYSession


class TestPTYSessionInit:
    def test_default_values(self):
        session = PTYSession()
        assert session.pid == 0
        assert session.master_fd == -1
        assert session._original_termios is None
        assert session._output_callbacks == []
        assert session._input_callbacks == []

    def test_custom_values(self):
        session = PTYSession(pid=42, master_fd=5)
        assert session.pid == 42
        assert session.master_fd == 5


class TestCallbackRegistration:
    def test_on_output_registers_callback(self):
        session = PTYSession()
        cb = MagicMock()
        session.on_output(cb)
        assert cb in session._output_callbacks

    def test_on_input_registers_callback(self):
        session = PTYSession()
        cb = MagicMock()
        session.on_input(cb)
        assert cb in session._input_callbacks

    def test_multiple_output_callbacks(self):
        session = PTYSession()
        cb1 = MagicMock()
        cb2 = MagicMock()
        session.on_output(cb1)
        session.on_output(cb2)
        assert len(session._output_callbacks) == 2

    def test_multiple_input_callbacks(self):
        session = PTYSession()
        cb1 = MagicMock()
        cb2 = MagicMock()
        session.on_input(cb1)
        session.on_input(cb2)
        assert len(session._input_callbacks) == 2


class TestInjectInput:
    @patch("os.write")
    def test_inject_writes_to_master_fd(self, mock_write):
        session = PTYSession(master_fd=7)
        session.inject_input(b"hello")
        mock_write.assert_called_once_with(7, b"hello")

    @patch("os.write")
    def test_inject_skips_when_fd_negative(self, mock_write):
        session = PTYSession(master_fd=-1)
        session.inject_input(b"hello")
        mock_write.assert_not_called()

    @patch("os.write")
    def test_inject_empty_bytes(self, mock_write):
        session = PTYSession(master_fd=7)
        session.inject_input(b"")
        mock_write.assert_called_once_with(7, b"")


class TestSetupTerminal:
    @patch("tty.setraw")
    @patch("termios.tcgetattr", return_value=[1, 2, 3])
    def test_saves_and_sets_raw_when_tty(self, mock_tcgetattr, mock_setraw):
        session = PTYSession()
        with patch.object(sys.stdin, "isatty", return_value=True):
            session._setup_terminal()
        mock_tcgetattr.assert_called_once_with(sys.stdin)
        mock_setraw.assert_called_once_with(sys.stdin)
        assert session._original_termios == [1, 2, 3]

    @patch("tty.setraw")
    @patch("termios.tcgetattr")
    def test_skips_when_not_tty(self, mock_tcgetattr, mock_setraw):
        session = PTYSession()
        with patch.object(sys.stdin, "isatty", return_value=False):
            session._setup_terminal()
        mock_tcgetattr.assert_not_called()
        mock_setraw.assert_not_called()
        assert session._original_termios is None


class TestRestoreTerminal:
    @patch("termios.tcsetattr")
    def test_restores_saved_settings(self, mock_tcsetattr):
        import termios

        session = PTYSession()
        saved = [1, 2, 3, 4, 5, 6, []]
        session._original_termios = saved
        session._restore_terminal()
        mock_tcsetattr.assert_called_once_with(sys.stdin, termios.TCSAFLUSH, saved)
        assert session._original_termios is None

    @patch("termios.tcsetattr")
    def test_noop_when_no_saved_settings(self, mock_tcsetattr):
        session = PTYSession()
        session._original_termios = None
        session._restore_terminal()
        mock_tcsetattr.assert_not_called()


class TestCopyWinsize:
    @patch("fcntl.ioctl")
    def test_copies_winsize_between_fds(self, mock_ioctl):
        import termios

        winsize_bytes = b"\x18\x00\x50\x00\x00\x00\x00\x00"
        mock_ioctl.side_effect = [winsize_bytes, None]

        session = PTYSession()
        session._copy_winsize(0, 5)

        assert mock_ioctl.call_count == 2
        # First call: TIOCGWINSZ to read from src
        assert mock_ioctl.call_args_list[0][0][0] == 0
        assert mock_ioctl.call_args_list[0][0][1] == termios.TIOCGWINSZ
        # Second call: TIOCSWINSZ to write to dst
        assert mock_ioctl.call_args_list[1][0][0] == 5
        assert mock_ioctl.call_args_list[1][0][1] == termios.TIOCSWINSZ

    @patch("fcntl.ioctl", side_effect=OSError("not a tty"))
    def test_handles_oserror_gracefully(self, mock_ioctl):
        session = PTYSession()
        # Should not raise
        session._copy_winsize(0, 5)


class TestInstallSigwinch:
    @patch("signal.signal")
    def test_installs_sigwinch_handler(self, mock_signal):
        session = PTYSession()
        session._install_sigwinch()
        mock_signal.assert_called_once()
        assert mock_signal.call_args[0][0] == signal.SIGWINCH
        # Second argument should be a callable
        handler = mock_signal.call_args[0][1]
        assert callable(handler)

    @patch("signal.signal")
    @patch("os.kill")
    def test_sigwinch_handler_forwards_to_child(self, mock_kill, mock_signal):
        session = PTYSession(pid=1234, master_fd=5)

        # Capture the handler
        session._install_sigwinch()
        handler = mock_signal.call_args[0][1]

        # Call the handler; must mock stdin.fileno since pytest captures stdin
        with patch.object(sys.stdin, "fileno", return_value=0):
            with patch.object(session, "_copy_winsize"):
                handler(signal.SIGWINCH, None)

        mock_kill.assert_called_once_with(1234, signal.SIGWINCH)

    @patch("signal.signal")
    @patch("os.kill", side_effect=OSError("no such process"))
    def test_sigwinch_handler_ignores_kill_oserror(self, mock_kill, mock_signal):
        session = PTYSession(pid=1234, master_fd=5)

        session._install_sigwinch()
        handler = mock_signal.call_args[0][1]

        # Should not raise even if os.kill fails
        with patch.object(sys.stdin, "fileno", return_value=0):
            with patch.object(session, "_copy_winsize"):
                handler(signal.SIGWINCH, None)


class TestIOLoop:
    @patch("os.read")
    @patch("os.write")
    @patch("select.select")
    def test_master_output_written_to_stdout(self, mock_select, mock_write, mock_read):
        session = PTYSession(master_fd=5)
        stdout_fd = sys.stdout.fileno()

        # First iteration: master has data; second iteration: EIO (child exit)
        mock_select.side_effect = [
            ([5], [], []),
            ([], [], []),
        ]
        mock_read.side_effect = [
            b"hello from agent",
            OSError(errno.EIO, "I/O error"),
        ]

        # waitpid on second iteration (empty rfds -> check child)
        with patch("os.waitpid", return_value=(1, 0)):
            with patch.object(sys.stdin, "isatty", return_value=False):
                session._io_loop()

        # First read result should be written to stdout
        mock_write.assert_any_call(stdout_fd, b"hello from agent")

    @patch("os.read")
    @patch("os.write")
    @patch("select.select")
    def test_output_callbacks_invoked(self, mock_select, mock_write, mock_read):
        session = PTYSession(master_fd=5)
        cb = MagicMock()
        session.on_output(cb)

        mock_select.side_effect = [
            ([5], [], []),
            ([], [], []),
        ]
        mock_read.side_effect = [
            b"data chunk",
            OSError(errno.EIO, "I/O error"),
        ]

        with patch("os.waitpid", return_value=(1, 0)):
            with patch.object(sys.stdin, "isatty", return_value=False):
                session._io_loop()

        cb.assert_called_once_with(b"data chunk")

    @patch("os.read")
    @patch("os.write")
    @patch("select.select")
    def test_callback_exception_does_not_break_loop(self, mock_select, mock_write, mock_read):
        session = PTYSession(master_fd=5)
        bad_cb = MagicMock(side_effect=RuntimeError("callback failed"))
        good_cb = MagicMock()
        session.on_output(bad_cb)
        session.on_output(good_cb)

        mock_select.side_effect = [
            ([5], [], []),
            ([], [], []),
        ]
        mock_read.side_effect = [
            b"data",
            OSError(errno.EIO, "I/O error"),
        ]

        with patch("os.waitpid", return_value=(1, 0)):
            with patch.object(sys.stdin, "isatty", return_value=False):
                session._io_loop()

        # Both callbacks called; bad one raised but good one still ran
        bad_cb.assert_called_once_with(b"data")
        good_cb.assert_called_once_with(b"data")

    @patch("os.read")
    @patch("os.write")
    @patch("select.select")
    def test_eio_breaks_loop(self, mock_select, mock_write, mock_read):
        session = PTYSession(master_fd=5)

        mock_select.return_value = ([5], [], [])
        mock_read.side_effect = OSError(errno.EIO, "I/O error")

        with patch.object(sys.stdin, "isatty", return_value=False):
            session._io_loop()

        # Should break immediately on EIO
        assert mock_select.call_count == 1

    @patch("os.read")
    @patch("os.write")
    @patch("select.select")
    def test_empty_read_breaks_loop(self, mock_select, mock_write, mock_read):
        session = PTYSession(master_fd=5)

        mock_select.return_value = ([5], [], [])
        mock_read.return_value = b""  # EOF

        with patch.object(sys.stdin, "isatty", return_value=False):
            session._io_loop()

        assert mock_select.call_count == 1

    @patch("os.read")
    @patch("os.write")
    @patch("select.select")
    def test_stdin_input_written_to_master(self, mock_select, mock_write, mock_read):
        session = PTYSession(master_fd=5)
        stdin_fd = 0

        # First iteration: stdin has data; second: master EIO
        mock_select.side_effect = [
            ([stdin_fd], [], []),
            ([5], [], []),
        ]
        mock_read.side_effect = [
            b"user input",  # from stdin
            OSError(errno.EIO, "I/O error"),  # master gone
        ]

        with patch.object(sys.stdin, "isatty", return_value=True):
            with patch.object(sys.stdin, "fileno", return_value=stdin_fd):
                session._io_loop()

        # Input should be written to master_fd
        mock_write.assert_any_call(5, b"user input")

    @patch("os.read")
    @patch("os.write")
    @patch("select.select")
    def test_input_callbacks_invoked(self, mock_select, mock_write, mock_read):
        session = PTYSession(master_fd=5)
        cb = MagicMock()
        session.on_input(cb)
        stdin_fd = 0

        mock_select.side_effect = [
            ([stdin_fd], [], []),
            ([5], [], []),
        ]
        mock_read.side_effect = [
            b"typed text",
            OSError(errno.EIO, "I/O error"),
        ]

        with patch.object(sys.stdin, "isatty", return_value=True):
            with patch.object(sys.stdin, "fileno", return_value=stdin_fd):
                session._io_loop()

        cb.assert_called_once_with(b"typed text")

    @patch("os.read")
    @patch("os.write")
    @patch("select.select")
    def test_stdin_eof_removes_from_watch_fds(self, mock_select, mock_write, mock_read):
        """When stdin returns empty bytes, it should be removed from watch fds."""
        session = PTYSession(master_fd=5)
        stdin_fd = 0

        mock_select.side_effect = [
            ([stdin_fd], [], []),  # stdin ready
            ([5], [], []),  # now only master
        ]
        mock_read.side_effect = [
            b"",  # stdin EOF
            OSError(errno.EIO, "I/O error"),  # master done
        ]

        with patch.object(sys.stdin, "isatty", return_value=True):
            with patch.object(sys.stdin, "fileno", return_value=stdin_fd):
                session._io_loop()

        # Loop should have run twice (stdin EOF then master EIO)
        assert mock_select.call_count == 2

    @patch("os.read")
    @patch("os.write")
    @patch("select.select")
    def test_select_error_breaks_loop(self, mock_select, mock_write, mock_read):
        session = PTYSession(master_fd=5)

        mock_select.side_effect = ValueError("bad fd")

        with patch.object(sys.stdin, "isatty", return_value=False):
            session._io_loop()

        assert mock_select.call_count == 1

    @patch("os.read")
    @patch("os.write")
    @patch("select.select")
    @patch("os.waitpid")
    def test_child_exit_detected_on_timeout(self, mock_waitpid, mock_select, mock_write, mock_read):
        """When select times out, check if child exited."""
        session = PTYSession(pid=42, master_fd=5)

        # select returns empty (timeout) -> check child -> child exited
        mock_select.return_value = ([], [], [])
        mock_waitpid.return_value = (42, 0)
        # Drain remaining output
        mock_read.side_effect = OSError(errno.EIO, "done")

        with patch.object(sys.stdin, "isatty", return_value=False):
            session._io_loop()

    @patch("os.read")
    @patch("os.write")
    @patch("select.select")
    @patch("os.waitpid")
    def test_child_still_running_on_timeout(self, mock_waitpid, mock_select, mock_write, mock_read):
        """When select times out but child is still alive, loop continues."""
        session = PTYSession(pid=42, master_fd=5)

        # First: timeout + child still running; second: master has data; third: EIO
        mock_select.side_effect = [
            ([], [], []),
            ([5], [], []),
        ]
        mock_waitpid.return_value = (0, 0)  # child still running
        mock_read.side_effect = OSError(errno.EIO, "done")

        with patch.object(sys.stdin, "isatty", return_value=False):
            session._io_loop()

    @patch("os.read")
    @patch("os.write")
    @patch("select.select")
    @patch("os.waitpid", side_effect=ChildProcessError)
    def test_child_process_error_breaks_loop(self, mock_waitpid, mock_select, mock_write, mock_read):
        session = PTYSession(pid=42, master_fd=5)

        mock_select.return_value = ([], [], [])

        with patch.object(sys.stdin, "isatty", return_value=False):
            session._io_loop()


class TestSpawn:
    @patch("os.waitpid", return_value=(42, 0))
    @patch("os.close")
    @patch.object(PTYSession, "_io_loop")
    @patch.object(PTYSession, "_install_sigwinch")
    @patch.object(PTYSession, "_copy_winsize")
    @patch.object(PTYSession, "_restore_terminal")
    @patch.object(PTYSession, "_setup_terminal")
    @patch("pty.fork", return_value=(42, 7))
    def test_parent_runs_io_loop(
        self, mock_fork, mock_setup, mock_restore, mock_winsize, mock_sigwinch, mock_io_loop, mock_close, mock_waitpid
    ):
        """In the parent process (pid != 0), setup, io loop, and cleanup should run."""
        session = PTYSession()

        with patch.object(sys.stdin, "fileno", return_value=0):
            exit_code = session.spawn(["echo", "hi"])

        assert session.pid == 42
        mock_setup.assert_called_once()
        mock_winsize.assert_called_once_with(0, 7)
        mock_sigwinch.assert_called_once()
        mock_io_loop.assert_called_once()
        mock_restore.assert_called_once()
        mock_close.assert_called_once_with(7)
        assert session.master_fd == -1
        assert exit_code == 0

    @patch("os.waitpid", return_value=(42, 0))
    @patch("os.close")
    @patch.object(PTYSession, "_io_loop")
    @patch.object(PTYSession, "_install_sigwinch")
    @patch.object(PTYSession, "_copy_winsize")
    @patch.object(PTYSession, "_restore_terminal")
    @patch.object(PTYSession, "_setup_terminal")
    @patch("pty.fork", return_value=(42, 7))
    def test_returns_exit_code_from_child(
        self, mock_fork, mock_setup, mock_restore, mock_winsize, mock_sigwinch, mock_io_loop, mock_close, mock_waitpid
    ):
        # os.WIFEXITED(256) is True, os.WEXITSTATUS(256) is 1 on Linux/macOS
        # Simulate exit code 1: status byte = 0x0100
        mock_waitpid.return_value = (42, 256)

        session = PTYSession()
        with patch.object(sys.stdin, "fileno", return_value=0):
            exit_code = session.spawn(["false"])

        assert exit_code == 1

    @patch("os.waitpid", side_effect=ChildProcessError)
    @patch("os.close")
    @patch.object(PTYSession, "_io_loop")
    @patch.object(PTYSession, "_install_sigwinch")
    @patch.object(PTYSession, "_copy_winsize")
    @patch.object(PTYSession, "_restore_terminal")
    @patch.object(PTYSession, "_setup_terminal")
    @patch("pty.fork", return_value=(42, 7))
    def test_already_reaped_returns_zero(
        self, mock_fork, mock_setup, mock_restore, mock_winsize, mock_sigwinch, mock_io_loop, mock_close, mock_waitpid
    ):
        """If child already reaped (ChildProcessError), return 0."""
        session = PTYSession()
        with patch.object(sys.stdin, "fileno", return_value=0):
            exit_code = session.spawn(["echo", "hi"])
        assert exit_code == 0

    @patch("os.waitpid", return_value=(42, 9))
    @patch("os.close")
    @patch.object(PTYSession, "_io_loop")
    @patch.object(PTYSession, "_install_sigwinch")
    @patch.object(PTYSession, "_copy_winsize")
    @patch.object(PTYSession, "_restore_terminal")
    @patch.object(PTYSession, "_setup_terminal")
    @patch("pty.fork", return_value=(42, 7))
    def test_signaled_child_returns_one(
        self, mock_fork, mock_setup, mock_restore, mock_winsize, mock_sigwinch, mock_io_loop, mock_close, mock_waitpid
    ):
        """If child was killed by signal (not WIFEXITED), return 1."""
        # status=9 means killed by signal 9 (SIGKILL), WIFEXITED=False
        session = PTYSession()
        with patch.object(sys.stdin, "fileno", return_value=0):
            exit_code = session.spawn(["sleep", "100"])
        assert exit_code == 1

    @patch("os.waitpid", return_value=(42, 0))
    @patch("os.close")
    @patch.object(PTYSession, "_io_loop", side_effect=RuntimeError("io boom"))
    @patch.object(PTYSession, "_install_sigwinch")
    @patch.object(PTYSession, "_copy_winsize")
    @patch.object(PTYSession, "_restore_terminal")
    @patch.object(PTYSession, "_setup_terminal")
    @patch("pty.fork", return_value=(42, 7))
    def test_terminal_restored_on_io_loop_error(
        self, mock_fork, mock_setup, mock_restore, mock_winsize, mock_sigwinch, mock_io_loop, mock_close, mock_waitpid
    ):
        """Terminal should be restored even if _io_loop raises."""
        session = PTYSession()
        with pytest.raises(RuntimeError, match="io boom"):
            with patch.object(sys.stdin, "fileno", return_value=0):
                session.spawn(["echo", "hi"])

        mock_restore.assert_called_once()
        mock_close.assert_called_once_with(7)

    @patch("os.waitpid", return_value=(42, 0))
    @patch("os.close")
    @patch.object(PTYSession, "_io_loop")
    @patch.object(PTYSession, "_install_sigwinch")
    @patch.object(PTYSession, "_copy_winsize")
    @patch.object(PTYSession, "_restore_terminal")
    @patch.object(PTYSession, "_setup_terminal")
    @patch("pty.fork", return_value=(42, 7))
    def test_env_not_applied_in_parent(
        self, mock_fork, mock_setup, mock_restore, mock_winsize, mock_sigwinch, mock_io_loop, mock_close, mock_waitpid
    ):
        """The env dict should only be applied in the child (pid==0) branch."""
        session = PTYSession()
        env = {"ANTHROPIC_BASE_URL": "http://localhost:8080"}

        original_environ = os.environ.copy()
        with patch.object(sys.stdin, "fileno", return_value=0):
            session.spawn(["echo", "hi"], env=env)

        # Parent env should not have been modified
        assert os.environ.get("ANTHROPIC_BASE_URL") == original_environ.get("ANTHROPIC_BASE_URL")


class TestIOLoopDrainOnChildExit:
    @patch("os.read")
    @patch("os.write")
    @patch("select.select")
    @patch("os.waitpid")
    def test_drains_remaining_output_after_child_exits(self, mock_waitpid, mock_select, mock_write, mock_read):
        """When child exits during timeout check, remaining PTY output should be drained."""
        session = PTYSession(pid=42, master_fd=5)
        stdout_fd = sys.stdout.fileno()
        cb = MagicMock()
        session.on_output(cb)

        # select returns empty (timeout) -> child exited -> drain remaining
        mock_select.return_value = ([], [], [])
        mock_waitpid.return_value = (42, 0)  # child exited
        mock_read.side_effect = [
            b"remaining output",
            b"",  # EOF, stop draining
        ]

        with patch.object(sys.stdin, "isatty", return_value=False):
            session._io_loop()

        mock_write.assert_any_call(stdout_fd, b"remaining output")
        cb.assert_called_once_with(b"remaining output")

    @patch("os.read")
    @patch("os.write")
    @patch("select.select")
    @patch("os.waitpid")
    def test_drain_callback_exception_does_not_break_drain(self, mock_waitpid, mock_select, mock_write, mock_read):
        """Callback exceptions during drain should be silenced without stopping the drain."""
        session = PTYSession(pid=42, master_fd=5)
        bad_cb = MagicMock(side_effect=RuntimeError("drain callback failed"))
        good_cb = MagicMock()
        session.on_output(bad_cb)
        session.on_output(good_cb)

        mock_select.return_value = ([], [], [])
        mock_waitpid.return_value = (42, 0)
        mock_read.side_effect = [
            b"drain data",
            b"",  # EOF
        ]

        with patch.object(sys.stdin, "isatty", return_value=False):
            session._io_loop()

        bad_cb.assert_called_once_with(b"drain data")
        good_cb.assert_called_once_with(b"drain data")


class TestIOLoopInputCallbackExceptions:
    @patch("os.read")
    @patch("os.write")
    @patch("select.select")
    def test_input_callback_exception_does_not_break_loop(self, mock_select, mock_write, mock_read):
        """Input callback exceptions should be silenced, and other callbacks still invoked."""
        session = PTYSession(master_fd=5)
        bad_cb = MagicMock(side_effect=RuntimeError("input callback failed"))
        good_cb = MagicMock()
        session.on_input(bad_cb)
        session.on_input(good_cb)
        stdin_fd = 0

        mock_select.side_effect = [
            ([stdin_fd], [], []),
            ([5], [], []),
        ]
        mock_read.side_effect = [
            b"typed text",
            OSError(errno.EIO, "I/O error"),
        ]

        with patch.object(sys.stdin, "isatty", return_value=True):
            with patch.object(sys.stdin, "fileno", return_value=stdin_fd):
                session._io_loop()

        bad_cb.assert_called_once_with(b"typed text")
        good_cb.assert_called_once_with(b"typed text")


class TestChildProcessBranch:
    """Test the child process branch after pty.fork() (lines 55-57)."""

    @patch("os.execvp")
    @patch("pty.fork", return_value=(0, 7))
    def test_child_calls_execvp_with_argv(self, mock_fork, mock_execvp):
        """When pid=0 (child process), os.execvp should be called with the command."""
        session = PTYSession()
        # execvp would normally not return, but since it's mocked it does.
        # The code after execvp is unreachable, and we fall through to the
        # parent branch. Mock the parent-side methods to avoid errors.
        with (
            patch.object(session, "_setup_terminal"),
            patch.object(session, "_copy_winsize"),
            patch.object(session, "_install_sigwinch"),
            patch.object(session, "_io_loop"),
            patch.object(session, "_restore_terminal"),
            patch("os.close"),
            patch("os.waitpid", return_value=(0, 0)),
            patch.object(sys.stdin, "fileno", return_value=0),
        ):
            session.spawn(["claude", "--help"])

        mock_execvp.assert_called_once_with("claude", ["claude", "--help"])

    @patch("os.execvp")
    @patch("pty.fork", return_value=(0, 7))
    def test_child_updates_env_before_execvp(self, mock_fork, mock_execvp):
        """When pid=0 and env is provided, os.environ.update is called before execvp."""
        session = PTYSession()
        custom_env = {"ANTHROPIC_BASE_URL": "http://localhost:8080"}

        captured_environ = {}

        def capture_execvp(cmd, args):
            # Capture the current state of os.environ at the time execvp is called
            captured_environ.update(os.environ)

        mock_execvp.side_effect = capture_execvp

        with (
            patch.object(session, "_setup_terminal"),
            patch.object(session, "_copy_winsize"),
            patch.object(session, "_install_sigwinch"),
            patch.object(session, "_io_loop"),
            patch.object(session, "_restore_terminal"),
            patch("os.close"),
            patch("os.waitpid", return_value=(0, 0)),
            patch.object(sys.stdin, "fileno", return_value=0),
        ):
            session.spawn(["claude"], env=custom_env)

        # execvp should have been called
        mock_execvp.assert_called_once_with("claude", ["claude"])
        # env should have been updated before execvp
        assert captured_environ.get("ANTHROPIC_BASE_URL") == "http://localhost:8080"

        # Clean up the env we just polluted
        os.environ.pop("ANTHROPIC_BASE_URL", None)

    @patch("os.execvp")
    @patch("pty.fork", return_value=(0, 7))
    def test_child_without_env_skips_update(self, mock_fork, mock_execvp):
        """When pid=0 and no env is provided, os.environ.update is not called."""
        session = PTYSession()

        with (
            patch.object(session, "_setup_terminal"),
            patch.object(session, "_copy_winsize"),
            patch.object(session, "_install_sigwinch"),
            patch.object(session, "_io_loop"),
            patch.object(session, "_restore_terminal"),
            patch("os.close"),
            patch("os.waitpid", return_value=(0, 0)),
            patch.object(sys.stdin, "fileno", return_value=0),
            patch.dict(os.environ, {}, clear=False),
        ):
            session.spawn(["claude"])

        mock_execvp.assert_called_once_with("claude", ["claude"])


class TestIOLoopSelectError:
    """Test select.error exception handling in the I/O loop (line 125)."""

    @patch("os.read")
    @patch("os.write")
    @patch("select.select")
    def test_select_error_breaks_loop(self, mock_select, mock_write, mock_read):
        """select.error should break the I/O loop cleanly."""
        session = PTYSession(master_fd=5)

        mock_select.side_effect = select.error(errno.EINTR, "Interrupted")

        with patch.object(sys.stdin, "isatty", return_value=False):
            session._io_loop()

        assert mock_select.call_count == 1
        mock_read.assert_not_called()


class TestIOLoopNonEIOError:
    """Test that non-EIO OSError on master read is re-raised (line 134)."""

    @patch("os.read")
    @patch("os.write")
    @patch("select.select")
    def test_non_eio_oserror_is_reraised(self, mock_select, mock_write, mock_read):
        """OSError with errno != EIO should propagate out of the I/O loop."""
        session = PTYSession(master_fd=5)

        mock_select.return_value = ([5], [], [])
        mock_read.side_effect = OSError(errno.EBADF, "Bad file descriptor")

        with patch.object(sys.stdin, "isatty", return_value=False):
            with pytest.raises(OSError) as exc_info:
                session._io_loop()

        assert exc_info.value.errno == errno.EBADF

    @patch("os.read")
    @patch("os.write")
    @patch("select.select")
    def test_eperm_oserror_is_reraised(self, mock_select, mock_write, mock_read):
        """OSError with errno EPERM should propagate (not silently caught like EIO)."""
        session = PTYSession(master_fd=5)

        mock_select.return_value = ([5], [], [])
        mock_read.side_effect = OSError(errno.EPERM, "Operation not permitted")

        with patch.object(sys.stdin, "isatty", return_value=False):
            with pytest.raises(OSError) as exc_info:
                session._io_loop()

        assert exc_info.value.errno == errno.EPERM
