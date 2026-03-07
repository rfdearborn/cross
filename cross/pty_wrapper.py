"""PTY wrapper — spawns an agent process in a pseudo-terminal for full I/O control."""

from __future__ import annotations

import errno
import fcntl
import os
import pty
import select
import signal
import struct
import sys
import termios
import tty
from dataclasses import dataclass, field
from typing import Callable

# Callbacks receive raw bytes from the PTY
IOCallback = Callable[[bytes], None]


@dataclass
class PTYSession:
    """Manages a single agent process running inside a PTY."""

    pid: int = 0
    master_fd: int = -1
    _original_termios: list | None = field(default=None, repr=False)
    _output_callbacks: list[IOCallback] = field(default_factory=list, repr=False)
    _input_callbacks: list[IOCallback] = field(default_factory=list, repr=False)

    def on_output(self, cb: IOCallback):
        """Register a callback for agent output (stdout)."""
        self._output_callbacks.append(cb)

    def on_input(self, cb: IOCallback):
        """Register a callback for user input (stdin)."""
        self._input_callbacks.append(cb)

    def inject_input(self, data: bytes):
        """Inject input into the PTY as if the user typed it."""
        if self.master_fd >= 0:
            os.write(self.master_fd, data)

    def spawn(self, argv: list[str], env: dict[str, str] | None = None):
        """Spawn the agent process in a PTY and run the I/O loop.

        This blocks until the child process exits. The caller's terminal
        is put into raw mode so keystrokes pass through immediately.
        """
        # Create PTY pair
        self.pid, self.master_fd = pty.fork()

        if self.pid == 0:
            # Child — exec the agent
            if env:
                os.environ.update(env)
            os.execvp(argv[0], argv)
            # execvp doesn't return

        # Parent — set up I/O multiplexing
        try:
            self._setup_terminal()
            self._copy_winsize(sys.stdin.fileno(), self.master_fd)
            self._install_sigwinch()
            self._io_loop()
        finally:
            self._restore_terminal()
            os.close(self.master_fd)
            self.master_fd = -1

        # Reap child and return exit code
        try:
            _, status = os.waitpid(self.pid, 0)
            if os.WIFEXITED(status):
                return os.WEXITSTATUS(status)
            return 1
        except ChildProcessError:
            return 0  # Already reaped in io_loop

    def _setup_terminal(self):
        """Put stdin into raw mode, saving original settings."""
        if sys.stdin.isatty():
            self._original_termios = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin)

    def _restore_terminal(self):
        """Restore original terminal settings."""
        if self._original_termios is not None:
            termios.tcsetattr(sys.stdin, termios.TCSAFLUSH, self._original_termios)
            self._original_termios = None

    def _copy_winsize(self, src_fd: int, dst_fd: int):
        """Copy terminal window size from src to dst."""
        try:
            winsize = fcntl.ioctl(src_fd, termios.TIOCGWINSZ, b"\x00" * 8)
            fcntl.ioctl(dst_fd, termios.TIOCSWINSZ, winsize)
        except OSError:
            pass

    def _install_sigwinch(self):
        """Forward terminal resize events to the PTY."""
        def handle_winsize(signum, frame):
            self._copy_winsize(sys.stdin.fileno(), self.master_fd)
            # Also send SIGWINCH to the child process group
            try:
                os.kill(self.pid, signal.SIGWINCH)
            except OSError:
                pass

        signal.signal(signal.SIGWINCH, handle_winsize)

    def _io_loop(self):
        """Multiplex I/O between stdin/stdout and the PTY master."""
        stdout_fd = sys.stdout.fileno()
        master = self.master_fd

        # Only read from stdin if it's a real terminal
        stdin_fd = sys.stdin.fileno() if sys.stdin.isatty() else -1
        watch_fds = [fd for fd in [stdin_fd, master] if fd >= 0]

        while True:
            try:
                rfds, _, _ = select.select(watch_fds, [], [], 0.1)
            except (select.error, ValueError):
                break

            if master in rfds:
                try:
                    data = os.read(master, 16384)
                except OSError as e:
                    if e.errno == errno.EIO:
                        break  # Child exited
                    raise
                if not data:
                    break
                # Write to user's terminal
                os.write(stdout_fd, data)
                # Notify output callbacks
                for cb in self._output_callbacks:
                    try:
                        cb(data)
                    except Exception:
                        pass

            if stdin_fd >= 0 and stdin_fd in rfds:
                data = os.read(stdin_fd, 16384)
                if not data:
                    # stdin closed — stop reading it but keep the loop
                    # alive for agent output
                    watch_fds.remove(stdin_fd)
                    stdin_fd = -1
                    continue
                # Write to agent's PTY
                os.write(master, data)
                # Notify input callbacks
                for cb in self._input_callbacks:
                    try:
                        cb(data)
                    except Exception:
                        pass

            if not rfds:
                # Timeout — check if child is still alive
                try:
                    pid, status = os.waitpid(self.pid, os.WNOHANG)
                    if pid != 0:
                        # Drain any remaining output
                        try:
                            while True:
                                data = os.read(master, 16384)
                                if not data:
                                    break
                                os.write(stdout_fd, data)
                                for cb in self._output_callbacks:
                                    try:
                                        cb(data)
                                    except Exception:
                                        pass
                        except OSError:
                            pass
                        break
                except ChildProcessError:
                    break
