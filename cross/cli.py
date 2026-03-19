"""CLI entry point — routes to subcommands (daemon, wrap)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import shutil
import sys
import threading

import httpx
import websockets.sync.client

from cross.ansi import strip_ansi
from cross.config import settings
from cross.session import registry


def main():
    parser = argparse.ArgumentParser(prog="cross", description="Agent monitoring proxy and session manager")
    sub = parser.add_subparsers(dest="command")

    # cross start / cross daemon — start the central daemon (proxy + slack + session mgmt)
    daemon_p = sub.add_parser("daemon", help="Start the cross daemon (proxy + Slack + session management)")
    daemon_p.add_argument("--foreground", "-f", action="store_true", help="Run in foreground (default: background)")
    start_p = sub.add_parser("start", help="Start the cross daemon (alias for daemon)")
    start_p.add_argument("--foreground", "-f", action="store_true", help="Run in foreground (default: background)")

    # cross proxy — start just the network proxy (no Slack, no session mgmt)
    sub.add_parser("proxy", help="Start only the network monitoring proxy")

    # cross wrap -- <agent> [args...]
    wrap_p = sub.add_parser("wrap", help="Wrap an agent CLI in a cross-managed PTY")
    wrap_p.add_argument("agent_argv", nargs=argparse.REMAINDER, help="Agent command (after --)")

    # cross setup — interactive onboarding wizard
    sub.add_parser("setup", help="Interactive setup wizard")

    # cross reset — wipe configuration and start fresh
    sub.add_parser("reset", help="Remove cross configuration (~/.cross/.env and rules)")

    # cross stop — stop the running daemon
    sub.add_parser("stop", help="Stop the running cross daemon")

    # cross restart — restart the daemon
    sub.add_parser("restart", help="Restart the cross daemon")

    # cross update — self-update cross to the latest version
    update_p = sub.add_parser("update", help="Update cross to the latest version")
    update_p.add_argument(
        "--path", metavar="PATH", nargs="?", const=".", help="Install from local source (default: current directory)"
    )
    update_p.add_argument("--head", action="store_true", help="Install from main branch on GitHub")

    # cross pending [approve|deny <tool_use_id>]
    pending_p = sub.add_parser("pending", help="Show or resolve pending gate escalations")
    pending_sub = pending_p.add_subparsers(dest="pending_action")
    approve_p = pending_sub.add_parser("approve", help="Approve a pending escalation")
    approve_p.add_argument("tool_use_id", help="Tool use ID to approve")
    deny_p = pending_sub.add_parser("deny", help="Deny a pending escalation")
    deny_p.add_argument("tool_use_id", help="Tool use ID to deny")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("cross")

    if args.command == "setup":
        from cross.setup import run_setup

        run_setup()
        return
    elif args.command == "reset":
        sys.exit(_run_reset())
    elif args.command in ("daemon", "start"):
        if args.foreground:
            _run_daemon()
        else:
            _run_daemon_background()
    elif args.command == "stop":
        sys.exit(_run_stop())
    elif args.command == "restart":
        _run_stop(quiet=True)
        _run_daemon_background()
    elif args.command == "proxy":
        _run_proxy()
    elif args.command == "wrap":
        argv = _parse_agent_argv(args.agent_argv)
        if not argv:
            wrap_p.error("Usage: cross wrap -- <agent-command> [args...]")
        sys.exit(_run_wrap(argv, log))
    elif args.command == "update":
        sys.exit(_run_update(local_path=args.path, from_head=args.head))
    elif args.command == "pending":
        sys.exit(_run_pending(args))
    else:
        parser.print_help()


def _run_reset() -> int:
    """Remove cross configuration so the user can re-run setup."""
    from pathlib import Path

    cross_dir = Path.home() / ".cross"
    targets = [
        (cross_dir / ".env", "Configuration (.env)"),
        (cross_dir / "rules.d", "Rules (rules.d/)"),
    ]

    found = [(path, label) for path, label in targets if path.exists()]
    if not found:
        print("Nothing to remove — no configuration found.")
        return 0

    removed = []
    for path, label in found:
        answer = input(f"Remove {label}? (y/N): ").strip().lower()
        if answer in ("y", "yes"):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(str(path))
            print(f"  Removed: {path}")

    if removed:
        print("\nRun 'cross setup' to reconfigure.")
    else:
        print("Nothing removed.")
    return 0


_PYPI_PACKAGE = "cross-ai"
_GITHUB_REPO = "https://github.com/rfdearborn/cross"

_VERSION_CHECK = (
    "import importlib.metadata\n"
    "try:\n"
    "    print(importlib.metadata.version('cross-ai'))\n"
    "except importlib.metadata.PackageNotFoundError:\n"
    "    print(importlib.metadata.version('cross'))\n"
)


def _get_installed_version() -> str | None:
    """Get the installed version of cross, checking both package names."""
    import importlib.metadata

    for name in ("cross-ai", "cross"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _run_update(local_path: str | None = None, from_head: bool = False) -> int:
    """Update cross to the latest version via pip."""
    import subprocess

    if local_path and from_head:
        print("Cannot use --path and --head together.", file=sys.stderr)
        return 1

    old_version = _get_installed_version()

    print(f"Current version: {old_version or 'unknown'}")

    if local_path:
        local_path = os.path.abspath(local_path)
        print(f"Installing from local source: {local_path}")
        pip_args = [sys.executable, "-m", "pip", "install", "--upgrade", local_path]
    elif from_head:
        print(f"Installing from main branch: {_GITHUB_REPO}")
        pip_args = [sys.executable, "-m", "pip", "install", "--upgrade", f"git+{_GITHUB_REPO}@main"]
    else:
        print("Updating cross...")
        pip_args = [sys.executable, "-m", "pip", "install", "--upgrade", _PYPI_PACKAGE]

    # Use the same Python interpreter that's running this process
    result = subprocess.run(
        pip_args,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Update failed:\n{result.stderr.strip()}", file=sys.stderr)
        return 1

    # Re-check version after update (use subprocess to avoid stale cache)
    version_result = subprocess.run(
        [sys.executable, "-c", _VERSION_CHECK],
        capture_output=True,
        text=True,
    )
    new_version = version_result.stdout.strip() if version_result.returncode == 0 else None

    if new_version and new_version != old_version:
        print(f"Updated cross: {old_version} -> {new_version}")
    else:
        print(f"cross is already up to date ({new_version or old_version}).")
    return 0


def _run_pending(args) -> int:
    """Show or resolve pending gate escalations via the daemon API."""
    daemon_url = f"http://localhost:{settings.listen_port}"

    if getattr(args, "pending_action", None) in ("approve", "deny"):
        approved = args.pending_action == "approve"
        tool_use_id = args.tool_use_id
        try:
            resp = httpx.post(
                f"{daemon_url}/cross/api/pending/{tool_use_id}/resolve",
                json={"approved": approved, "username": "cli"},
                timeout=5,
            )
            if resp.status_code == 200:
                action_word = "Approved" if approved else "Denied"
                print(f"{action_word}: {tool_use_id}")
                return 0
            else:
                print(f"Error: {resp.status_code} {resp.text}", file=sys.stderr)
                return 1
        except httpx.ConnectError:
            print("Error: daemon not running", file=sys.stderr)
            return 1

    # Default: list pending escalations
    try:
        resp = httpx.get(f"{daemon_url}/cross/api/pending", timeout=5)
    except httpx.ConnectError:
        print("Error: daemon not running", file=sys.stderr)
        return 1

    if resp.status_code != 200:
        print(f"Error: {resp.status_code}", file=sys.stderr)
        return 1

    pending = resp.json()
    if not pending:
        print("No pending escalations.")
        return 0

    for item in pending:
        tool_id = item.get("tool_use_id", "?")
        tool_name = item.get("tool_name", "?")
        reason = item.get("reason", "")
        tool_input = item.get("tool_input")
        input_preview = ""
        if tool_input:
            input_str = json.dumps(tool_input)
            input_preview = input_str[:80] + "..." if len(input_str) > 80 else input_str
        print(f"  {tool_id}")
        print(f"    Tool:   {tool_name}")
        if reason:
            print(f"    Reason: {reason}")
        if input_preview:
            print(f"    Input:  {input_preview}")
        print()

    print(f"{len(pending)} pending escalation(s).")
    print("  cross pending approve <tool_use_id>")
    print("  cross pending deny <tool_use_id>")
    return 0


_PID_FILE = os.path.join(os.path.expanduser(settings.config_dir), "daemon.pid")


def _write_pid():
    """Write the current process PID to the PID file."""
    os.makedirs(os.path.dirname(_PID_FILE), exist_ok=True)
    with open(_PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def _read_pid() -> int | None:
    """Read the daemon PID from the PID file, or None if not found."""
    try:
        with open(_PID_FILE) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None


def _remove_pid():
    """Remove the PID file."""
    try:
        os.remove(_PID_FILE)
    except FileNotFoundError:
        pass


def _find_pid_by_port(port: int) -> int | None:
    """Find the PID of a process listening on the given port."""
    import subprocess

    try:
        result = subprocess.run(
            ["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip().splitlines()[0])
    except (OSError, ValueError):
        pass
    return None


def _run_stop(quiet: bool = False) -> int:
    """Stop the running daemon by sending SIGTERM."""
    import signal

    pid = _read_pid()
    if pid is None:
        # Fall back to finding by port
        pid = _find_pid_by_port(settings.listen_port)
        if pid is None:
            if not quiet:
                print("No running daemon found.")
            return 1

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        if not quiet:
            print(f"Daemon (pid {pid}) is not running.")
        _remove_pid()
        return 1

    if not quiet:
        print(f"Stopped daemon (pid {pid}).")
    _remove_pid()
    return 0


def _run_daemon():
    """Start the central daemon."""
    import uvicorn

    _write_pid()
    try:
        uvicorn.run(
            "cross.daemon:app",
            host=settings.listen_host,
            port=settings.listen_port,
            log_level="warning",
        )
    finally:
        _remove_pid()


def _run_daemon_background():
    """Start the daemon as a background process."""
    import subprocess
    import time

    proc = subprocess.Popen(
        [sys.executable, "-m", "cross", "daemon", "--foreground"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    # Brief wait to check it didn't crash immediately
    time.sleep(0.5)
    if proc.poll() is not None:
        print("Daemon failed to start.", file=sys.stderr)
        sys.exit(1)
    print(f"Daemon started (pid {proc.pid}) on port {settings.listen_port}.")


def _run_proxy():
    """Start just the network proxy (standalone, no Slack)."""
    import uvicorn
    from starlette.applications import Starlette
    from starlette.routing import Route

    from cross.events import EventBus
    from cross.plugins.logger import LoggerPlugin

    bus = EventBus()

    async def proxy_handler(request):
        from cross.proxy import handle_proxy_request

        return await handle_proxy_request(request, bus)

    async def on_startup():
        plugin = LoggerPlugin()
        bus.subscribe(plugin.handle)

    app = Starlette(
        routes=[Route("/{path:path}", proxy_handler, methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])],
        on_startup=[on_startup],
    )

    uvicorn.run(
        app,
        host=settings.listen_host,
        port=settings.listen_port,
        log_level="warning",
    )


def _run_wrap(argv: list[str], log: logging.Logger) -> int:
    """Wrap an agent command in a PTY, registering with the daemon."""
    from pathlib import Path

    # Resolve the agent binary
    agent_bin = shutil.which(argv[0])
    if not agent_bin:
        log.error(f"Agent not found: {argv[0]}")
        return 127

    agent_name = os.path.basename(argv[0])
    argv[0] = agent_bin

    # Set up env to route API traffic through the daemon's proxy
    env = {
        "ANTHROPIC_BASE_URL": f"http://localhost:{settings.listen_port}",
    }

    # OpenClaw: inject tool hook via --import (ESM)
    if agent_name == "openclaw":
        hook_path = Path(__file__).parent / "patches" / "openclaw_hook.mjs"
        if hook_path.exists():
            existing_node_opts = os.environ.get("NODE_OPTIONS", "")
            env["NODE_OPTIONS"] = f"--import {hook_path} {existing_node_opts}".strip()
            env["CROSS_LISTEN_PORT"] = str(settings.listen_port)

    # Create local session
    info = registry.create(agent=agent_name, argv=argv)
    log.info(f"Session {info.session_id} started: {agent_name} in {info.project} ({info.cwd})")

    # Set session ID for OpenClaw hook (must be after session creation)
    if agent_name == "openclaw":
        env["CROSS_SESSION_ID"] = info.session_id

    # Register session with daemon
    daemon_url = f"http://localhost:{settings.listen_port}"
    _register_session(daemon_url, info, log)

    # Queue for sending PTY output to daemon
    output_queue: queue.Queue[dict] = queue.Queue()

    # Connect WebSocket for bidirectional I/O relay
    _start_ws_relay(daemon_url, info, output_queue, log)

    # Register PTY I/O callbacks
    info.pty_session.on_output(lambda data: _on_pty_output(data, output_queue))

    # Spawn and block until agent exits
    exit_code = info.pty_session.spawn(argv, env=env)

    registry.complete(info.session_id, exit_code)
    log.info(f"Session {info.session_id} ended: exit code {exit_code}")

    # Notify daemon of session end
    _end_session(daemon_url, info, log)

    return exit_code


def _register_session(daemon_url: str, info, log: logging.Logger):
    """Register session with the daemon via HTTP."""
    try:
        resp = httpx.post(
            f"{daemon_url}/cross/sessions",
            json={
                "session_id": info.session_id,
                "agent": info.agent,
                "project": info.project,
                "cwd": info.cwd,
                "started_at": info.started_at,
            },
            timeout=5,
        )
        if resp.status_code == 200:
            log.info("Registered with daemon")
        else:
            log.warning(f"Daemon registration failed: {resp.status_code}")
    except httpx.ConnectError:
        log.warning("Daemon not running — session will not appear in Slack")


def _end_session(daemon_url: str, info, log: logging.Logger):
    """Notify daemon that session has ended."""
    try:
        httpx.post(
            f"{daemon_url}/cross/sessions/{info.session_id}/end",
            json={
                "session_id": info.session_id,
                "exit_code": info.exit_code,
                "started_at": info.started_at,
                "ended_at": info.ended_at,
            },
            timeout=5,
        )
    except Exception:
        pass


def _on_pty_output(data: bytes, output_queue: queue.Queue):
    """Called with each chunk of PTY output. Cleans and queues for daemon."""
    text = strip_ansi(data)
    # Skip empty or whitespace-only output
    text = text.strip()
    if not text:
        return
    output_queue.put({"type": "pty_output", "text": text})


def _start_ws_relay(daemon_url: str, info, output_queue: queue.Queue, log: logging.Logger) -> threading.Thread | None:
    """Connect WebSocket to daemon for bidirectional I/O relay."""
    ws_url = daemon_url.replace("http://", "ws://") + f"/cross/sessions/{info.session_id}/io"

    def relay():
        try:
            with websockets.sync.client.connect(ws_url) as ws:
                while True:
                    # Send queued PTY output to daemon
                    try:
                        while True:
                            msg = output_queue.get_nowait()
                            ws.send(json.dumps(msg))
                    except queue.Empty:
                        pass

                    # Receive inject messages from daemon (Slack -> PTY)
                    try:
                        msg = ws.recv(timeout=0.2)
                        data = json.loads(msg)
                        if data.get("type") == "inject" and info.pty_session:
                            info.pty_session.inject_input(data["text"].encode())
                            log.info(f"Injected from Slack: {data['text'][:50]}")
                    except TimeoutError:
                        continue
                    except Exception:
                        break
        except Exception as e:
            log.warning(f"WebSocket relay failed: {e}")

    thread = threading.Thread(target=relay, daemon=True)
    thread.start()
    return thread


def _parse_agent_argv(remainder: list[str]) -> list[str]:
    """Strip leading '--' from argparse REMAINDER."""
    if remainder and remainder[0] == "--":
        return remainder[1:]
    return remainder
