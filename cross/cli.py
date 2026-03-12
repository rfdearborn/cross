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

    # cross daemon — start the central daemon (proxy + slack + session mgmt)
    sub.add_parser("daemon", help="Start the Cross daemon (proxy + Slack + session management)")

    # cross proxy — start just the network proxy (no Slack, no session mgmt)
    sub.add_parser("proxy", help="Start only the network monitoring proxy")

    # cross wrap -- <agent> [args...]
    wrap_p = sub.add_parser("wrap", help="Wrap an agent CLI in a Cross-managed PTY")
    wrap_p.add_argument("agent_argv", nargs=argparse.REMAINDER, help="Agent command (after --)")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("cross")

    if args.command == "daemon":
        _run_daemon()
    elif args.command == "proxy":
        _run_proxy()
    elif args.command == "wrap":
        argv = _parse_agent_argv(args.agent_argv)
        if not argv:
            wrap_p.error("Usage: cross wrap -- <agent-command> [args...]")
        sys.exit(_run_wrap(argv, log))
    else:
        parser.print_help()


def _run_daemon():
    """Start the central daemon."""
    import uvicorn

    uvicorn.run(
        "cross.daemon:app",
        host=settings.listen_host,
        port=settings.listen_port,
        log_level="warning",
    )


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

    # Create local session
    info = registry.create(agent=agent_name, argv=argv)
    log.info(f"Session {info.session_id} started: {agent_name} in {info.project} ({info.cwd})")

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
