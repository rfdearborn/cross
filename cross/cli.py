"""CLI entry point — routes to subcommands (proxy, wrap)."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys

from cross.session import registry


def main():
    parser = argparse.ArgumentParser(prog="cross", description="Agent monitoring proxy and session manager")
    sub = parser.add_subparsers(dest="command")

    # cross proxy — start the network proxy
    sub.add_parser("proxy", help="Start the network monitoring proxy")

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

    if args.command == "proxy":
        _run_proxy()
    elif args.command == "wrap":
        argv = _parse_agent_argv(args.agent_argv)
        if not argv:
            wrap_p.error("Usage: cross wrap -- <agent-command> [args...]")
        sys.exit(_run_wrap(argv, log))
    else:
        parser.print_help()


def _run_proxy():
    """Start the network monitoring proxy."""
    import uvicorn
    from cross.config import settings

    uvicorn.run(
        "cross.proxy:app",
        host=settings.listen_host,
        port=settings.listen_port,
        log_level="warning",
    )


def _run_wrap(argv: list[str], log: logging.Logger) -> int:
    """Wrap an agent command in a PTY-managed session."""
    # Resolve the agent binary
    agent_bin = shutil.which(argv[0])
    if not agent_bin:
        log.error(f"Agent not found: {argv[0]}")
        return 127

    agent_name = os.path.basename(argv[0])
    argv[0] = agent_bin

    # Set up env to route API traffic through Cross proxy
    from cross.config import settings
    env = {
        "ANTHROPIC_BASE_URL": f"http://localhost:{settings.listen_port}",
    }

    # Create session
    info = registry.create(agent=agent_name, argv=argv)
    log.info(f"Session {info.session_id} started: {agent_name} in {info.project} ({info.cwd})")

    # Register I/O callbacks
    info.pty_session.on_output(lambda data: _on_output(info.session_id, data))
    info.pty_session.on_input(lambda data: _on_input(info.session_id, data))

    # Spawn and block until agent exits
    exit_code = info.pty_session.spawn(argv, env=env)

    registry.complete(info.session_id, exit_code)
    log.info(f"Session {info.session_id} ended: exit code {exit_code}")
    return exit_code


def _on_output(session_id: str, data: bytes):
    """Called with each chunk of agent output. Placeholder for Slack relay."""
    pass


def _on_input(session_id: str, data: bytes):
    """Called with each chunk of user input. Placeholder for Slack relay."""
    pass


def _parse_agent_argv(remainder: list[str]) -> list[str]:
    """Strip leading '--' from argparse REMAINDER."""
    if remainder and remainder[0] == "--":
        return remainder[1:]
    return remainder
