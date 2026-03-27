"""Shared fixtures for e2e tests.

Spins up:
- Mock upstream LLM server (Anthropic or OpenAI format)
- Mock gate LLM server (for denylist + LLM review gate)
- The real Cross daemon (Starlette app) on an ephemeral port
- httpx clients for making requests through the proxy

All network I/O stays in-process via ASGI transports.
"""

from __future__ import annotations

import asyncio
import logging
import textwrap
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest
import uvicorn

from tests.e2e.mock_servers import (
    MockAnthropicServer,
    MockGateLLMServer,
    MockOpenAIServer,
    MockSlackAPI,
    MockSMTP,
)

logger = logging.getLogger("tests.e2e")


# ---------------------------------------------------------------------------
# Port allocation
# ---------------------------------------------------------------------------

_next_port = 19200


def _alloc_port() -> int:
    global _next_port
    p = _next_port
    _next_port += 1
    return p


# ---------------------------------------------------------------------------
# Fixture: run a Starlette app on a real TCP port (background task)
# ---------------------------------------------------------------------------


async def _serve(app, host: str, port: int, started: asyncio.Event, shutdown: asyncio.Event):
    """Run a uvicorn server until *shutdown* is set."""
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)

    # Override install_signal_handlers to be a no-op in async context
    server.install_signal_handlers = lambda: None

    # Start serving in background
    loop = asyncio.get_running_loop()
    serve_task = loop.create_task(server.serve())
    # Wait until server is started
    while not server.started:
        await asyncio.sleep(0.05)
    started.set()
    # Wait for shutdown signal
    await shutdown.wait()
    server.should_exit = True
    await serve_task


@pytest.fixture()
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# Fixture: mock upstream Anthropic LLM
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_anthropic():
    """Provides a MockAnthropicServer instance (not yet serving)."""
    return MockAnthropicServer()


@pytest.fixture()
def mock_openai():
    """Provides a MockOpenAIServer instance (not yet serving)."""
    return MockOpenAIServer()


@pytest.fixture()
def mock_gate_llm():
    """Provides a MockGateLLMServer for the Cross LLM review gate."""
    return MockGateLLMServer()


@pytest.fixture()
def mock_slack():
    """Provides a MockSlackAPI instance."""
    return MockSlackAPI()


@pytest.fixture()
def mock_smtp():
    """Provides a MockSMTP instance that captures sent emails."""
    return MockSMTP()


# ---------------------------------------------------------------------------
# Fixture: start mock servers on real ports
# ---------------------------------------------------------------------------


@pytest.fixture()
async def anthropic_server(mock_anthropic):
    """Start the mock Anthropic API on a TCP port. Yields (server, base_url)."""
    port = _alloc_port()
    started, shutdown = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(_serve(mock_anthropic.build_app(), "127.0.0.1", port, started, shutdown))
    await asyncio.wait_for(started.wait(), timeout=5)
    yield mock_anthropic, f"http://127.0.0.1:{port}"
    shutdown.set()
    await task


@pytest.fixture()
async def openai_server(mock_openai):
    """Start the mock OpenAI API on a TCP port. Yields (server, base_url)."""
    port = _alloc_port()
    started, shutdown = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(_serve(mock_openai.build_app(), "127.0.0.1", port, started, shutdown))
    await asyncio.wait_for(started.wait(), timeout=5)
    yield mock_openai, f"http://127.0.0.1:{port}"
    shutdown.set()
    await task


@pytest.fixture()
async def gate_llm_server(mock_gate_llm):
    """Start the mock gate LLM API on a TCP port. Yields (server, base_url)."""
    port = _alloc_port()
    started, shutdown = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(_serve(mock_gate_llm.build_app(), "127.0.0.1", port, started, shutdown))
    await asyncio.wait_for(started.wait(), timeout=5)
    yield mock_gate_llm, f"http://127.0.0.1:{port}"
    shutdown.set()
    await task


@pytest.fixture()
async def slack_server(mock_slack):
    """Start the mock Slack API on a TCP port. Yields (server, base_url)."""
    port = _alloc_port()
    started, shutdown = asyncio.Event(), asyncio.Event()
    task = asyncio.create_task(_serve(mock_slack.build_app(), "127.0.0.1", port, started, shutdown))
    await asyncio.wait_for(started.wait(), timeout=5)
    yield mock_slack, f"http://127.0.0.1:{port}"
    shutdown.set()
    await task


# ---------------------------------------------------------------------------
# Fixture: Cross daemon running on a real port
# ---------------------------------------------------------------------------


@pytest.fixture()
async def cross_daemon(tmp_path, anthropic_server, gate_llm_server):
    """Start the real Cross daemon with mocked settings.

    The daemon proxies to the mock Anthropic upstream and uses the mock gate LLM.
    Returns (daemon_base_url, mock_anthropic, mock_gate_llm).
    """
    mock_anthropic, anthropic_url = anthropic_server
    mock_gate_llm, gate_url = gate_llm_server
    daemon_port = _alloc_port()

    # Write minimal denylist rules
    rules_dir = tmp_path / "rules.d"
    rules_dir.mkdir()
    rules_file = rules_dir / "test.yaml"
    rules_file.write_text(
        textwrap.dedent("""\
        rules:
          - name: block-rm-rf
            tools: ["bash"]
            field: command
            patterns: ["rm\\\\s+-rf\\\\s+/"]
            action: block
            description: "Block recursive delete of root"
          - name: flag-curl
            tools: ["bash"]
            field: command
            patterns: ["curl\\\\s+"]
            action: review
            description: "Flag curl commands for review"
    """)
    )

    log_file = str(tmp_path / "cross.log")
    events_file = str(tmp_path / "events.jsonl")

    settings_overrides = {
        "listen_host": "127.0.0.1",
        "listen_port": daemon_port,
        "anthropic_base_url": anthropic_url,
        "openai_base_url": "http://127.0.0.1:1",
        "chatgpt_base_url": "http://127.0.0.1:1",
        "log_file": log_file,
        "cli_strip_anthropic_api_key": True,
        "gating_enabled": True,
        "rules_dir": str(rules_dir),
        "llm_gate_enabled": True,
        "llm_gate_model": "anthropic/claude-sonnet-4-6",
        "llm_gate_api_key": "test-key",
        "llm_gate_base_url": gate_url,
        "llm_gate_temperature": 0.0,
        "llm_gate_max_tokens": 256,
        "llm_gate_reasoning": "",
        "llm_gate_timeout_ms": 10000,
        "llm_gate_context_tools": 3,
        "llm_gate_context_turns": 5,
        "llm_gate_context_chars_per_turn": 300,
        "llm_gate_context_intent_chars": 500,
        "llm_gate_justification": False,
        "llm_gate_shadow": False,
        "llm_gate_backup_model": "",
        "llm_gate_backup_api_key": "",
        "llm_gate_backup_base_url": "",
        "gate_approval_timeout": 5,
        "gate_max_retries": 1,
        "llm_sentinel_enabled": False,
        "llm_sentinel_model": "",
        "llm_sentinel_api_key": "",
        "llm_sentinel_base_url": "",
        "llm_sentinel_temperature": 0.0,
        "llm_sentinel_max_tokens": 1024,
        "llm_sentinel_reasoning": "",
        "llm_sentinel_interval_seconds": 60,
        "llm_sentinel_backup_model": "",
        "llm_sentinel_backup_api_key": "",
        "llm_sentinel_backup_base_url": "",
        "slack_bot_token": "",
        "slack_app_token": "",
        "slack_channel_base": "cross",
        "slack_channel_append_project": False,
        "slack_channel_append_user": False,
        "auto_update_enabled": False,
        "native_notifications_enabled": False,
        "config_dir": str(tmp_path / ".cross"),
        "custom_instructions_file": str(tmp_path / "instructions.md"),
        "email_from": "",
        "email_to": "",
        "email_smtp_host": "localhost",
        "email_smtp_port": 587,
        "email_smtp_ssl": False,
        "email_smtp_starttls": False,
        "email_smtp_username": "",
        "email_smtp_password": "",
        "email_imap_host": "",
        "email_imap_port": 993,
        "email_imap_ssl": True,
        "email_imap_username": "",
        "email_imap_password": "",
        "email_imap_poll_interval": 30,
    }

    # Build a mock settings object
    mock_settings = MagicMock()
    for k, v in settings_overrides.items():
        setattr(mock_settings, k, v)

    # Patch settings and event store path, then import and build the app fresh
    with (
        patch("cross.daemon.settings", mock_settings),
        patch("cross.config.settings", mock_settings),
        patch("cross.proxy.settings", mock_settings),
        patch("cross.plugins.logger.settings", mock_settings),
        patch("cross.event_store._default_path", lambda: events_file),
        patch("cross.daemon._detect_hooked_agents", return_value=set()),
        patch("cross.daemon._detect_running_agents", return_value={}),
        patch(
            "cross.state.load_state",
            return_value={
                "sessions": {},
                "project_cwds": {},
                "gate_agents": set(),
                "sentinel_events": [],
            },
        ),
        patch("cross.state.save_state"),
        patch("cross.daemon._persist_state"),
    ):
        import cross.daemon as daemon

        # Reset daemon globals
        daemon._gate_chain = None
        daemon._sentinel = None
        daemon._slack = None
        daemon._email = None
        daemon._dashboard = None
        daemon._sessions.clear()
        daemon._session_ws.clear()
        daemon._gate_agents.clear()
        daemon._gate_recent_tools.clear()
        daemon._permission_pending.clear()
        daemon._permission_notify_tasks.clear()

        # Reset proxy globals
        import cross.proxy as proxy_module

        proxy_module._client = None
        proxy_module._openai_client = None
        proxy_module._chatgpt_client = None
        proxy_module._blocked_tool_ids.clear()
        proxy_module._blocked_tool_info.clear()
        proxy_module._blocked_tool_timestamps.clear()
        proxy_module._recent_tools.clear()
        proxy_module._sentinel_halted = False
        proxy_module._sentinel_halt_reason = ""

        # Reset the event bus
        daemon.event_bus._handlers.clear()

        # Recreate app with fresh startup
        from starlette.applications import Starlette

        app = Starlette(
            routes=daemon._api_routes + daemon._dashboard_routes + daemon._ws_routes + daemon._proxy_routes,
            on_startup=[daemon.on_startup],
            on_shutdown=[daemon.on_shutdown],
        )

        started, shutdown_ev = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(_serve(app, "127.0.0.1", daemon_port, started, shutdown_ev))
        await asyncio.wait_for(started.wait(), timeout=10)

        base_url = f"http://127.0.0.1:{daemon_port}"
        yield {
            "base_url": base_url,
            "port": daemon_port,
            "mock_anthropic": mock_anthropic,
            "mock_gate_llm": mock_gate_llm,
            "settings": mock_settings,
            "log_file": log_file,
            "events_file": events_file,
            "tmp_path": tmp_path,
        }

        shutdown_ev.set()
        await task

        # Cleanup proxy clients
        if proxy_module._client:
            await proxy_module._client.aclose()
            proxy_module._client = None


# ---------------------------------------------------------------------------
# Fixture: Cross daemon with OpenAI upstream (for Codex tests)
# ---------------------------------------------------------------------------


@pytest.fixture()
async def cross_daemon_openai(tmp_path, openai_server, gate_llm_server):
    """Start Cross daemon configured with OpenAI upstream (for Codex/OpenAI agents)."""
    mock_openai, openai_url = openai_server
    mock_gate_llm, gate_url = gate_llm_server
    daemon_port = _alloc_port()

    rules_dir = tmp_path / "rules.d"
    rules_dir.mkdir()
    rules_file = rules_dir / "test.yaml"
    rules_file.write_text(
        textwrap.dedent("""\
        rules:
          - name: block-rm-rf
            tools: ["bash"]
            field: command
            patterns: ["rm\\\\s+-rf\\\\s+/"]
            action: block
            description: "Block recursive delete of root"
          - name: flag-curl
            tools: ["bash"]
            field: command
            patterns: ["curl\\\\s+"]
            action: review
            description: "Flag curl commands for review"
    """)
    )

    log_file = str(tmp_path / "cross.log")
    events_file = str(tmp_path / "events.jsonl")

    settings_overrides = {
        "listen_host": "127.0.0.1",
        "listen_port": daemon_port,
        "openai_base_url": openai_url,
        "anthropic_base_url": "http://127.0.0.1:1",
        "chatgpt_base_url": "http://127.0.0.1:1",
        "log_file": log_file,
        "cli_strip_anthropic_api_key": True,
        "gating_enabled": True,
        "rules_dir": str(rules_dir),
        "llm_gate_enabled": True,
        "llm_gate_model": "anthropic/claude-sonnet-4-6",
        "llm_gate_api_key": "test-key",
        "llm_gate_base_url": gate_url,
        "llm_gate_temperature": 0.0,
        "llm_gate_max_tokens": 256,
        "llm_gate_reasoning": "",
        "llm_gate_timeout_ms": 10000,
        "llm_gate_context_tools": 3,
        "llm_gate_context_turns": 5,
        "llm_gate_context_chars_per_turn": 300,
        "llm_gate_context_intent_chars": 500,
        "llm_gate_justification": False,
        "llm_gate_shadow": False,
        "llm_gate_backup_model": "",
        "llm_gate_backup_api_key": "",
        "llm_gate_backup_base_url": "",
        "gate_approval_timeout": 5,
        "gate_max_retries": 1,
        "llm_sentinel_enabled": False,
        "llm_sentinel_model": "",
        "llm_sentinel_api_key": "",
        "llm_sentinel_base_url": "",
        "llm_sentinel_temperature": 0.0,
        "llm_sentinel_max_tokens": 1024,
        "llm_sentinel_reasoning": "",
        "llm_sentinel_interval_seconds": 60,
        "llm_sentinel_backup_model": "",
        "llm_sentinel_backup_api_key": "",
        "llm_sentinel_backup_base_url": "",
        "slack_bot_token": "",
        "slack_app_token": "",
        "slack_channel_base": "cross",
        "slack_channel_append_project": False,
        "slack_channel_append_user": False,
        "auto_update_enabled": False,
        "native_notifications_enabled": False,
        "config_dir": str(tmp_path / ".cross"),
        "custom_instructions_file": str(tmp_path / "instructions.md"),
        "email_from": "",
        "email_to": "",
        "email_smtp_host": "localhost",
        "email_smtp_port": 587,
        "email_smtp_ssl": False,
        "email_smtp_starttls": False,
        "email_smtp_username": "",
        "email_smtp_password": "",
        "email_imap_host": "",
        "email_imap_port": 993,
        "email_imap_ssl": True,
        "email_imap_username": "",
        "email_imap_password": "",
        "email_imap_poll_interval": 30,
    }

    mock_settings = MagicMock()
    for k, v in settings_overrides.items():
        setattr(mock_settings, k, v)

    with (
        patch("cross.daemon.settings", mock_settings),
        patch("cross.config.settings", mock_settings),
        patch("cross.proxy.settings", mock_settings),
        patch("cross.plugins.logger.settings", mock_settings),
        patch("cross.event_store._default_path", lambda: events_file),
        patch("cross.daemon._detect_hooked_agents", return_value=set()),
        patch("cross.daemon._detect_running_agents", return_value={}),
        patch(
            "cross.state.load_state",
            return_value={
                "sessions": {},
                "project_cwds": {},
                "gate_agents": set(),
                "sentinel_events": [],
            },
        ),
        patch("cross.state.save_state"),
        patch("cross.daemon._persist_state"),
    ):
        import cross.daemon as daemon

        daemon._gate_chain = None
        daemon._sentinel = None
        daemon._slack = None
        daemon._email = None
        daemon._dashboard = None
        daemon._sessions.clear()
        daemon._session_ws.clear()
        daemon._gate_agents.clear()
        daemon._gate_recent_tools.clear()
        daemon._permission_pending.clear()
        daemon._permission_notify_tasks.clear()

        import cross.proxy as proxy_module

        proxy_module._client = None
        proxy_module._openai_client = None
        proxy_module._chatgpt_client = None
        proxy_module._blocked_tool_ids.clear()
        proxy_module._blocked_tool_info.clear()
        proxy_module._blocked_tool_timestamps.clear()
        proxy_module._recent_tools.clear()
        proxy_module._sentinel_halted = False
        proxy_module._sentinel_halt_reason = ""

        daemon.event_bus._handlers.clear()

        from starlette.applications import Starlette

        app = Starlette(
            routes=daemon._api_routes + daemon._dashboard_routes + daemon._ws_routes + daemon._proxy_routes,
            on_startup=[daemon.on_startup],
            on_shutdown=[daemon.on_shutdown],
        )

        started, shutdown_ev = asyncio.Event(), asyncio.Event()
        task = asyncio.create_task(_serve(app, "127.0.0.1", daemon_port, started, shutdown_ev))
        await asyncio.wait_for(started.wait(), timeout=10)

        base_url = f"http://127.0.0.1:{daemon_port}"
        yield {
            "base_url": base_url,
            "port": daemon_port,
            "mock_openai": mock_openai,
            "mock_gate_llm": mock_gate_llm,
            "settings": mock_settings,
            "log_file": log_file,
            "events_file": events_file,
            "tmp_path": tmp_path,
        }

        shutdown_ev.set()
        await task

        if proxy_module._client:
            await proxy_module._client.aclose()
            proxy_module._client = None


# ---------------------------------------------------------------------------
# Helper: send Anthropic Messages API request through Cross proxy
# ---------------------------------------------------------------------------


async def send_anthropic_message(
    base_url: str,
    *,
    content: str = "Hello",
    model: str = "claude-sonnet-4-6-20250514",
    stream: bool = True,
    tools: list[dict] | None = None,
) -> httpx.Response:
    """Send a Messages API request through the Cross proxy."""
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
    }
    if tools:
        body["tools"] = tools
    async with httpx.AsyncClient() as client:
        return await client.post(
            f"{base_url}/v1/messages",
            json=body,
            headers={
                "x-api-key": "test-api-key",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=30,
        )


async def send_openai_message(
    base_url: str,
    *,
    content: str = "Hello",
    model: str = "gpt-4o",
    stream: bool = True,
    tools: list[dict] | None = None,
) -> httpx.Response:
    """Send an OpenAI Chat Completions request through the Cross proxy."""
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "stream": stream,
    }
    if tools:
        body["tools"] = tools
    async with httpx.AsyncClient() as client:
        return await client.post(
            f"{base_url}/v1/chat/completions",
            json=body,
            headers={
                "authorization": "Bearer test-api-key",
                "content-type": "application/json",
            },
            timeout=30,
        )


async def collect_sse_events(response: httpx.Response) -> list[str]:
    """Collect all SSE data lines from a streaming response."""
    lines = []
    async for line in response.aiter_lines():
        lines.append(line)
    return lines


async def register_session(
    base_url: str,
    *,
    session_id: str = "test-session-1",
    agent: str = "claude",
    project: str = "test-project",
    cwd: str = "/tmp/test",
) -> httpx.Response:
    """Register a session with the Cross daemon."""
    async with httpx.AsyncClient() as client:
        return await client.post(
            f"{base_url}/cross/sessions",
            json={
                "session_id": session_id,
                "agent": agent,
                "project": project,
                "cwd": cwd,
            },
            timeout=5,
        )


async def end_session(
    base_url: str,
    session_id: str = "test-session-1",
    exit_code: int = 0,
) -> httpx.Response:
    """End a registered session."""
    async with httpx.AsyncClient() as client:
        return await client.post(
            f"{base_url}/cross/sessions/{session_id}/end",
            json={"session_id": session_id, "exit_code": exit_code},
            timeout=5,
        )


async def call_gate_api(
    base_url: str,
    *,
    tool_name: str = "bash",
    tool_input: dict | None = None,
    agent: str = "claude",
    session_id: str = "test-session-1",
) -> httpx.Response:
    """Call the external gate API (used by hook-based agents)."""
    async with httpx.AsyncClient() as client:
        return await client.post(
            f"{base_url}/cross/api/gate",
            json={
                "tool_name": tool_name,
                "tool_input": tool_input or {"command": "echo hello"},
                "agent": agent,
                "session_id": session_id,
            },
            timeout=30,
        )
