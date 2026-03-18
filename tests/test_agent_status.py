"""Tests for agent detection and monitoring status."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cross.daemon import (
    _detect_hooked_agents,
    _detect_running_agents,
    _gate_agents,
    _sessions,
    get_agent_status,
)


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset daemon state between tests."""
    _sessions.clear()
    _gate_agents.clear()
    yield
    _sessions.clear()
    _gate_agents.clear()


class TestDetectHookedAgents:
    def test_openclaw_hook_detected(self, tmp_path):
        from pathlib import Path

        with patch.object(Path, "home", return_value=tmp_path):
            launch_dir = tmp_path / "Library" / "LaunchAgents"
            launch_dir.mkdir(parents=True)
            plist_path = launch_dir / "ai.openclaw.gateway.plist"
            plist_path.write_text("<string>--import /path/to/openclaw_hook.mjs</string>")
            result = _detect_hooked_agents()
        assert "openclaw" in result

    def test_no_plist_returns_empty(self, tmp_path):
        from pathlib import Path

        with patch.object(Path, "home", return_value=tmp_path):
            result = _detect_hooked_agents()
        assert result == set()

    def test_plist_without_hook_returns_empty(self, tmp_path):
        from pathlib import Path

        with patch.object(Path, "home", return_value=tmp_path):
            launch_dir = tmp_path / "Library" / "LaunchAgents"
            launch_dir.mkdir(parents=True)
            plist_path = launch_dir / "ai.openclaw.gateway.plist"
            plist_path.write_text("<string>some other config</string>")
            result = _detect_hooked_agents()
        assert result == set()


class TestDetectRunningAgents:
    @patch("subprocess.run")
    def test_detects_claude(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="1234\n")
        result = _detect_running_agents()
        assert "claude" in result
        assert 1234 in result["claude"]

    @patch("subprocess.run")
    def test_no_agents_running(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        result = _detect_running_agents()
        assert result == {}

    @patch("subprocess.run")
    def test_filters_own_pid(self, mock_run):
        import os

        own_pid = str(os.getpid())
        mock_run.return_value = MagicMock(returncode=0, stdout=f"{own_pid}\n")
        result = _detect_running_agents()
        # Own PID should be filtered out
        assert "claude" not in result or own_pid not in [str(p) for p in result.get("claude", [])]


class TestGetAgentStatus:
    @patch("cross.daemon._detect_running_agents", return_value={"claude": [1234]})
    def test_session_shows_as_monitored(self, mock_detect):
        _sessions["s1"] = {"agent": "claude", "project": "cross", "pid": 1234}
        status = get_agent_status()
        assert status["monitored_count"] == 1
        assert status["monitored"][0]["label"] == "claude - cross"
        assert status["unmonitored_count"] == 0

    @patch("cross.daemon._detect_running_agents", return_value={"claude": [1234]})
    def test_unmonitored_agent_detected(self, mock_detect):
        status = get_agent_status()
        assert status["unmonitored_count"] == 1
        assert status["unmonitored"][0]["agent"] == "claude"

    @patch("cross.daemon._detect_running_agents", return_value={"openclaw": [5678]})
    def test_gate_agent_shows_as_monitored(self, mock_detect):
        _gate_agents.add("openclaw")
        status = get_agent_status()
        assert status["monitored_count"] == 1
        assert status["monitored"][0]["agent"] == "openclaw"
        assert status["unmonitored_count"] == 0

    @patch("cross.daemon._detect_running_agents", return_value={})
    def test_stale_session_cleaned_up(self, mock_detect):
        _sessions["s1"] = {"agent": "claude", "project": "old", "pid": 9999}
        status = get_agent_status()
        assert status["monitored_count"] == 0
        assert "s1" not in _sessions

    @patch("cross.daemon._detect_running_agents", return_value={})
    def test_stale_gate_agent_cleaned_up(self, mock_detect):
        _gate_agents.add("openclaw")
        status = get_agent_status()
        assert "openclaw" not in _gate_agents
        assert status["monitored_count"] == 0

    @patch("cross.daemon._detect_running_agents", return_value={})
    def test_no_agents(self, mock_detect):
        status = get_agent_status()
        assert status["monitored_count"] == 0
        assert status["unmonitored_count"] == 0

    @patch(
        "cross.daemon._detect_running_agents",
        return_value={"claude": [1234], "openclaw": [5678]},
    )
    def test_mixed_monitored_and_unmonitored(self, mock_detect):
        _sessions["s1"] = {"agent": "claude", "project": "cross", "pid": 1234}
        status = get_agent_status()
        assert status["monitored_count"] == 1
        assert status["unmonitored_count"] == 1
        assert status["unmonitored"][0]["agent"] == "openclaw"
