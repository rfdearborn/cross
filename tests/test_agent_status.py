"""Tests for agent detection and monitoring status."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from cross.daemon import (
    _detect_hooked_agents,
    _detect_running_agents,
    _gate_agents,
    _is_desktop_pid,
    _session_last_activity,
    _sessions,
    get_agent_status,
    record_session_activity,
)


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset daemon state between tests."""
    _sessions.clear()
    _gate_agents.clear()
    _session_last_activity.clear()
    yield
    _sessions.clear()
    _gate_agents.clear()
    _session_last_activity.clear()


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


class TestIsDesktopPid:
    @patch("cross.daemon.subprocess.run")
    def test_desktop_path_detected(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="/Users/me/Library/Application Support/Claude/claude-code/2.1.78/claude.app/Contents/MacOS/claude\n",
        )
        assert _is_desktop_pid(1234) is True

    @patch("cross.daemon.subprocess.run")
    def test_cli_path_not_desktop(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="/Users/me/.local/bin/claude\n")
        assert _is_desktop_pid(1234) is False

    @patch("cross.daemon.subprocess.run")
    def test_ps_failure_returns_false(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        assert _is_desktop_pid(1234) is False


class TestDetectRunningAgents:
    @patch("cross.daemon._is_desktop_pid", return_value=False)
    @patch("subprocess.run")
    def test_detects_claude_cli(self, mock_run, mock_desktop):
        mock_run.return_value = MagicMock(returncode=0, stdout="1234\n")
        result = _detect_running_agents()
        assert "claude" in result
        assert 1234 in result["claude"]

    @patch("cross.daemon._is_desktop_pid", return_value=True)
    @patch("subprocess.run")
    def test_detects_claude_desktop(self, mock_run, mock_desktop):
        mock_run.return_value = MagicMock(returncode=0, stdout="1234\n")
        result = _detect_running_agents()
        assert "claude" not in result
        assert "claude (desktop)" in result
        assert 1234 in result["claude (desktop)"]

    @patch("cross.daemon._is_desktop_pid")
    @patch("subprocess.run")
    def test_mixed_cli_and_desktop(self, mock_run, mock_desktop):
        mock_run.return_value = MagicMock(returncode=0, stdout="1234\n5678\n")
        mock_desktop.side_effect = lambda pid: pid == 5678
        result = _detect_running_agents()
        assert result["claude"] == [1234]
        assert result["claude (desktop)"] == [5678]

    @patch("subprocess.run")
    def test_no_agents_running(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        result = _detect_running_agents()
        assert result == {}

    @patch("cross.daemon._is_desktop_pid", return_value=False)
    @patch("subprocess.run")
    def test_filters_own_pid(self, mock_run, mock_desktop):
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

    @patch(
        "cross.daemon._detect_running_agents",
        return_value={"claude": [1234], "claude (desktop)": [5678]},
    )
    def test_desktop_session_shows_as_unmonitored(self, mock_detect):
        """CLI session monitored via wrap, Desktop session shown as unmonitored."""
        _sessions["s1"] = {"agent": "claude", "project": "cross", "pid": 1234}
        status = get_agent_status()
        assert status["monitored_count"] == 1
        assert status["unmonitored_count"] == 1
        assert status["unmonitored"][0]["agent"] == "claude (desktop)"

    @patch(
        "cross.daemon._detect_running_agents",
        return_value={"claude (desktop)": [5678]},
    )
    def test_desktop_only_shows_as_unmonitored(self, mock_detect):
        status = get_agent_status()
        assert status["monitored_count"] == 0
        assert status["unmonitored_count"] == 1
        assert status["unmonitored"][0]["agent"] == "claude (desktop)"

    @patch("cross.daemon._detect_running_agents", return_value={"claude": [1234]})
    def test_active_agent_has_active_true(self, mock_detect):
        """Agent with recent activity should be marked active."""
        _sessions["s1"] = {"agent": "claude", "project": "cross", "pid": 1234}
        record_session_activity("s1")
        status = get_agent_status()
        assert status["monitored"][0]["active"] is True

    @patch("cross.daemon._detect_running_agents", return_value={"claude": [1234]})
    def test_idle_agent_has_active_false(self, mock_detect):
        """Agent with no recent activity should be marked inactive."""
        _sessions["s1"] = {"agent": "claude", "project": "cross", "pid": 1234}
        # No activity recorded
        status = get_agent_status()
        assert status["monitored"][0]["active"] is False

    @patch("cross.daemon._detect_running_agents", return_value={"claude": [1234]})
    def test_stale_activity_has_active_false(self, mock_detect):
        """Agent with old activity should be marked inactive."""
        import time

        _sessions["s1"] = {"agent": "claude", "project": "cross", "pid": 1234}
        _session_last_activity["s1"] = time.time() - 60  # 60 seconds ago
        status = get_agent_status()
        assert status["monitored"][0]["active"] is False
