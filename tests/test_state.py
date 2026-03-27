"""Tests for cross.state — daemon state persistence."""

from __future__ import annotations

import json
import os
import time

import pytest

from cross.state import _SENTINEL_MAX_EVENTS, clear_state, load_state, save_state


@pytest.fixture
def state_path(tmp_path):
    return str(tmp_path / "state.json")


class TestSaveLoad:
    def test_round_trip(self, state_path):
        sessions = {"abc123": {"agent": "claude", "project": "myproj", "started_at": time.time()}}
        project_cwds = {"myproj": "/home/user/myproj"}
        gate_agents = {"claude", "openclaw"}

        save_state(
            sessions=sessions,
            project_cwds=project_cwds,
            gate_agents=gate_agents,
            path=state_path,
        )

        restored = load_state(path=state_path)
        assert restored["sessions"] == sessions
        assert restored["project_cwds"] == project_cwds
        assert restored["gate_agents"] == gate_agents

    def test_sentinel_events_round_trip(self, state_path):
        sentinel_events = [
            {"type": "tool_use", "name": "bash", "ts": time.time()},
            {"type": "user_request", "intent": "fix bug", "ts": time.time()},
        ]

        save_state(
            sessions={},
            project_cwds={},
            gate_agents=set(),
            sentinel_events=sentinel_events,
            path=state_path,
        )

        restored = load_state(path=state_path)
        assert len(restored["sentinel_events"]) == 2
        assert restored["sentinel_events"][0]["name"] == "bash"

    def test_sentinel_events_capped_by_count(self, state_path):
        events = [{"type": "tool_use", "name": f"ev{i}", "ts": time.time()} for i in range(_SENTINEL_MAX_EVENTS + 10)]

        save_state(sessions={}, project_cwds={}, gate_agents=set(), sentinel_events=events, path=state_path)

        restored = load_state(path=state_path)
        assert len(restored["sentinel_events"]) == _SENTINEL_MAX_EVENTS
        # Should keep the most recent (last N)
        assert restored["sentinel_events"][0]["name"] == "ev10"
        assert restored["sentinel_events"][-1]["name"] == f"ev{_SENTINEL_MAX_EVENTS + 9}"

    def test_missing_file_returns_empty(self, state_path):
        restored = load_state(path=state_path)
        assert restored["sessions"] == {}
        assert restored["project_cwds"] == {}
        assert restored["gate_agents"] == set()
        assert restored["sentinel_events"] == []

    def test_corrupt_file_returns_empty(self, state_path):
        with open(state_path, "w") as f:
            f.write("not json")

        restored = load_state(path=state_path)
        assert restored["sessions"] == {}

    def test_wrong_version_returns_empty(self, state_path):
        with open(state_path, "w") as f:
            json.dump({"version": 999}, f)

        restored = load_state(path=state_path)
        assert restored["sessions"] == {}

    def test_atomic_write(self, state_path):
        """save_state should not leave partial files on crash."""
        save_state(sessions={}, project_cwds={}, gate_agents=set(), path=state_path)
        assert os.path.exists(state_path)
        assert not os.path.exists(state_path + ".tmp")


class TestClearState:
    def test_clear_removes_file(self, state_path):
        save_state(sessions={}, project_cwds={}, gate_agents=set(), path=state_path)
        assert os.path.exists(state_path)
        clear_state(path=state_path)
        assert not os.path.exists(state_path)

    def test_clear_missing_file_no_error(self, state_path):
        clear_state(path=state_path)  # should not raise


class TestSentinelSeedEvents:
    def test_sentinel_accepts_seed_events(self):
        """LLMSentinel should populate its event deque from seed_events."""
        from cross.events import EventBus
        from cross.llm import LLMConfig
        from cross.sentinels.llm_reviewer import LLMSentinel

        bus = EventBus()
        config = LLMConfig(model="test", api_key="test")
        seed = [
            {"type": "tool_use", "name": "bash", "ts": time.time()},
            {"type": "user_request", "intent": "hello", "ts": time.time()},
        ]

        sentinel = LLMSentinel(config=config, event_bus=bus, seed_events=seed)
        assert len(sentinel.get_events()) == 2
        assert sentinel.get_events()[0]["name"] == "bash"

    def test_sentinel_get_events_returns_copy(self):
        from cross.events import EventBus
        from cross.llm import LLMConfig
        from cross.sentinels.llm_reviewer import LLMSentinel

        bus = EventBus()
        config = LLMConfig(model="test", api_key="test")
        sentinel = LLMSentinel(config=config, event_bus=bus)

        events = sentinel.get_events()
        assert events == []
        # Modifying returned list should not affect sentinel
        events.append({"fake": True})
        assert sentinel.get_events() == []
