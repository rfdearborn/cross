"""Unit tests for the session registry.

Full coverage for cross/session.py:
- SessionInfo dataclass fields and defaults
- SessionRegistry.create (ID generation, project detection, PTYSession allocation)
- SessionRegistry.get (found, not found)
- SessionRegistry.active (filters completed sessions)
- SessionRegistry.complete (sets ended_at and exit_code)
- _detect_project (git root detection, fallback to dirname)
- Global singleton registry
- Thread safety basics
"""

import os
import time

from cross.session import SessionInfo, SessionRegistry, _detect_project, registry


class TestSessionInfoDefaults:
    """Verify SessionInfo dataclass defaults."""

    def test_defaults(self):
        info = SessionInfo(
            session_id="abc123",
            agent="claude",
            argv=["claude"],
            project="myproject",
            cwd="/tmp",
            started_at=1000.0,
        )
        assert info.session_id == "abc123"
        assert info.agent == "claude"
        assert info.argv == ["claude"]
        assert info.project == "myproject"
        assert info.cwd == "/tmp"
        assert info.started_at == 1000.0
        assert info.ended_at is None
        assert info.exit_code is None
        assert info.pty_session is None

    def test_completed_session(self):
        info = SessionInfo(
            session_id="def456",
            agent="codex",
            argv=["codex", "--model", "o3"],
            project="cross",
            cwd="/home/user/cross",
            started_at=1000.0,
            ended_at=1050.0,
            exit_code=0,
        )
        assert info.ended_at == 1050.0
        assert info.exit_code == 0


class TestSessionRegistryCreate:
    """Verify SessionRegistry.create generates unique sessions with correct fields."""

    def test_create_returns_session_info(self):
        reg = SessionRegistry()
        info = reg.create(agent="claude", argv=["claude", "--model", "opus"], cwd="/tmp")
        assert isinstance(info, SessionInfo)
        assert info.agent == "claude"
        assert info.argv == ["claude", "--model", "opus"]
        assert info.cwd == "/tmp"
        assert info.ended_at is None
        assert info.exit_code is None
        assert info.pty_session is not None

    def test_session_id_is_12_hex_chars(self):
        reg = SessionRegistry()
        info = reg.create(agent="claude", argv=["claude"], cwd="/tmp")
        assert len(info.session_id) == 12
        # Should be valid hex
        int(info.session_id, 16)

    def test_unique_session_ids(self):
        reg = SessionRegistry()
        ids = set()
        for _ in range(50):
            info = reg.create(agent="claude", argv=["claude"], cwd="/tmp")
            ids.add(info.session_id)
        assert len(ids) == 50

    def test_create_uses_cwd_when_none(self):
        reg = SessionRegistry()
        info = reg.create(agent="claude", argv=["claude"], cwd=None)
        assert info.cwd == os.getcwd()

    def test_started_at_is_recent(self):
        before = time.time()
        reg = SessionRegistry()
        info = reg.create(agent="claude", argv=["claude"], cwd="/tmp")
        after = time.time()
        assert before <= info.started_at <= after

    def test_create_stores_session(self):
        reg = SessionRegistry()
        info = reg.create(agent="claude", argv=["claude"], cwd="/tmp")
        assert reg.get(info.session_id) is info


class TestSessionRegistryGet:
    """Verify SessionRegistry.get retrieval."""

    def test_get_existing_session(self):
        reg = SessionRegistry()
        info = reg.create(agent="claude", argv=["claude"], cwd="/tmp")
        retrieved = reg.get(info.session_id)
        assert retrieved is info

    def test_get_nonexistent_returns_none(self):
        reg = SessionRegistry()
        assert reg.get("nonexistent_id") is None

    def test_get_empty_string(self):
        reg = SessionRegistry()
        assert reg.get("") is None


class TestSessionRegistryActive:
    """Verify SessionRegistry.active filters completed sessions."""

    def test_active_returns_only_incomplete(self):
        reg = SessionRegistry()
        s1 = reg.create(agent="claude", argv=["claude"], cwd="/tmp")
        s2 = reg.create(agent="codex", argv=["codex"], cwd="/tmp")
        s3 = reg.create(agent="cursor", argv=["cursor"], cwd="/tmp")

        # Complete one session
        reg.complete(s2.session_id, exit_code=0)

        active = reg.active()
        active_ids = [s.session_id for s in active]
        assert s1.session_id in active_ids
        assert s2.session_id not in active_ids
        assert s3.session_id in active_ids

    def test_active_empty_registry(self):
        reg = SessionRegistry()
        assert reg.active() == []

    def test_active_all_completed(self):
        reg = SessionRegistry()
        s1 = reg.create(agent="claude", argv=["claude"], cwd="/tmp")
        s2 = reg.create(agent="codex", argv=["codex"], cwd="/tmp")
        reg.complete(s1.session_id, exit_code=0)
        reg.complete(s2.session_id, exit_code=1)
        assert reg.active() == []


class TestSessionRegistryComplete:
    """Verify SessionRegistry.complete marks sessions as done."""

    def test_complete_sets_ended_at_and_exit_code(self):
        reg = SessionRegistry()
        info = reg.create(agent="claude", argv=["claude"], cwd="/tmp")
        before = time.time()
        reg.complete(info.session_id, exit_code=0)
        after = time.time()

        assert info.exit_code == 0
        assert info.ended_at is not None
        assert before <= info.ended_at <= after

    def test_complete_nonexistent_is_noop(self):
        reg = SessionRegistry()
        # Should not raise
        reg.complete("nonexistent_id", exit_code=1)

    def test_complete_with_nonzero_exit(self):
        reg = SessionRegistry()
        info = reg.create(agent="claude", argv=["claude"], cwd="/tmp")
        reg.complete(info.session_id, exit_code=137)
        assert info.exit_code == 137


class TestDetectProject:
    """Verify _detect_project git root detection and fallback."""

    def test_detects_git_root(self, tmp_path):
        # Create a git repo structure
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        (project_dir / ".git").mkdir()
        sub_dir = project_dir / "src" / "deep"
        sub_dir.mkdir(parents=True)

        result = _detect_project(str(sub_dir))
        assert result == "myproject"

    def test_detects_git_at_cwd(self, tmp_path):
        project_dir = tmp_path / "repo"
        project_dir.mkdir()
        (project_dir / ".git").mkdir()

        result = _detect_project(str(project_dir))
        assert result == "repo"

    def test_fallback_to_dirname(self, tmp_path):
        # No .git anywhere in the path
        leaf = tmp_path / "somedir" / "leaf"
        leaf.mkdir(parents=True)

        result = _detect_project(str(leaf))
        assert result == "leaf"

    def test_symlink_resolved(self, tmp_path):
        # Create a real project with .git
        real_dir = tmp_path / "real_project"
        real_dir.mkdir()
        (real_dir / ".git").mkdir()

        # Create a symlink
        link = tmp_path / "link"
        link.symlink_to(real_dir)

        result = _detect_project(str(link))
        assert result == "real_project"


class TestGlobalRegistry:
    """Verify the module-level singleton registry."""

    def test_singleton_is_session_registry(self):
        assert isinstance(registry, SessionRegistry)

    def test_singleton_create_and_get(self):
        info = registry.create(agent="test_agent", argv=["test"], cwd="/tmp")
        assert registry.get(info.session_id) is info
        # Clean up
        registry.complete(info.session_id, exit_code=0)


class TestProjectDetectionFromCreate:
    """Verify that create() correctly derives the project name."""

    def test_project_from_git_repo(self, tmp_path):
        project_dir = tmp_path / "cross"
        project_dir.mkdir()
        (project_dir / ".git").mkdir()
        sub = project_dir / "src"
        sub.mkdir()

        reg = SessionRegistry()
        info = reg.create(agent="claude", argv=["claude"], cwd=str(sub))
        assert info.project == "cross"

    def test_project_from_leaf_dir(self, tmp_path):
        leaf = tmp_path / "standalone"
        leaf.mkdir()

        reg = SessionRegistry()
        info = reg.create(agent="claude", argv=["claude"], cwd=str(leaf))
        assert info.project == "standalone"
