"""Tests for cross.custom_instructions — hot-reloadable custom instructions."""

from __future__ import annotations

import os
import time

import pytest

from cross.custom_instructions import CustomInstructions, format_instructions_block


class TestFormatInstructionsBlock:
    def test_empty_string(self):
        assert format_instructions_block("") == ""

    def test_whitespace_only(self):
        assert format_instructions_block("   \n\n  ") == ""

    def test_wraps_content(self):
        result = format_instructions_block("Be strict about network calls.")
        assert "--- Custom Instructions" in result
        assert "Be strict about network calls." in result
        assert "--- End Custom Instructions ---" in result

    def test_strips_leading_trailing_whitespace(self):
        result = format_instructions_block("  hello  \n")
        assert "hello" in result
        # Should not have leading/trailing whitespace in the content
        assert "  hello" not in result


class TestCustomInstructions:
    def test_missing_file(self, tmp_path):
        ci = CustomInstructions(path=tmp_path / "nonexistent.md")
        assert ci.content == ""

    def test_load_existing_file(self, tmp_path):
        f = tmp_path / "instructions.md"
        f.write_text("Block all network calls.")
        ci = CustomInstructions(path=f)
        assert ci.content == "Block all network calls."

    def test_save_creates_file(self, tmp_path):
        f = tmp_path / "subdir" / "instructions.md"
        ci = CustomInstructions(path=f)
        assert ci.content == ""
        ci.save("New instructions")
        assert ci.content == "New instructions"
        assert f.read_text() == "New instructions"

    def test_hot_reload_on_mtime_change(self, tmp_path):
        f = tmp_path / "instructions.md"
        f.write_text("Version 1")
        ci = CustomInstructions(path=f)
        assert ci.content == "Version 1"

        # Modify the file (force different mtime)
        time.sleep(0.05)
        f.write_text("Version 2")
        os.utime(f, (time.time() + 1, time.time() + 1))

        # Access triggers reload
        assert ci.content == "Version 2"

    def test_detects_file_deletion(self, tmp_path):
        f = tmp_path / "instructions.md"
        f.write_text("Some instructions")
        ci = CustomInstructions(path=f)
        assert ci.content == "Some instructions"

        f.unlink()
        assert ci.content == ""

    def test_detects_file_creation(self, tmp_path):
        f = tmp_path / "instructions.md"
        ci = CustomInstructions(path=f)
        assert ci.content == ""

        f.write_text("Created later")
        # Force mtime difference
        os.utime(f, (time.time() + 1, time.time() + 1))
        assert ci.content == "Created later"

    def test_save_and_reload(self, tmp_path):
        f = tmp_path / "instructions.md"
        ci = CustomInstructions(path=f)
        ci.save("First save")
        assert ci.content == "First save"

        # Simulate external edit
        time.sleep(0.05)
        f.write_text("External edit")
        os.utime(f, (time.time() + 1, time.time() + 1))
        assert ci.content == "External edit"
