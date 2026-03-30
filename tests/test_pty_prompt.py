"""Tests for PTY interactive prompt rendering and key parsing."""

from cross.pty_prompt import parse_key


class TestParseKey:
    def test_up_arrow(self):
        assert parse_key(b"\x1b[A") == "up"

    def test_down_arrow(self):
        assert parse_key(b"\x1b[B") == "down"

    def test_enter_cr(self):
        assert parse_key(b"\r") == "enter"

    def test_enter_lf(self):
        assert parse_key(b"\n") == "enter"

    def test_escape(self):
        assert parse_key(b"\x1b") == "escape"

    def test_q_quit(self):
        assert parse_key(b"q") == "quit"

    def test_Q_quit(self):
        assert parse_key(b"Q") == "quit"

    def test_unknown_key(self):
        assert parse_key(b"x") == ""

    def test_arrow_in_longer_buffer(self):
        # Arrow key at start of a larger buffer still detected
        assert parse_key(b"\x1b[Aextra") == "up"
