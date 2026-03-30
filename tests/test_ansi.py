"""Unit tests for ANSI escape code stripping and terminal output cleaning.

Full coverage for cross/ansi.py:
- ANSI CSI escape sequences (colors, cursor movement, etc.)
- OSC sequences (title setting, hyperlinks)
- Character set selection and keypad mode escapes
- Partial CSI parameter cleanup
- Control character removal
- Box drawing / decoration character removal
- Whitespace collapsing (multiple spaces, blank lines)
- Edge cases (empty input, only ANSI codes, only decorations, mixed content)
"""

from cross.ansi import format_notification, strip_ansi


class TestBasicANSIStripping:
    """Verify standard ANSI CSI sequences are removed."""

    def test_color_codes_removed(self):
        result = strip_ansi(b"\x1b[31mHello\x1b[0m")
        assert result == "Hello"

    def test_bold_and_underline(self):
        result = strip_ansi(b"\x1b[1m\x1b[4mBold Underline\x1b[0m")
        assert result == "Bold Underline"

    def test_256_color(self):
        result = strip_ansi(b"\x1b[38;5;196mColored\x1b[0m")
        assert result == "Colored"

    def test_24bit_truecolor(self):
        result = strip_ansi(b"\x1b[38;2;255;128;0mOrange\x1b[0m")
        assert result == "Orange"

    def test_cursor_movement(self):
        result = strip_ansi(b"\x1b[5AHello\x1b[10B")
        assert result == "Hello"

    def test_cursor_position(self):
        result = strip_ansi(b"\x1b[10;20HPositioned")
        assert result == "Positioned"

    def test_erase_sequences(self):
        result = strip_ansi(b"\x1b[2J\x1b[KClean")
        assert result == "Clean"

    def test_question_mark_csi(self):
        result = strip_ansi(b"\x1b[?25hVisible\x1b[?25l")
        assert result == "Visible"


class TestOSCSequences:
    """Verify OSC (Operating System Command) sequences are removed."""

    def test_osc_title_with_bel(self):
        result = strip_ansi(b"\x1b]0;My Terminal Title\x07Content")
        assert result == "Content"

    def test_osc_title_with_st(self):
        result = strip_ansi(b"\x1b]0;Window Title\x1b\\Content")
        assert result == "Content"

    def test_osc_hyperlink(self):
        result = strip_ansi(b"\x1b]8;;https://example.com\x07Click\x1b]8;;\x07")
        assert result == "Click"


class TestCharacterSetAndKeypad:
    """Verify character set selection and keypad mode escapes are removed."""

    def test_character_set_selection(self):
        result = strip_ansi(b"\x1b(B\x1b)0Text")
        assert result == "Text"

    def test_keypad_modes(self):
        result = strip_ansi(b"\x1b=\x1b>Keys")
        assert result == "Keys"

    def test_single_char_escape(self):
        result = strip_ansi(b"\x1bMReversed")
        assert result == "Reversed"


class TestPartialCSICleanup:
    """Verify leftover partial CSI parameters are cleaned up."""

    def test_orphaned_color_params(self):
        result = strip_ansi(b"prefix 38;2;248;242m suffix")
        assert "prefix" in result
        assert "suffix" in result
        assert "38;2;248;242m" not in result

    def test_simple_semicolon_params(self):
        result = strip_ansi(b"text 1;2m more")
        assert "text" in result
        assert "more" in result
        assert "1;2m" not in result

    def test_single_number_not_matched(self):
        # Single number followed by 'm' should NOT be stripped (avoid false positives)
        result = strip_ansi(b"item 5m away")
        assert "5m" in result


class TestControlCharacters:
    """Verify control characters are stripped (except newline, tab, CR)."""

    def test_null_byte(self):
        result = strip_ansi(b"before\x00after")
        assert result == "beforeafter"

    def test_bell_character(self):
        result = strip_ansi(b"alert\x07here")
        assert result == "alerthere"

    def test_backspace(self):
        result = strip_ansi(b"back\x08space")
        assert result == "backspace"

    def test_delete_character(self):
        result = strip_ansi(b"del\x7fete")
        assert result == "delete"

    def test_preserved_newline(self):
        result = strip_ansi(b"line1\nline2")
        assert result == "line1\nline2"

    def test_preserved_tab(self):
        # Tab is converted to space by whitespace collapsing
        result = strip_ansi(b"col1\tcol2")
        assert result == "col1 col2"

    def test_preserved_carriage_return(self):
        result = strip_ansi(b"old\rnew")
        assert "\r" in result
        assert "old" in result
        assert "new" in result

    def test_escape_without_sequence(self):
        result = strip_ansi(b"\x1b\x1b")
        assert "\x1b" not in result

    def test_vertical_tab_stripped(self):
        result = strip_ansi(b"before\x0bafter")
        assert result == "beforeafter"

    def test_form_feed_stripped(self):
        result = strip_ansi(b"before\x0cafter")
        assert result == "beforeafter"


class TestBoxDrawingRemoval:
    """Verify box drawing and decoration characters are removed."""

    def test_horizontal_line(self):
        result = strip_ansi(b"Title\n" + "─".encode() * 40 + b"\nContent")
        assert "Title" in result
        assert "Content" in result
        assert "─" not in result

    def test_box_corners_fully_stripped(self):
        box = "╭──────╮\n│ text │\n╰──────╯".encode()
        result = strip_ansi(box)
        assert "text" in result
        # All box drawing chars should be gone
        for ch in "╭╮╯╰│──":
            assert ch not in result

    def test_only_decorations(self):
        result = strip_ansi("╭╮╯╰".encode())
        assert result.strip() == ""

    def test_pipe_and_double_line(self):
        result = strip_ansi("│║═".encode())
        assert result.strip() == ""

    def test_mixed_decorations(self):
        result = strip_ansi("├── Item ──┤".encode())
        assert "Item" in result
        assert "├" not in result
        assert "┤" not in result

    def test_dashed_line(self):
        result = strip_ansi("╌╌╌╌╌".encode())
        assert result.strip() == ""


class TestWhitespaceCollapsing:
    """Verify multiple spaces and blank lines are collapsed."""

    def test_multiple_spaces_collapsed(self):
        result = strip_ansi(b"word1     word2")
        assert result == "word1 word2"

    def test_tabs_collapsed_to_single_space(self):
        result = strip_ansi(b"col1\t\t\tcol2")
        assert result == "col1 col2"

    def test_triple_newlines_collapsed_to_double(self):
        result = strip_ansi(b"para1\n\n\n\npara2")
        assert result == "para1\n\npara2"

    def test_double_newline_preserved(self):
        result = strip_ansi(b"para1\n\npara2")
        assert result == "para1\n\npara2"


class TestEdgeCases:
    """Verify edge cases and boundary conditions."""

    def test_empty_input(self):
        assert strip_ansi(b"") == ""

    def test_only_ansi_codes(self):
        assert strip_ansi(b"\x1b[31m\x1b[1m\x1b[0m").strip() == ""

    def test_only_control_chars(self):
        assert strip_ansi(b"\x00\x01\x02\x03").strip() == ""

    def test_only_box_drawing(self):
        assert strip_ansi("───────".encode()).strip() == ""

    def test_mixed_content(self):
        """Realistic terminal output with ANSI codes, box drawing, and control chars."""
        data = (
            b"\x1b[1m\x1b[34m"
            + "╭── Status ".encode()
            + "─".encode() * 10
            + "╮".encode()
            + b"\x1b[0m\n"
            + "│ ".encode()
            + b"\x1b[32m"
            + b"OK"
            + b"\x1b[0m"
            + " │".encode()
            + b"\n"
            + "╰".encode()
            + "─".encode() * 14
            + "╯".encode()
        )
        result = strip_ansi(data)
        assert "Status" in result
        assert "OK" in result
        assert "\x1b" not in result
        # All box drawing chars gone
        for ch in "╭╮╯╰│──":
            assert ch not in result

    def test_utf8_replacement(self):
        result = strip_ansi(b"valid \xff\xfe text")
        assert "valid" in result
        assert "text" in result

    def test_real_world_claude_output(self):
        data = b"\x1b[?25l\x1b[1m\x1b[36mClaude\x1b[0m > I'll read that file for you.\x1b[?25h"
        result = strip_ansi(data)
        assert result == "Claude > I'll read that file for you."


class TestFormatNotification:
    """Verify OSC 9 terminal notification formatting."""

    def test_osc9_structure(self):
        notif = format_notification("Gate HALT: Bash", style="error")
        assert notif.startswith("\033]9;")  # OSC 9 prefix
        assert notif.endswith("\a")  # BEL terminator
        assert "Gate HALT: Bash" in notif

    def test_cross_prefix(self):
        notif = format_notification("Title")
        assert "cross:" in notif

    def test_error_icon(self):
        notif = format_notification("Title", style="error")
        assert "\U0001f6d1" in notif  # stop sign

    def test_warning_icon(self):
        notif = format_notification("Title", style="warning")
        assert "\u26a0\ufe0f" in notif  # warning sign

    def test_alert_icon(self):
        notif = format_notification("Title", style="alert")
        assert "\U0001f514" in notif  # bell

    def test_unknown_style_falls_back_to_error(self):
        notif = format_notification("Test", style="nonexistent")
        assert "\U0001f6d1" in notif  # stop sign fallback

    def test_body_included(self):
        notif = format_notification("Title", body="Some reason here")
        assert "Some reason here" in notif

    def test_empty_body_omitted(self):
        notif = format_notification("Title")
        assert " — " not in notif

    def test_body_truncated(self):
        long_body = "x" * 300
        notif = format_notification("Title", body=long_body)
        assert "..." in notif
