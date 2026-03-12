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

from cross.ansi import strip_ansi


class TestBasicANSIStripping:
    """Verify standard ANSI CSI sequences are removed."""

    def test_color_codes_removed(self):
        # ESC[31m = red, ESC[0m = reset
        result = strip_ansi(b"\x1b[31mHello\x1b[0m")
        assert "Hello" in result
        assert "\x1b" not in result

    def test_bold_and_underline(self):
        # ESC[1m = bold, ESC[4m = underline
        result = strip_ansi(b"\x1b[1m\x1b[4mBold Underline\x1b[0m")
        assert "Bold Underline" in result
        assert "\x1b" not in result

    def test_256_color(self):
        # ESC[38;5;196m = 256-color red
        result = strip_ansi(b"\x1b[38;5;196mColored\x1b[0m")
        assert "Colored" in result

    def test_24bit_truecolor(self):
        # ESC[38;2;255;128;0m = RGB orange
        result = strip_ansi(b"\x1b[38;2;255;128;0mOrange\x1b[0m")
        assert "Orange" in result

    def test_cursor_movement(self):
        # ESC[5A = cursor up 5, ESC[10B = cursor down 10
        result = strip_ansi(b"\x1b[5AHello\x1b[10B")
        assert "Hello" in result

    def test_cursor_position(self):
        # ESC[10;20H = move cursor to row 10, col 20
        result = strip_ansi(b"\x1b[10;20HPositioned")
        assert "Positioned" in result

    def test_erase_sequences(self):
        # ESC[2J = clear screen, ESC[K = clear to end of line
        result = strip_ansi(b"\x1b[2J\x1b[KClean")
        assert "Clean" in result

    def test_question_mark_csi(self):
        # ESC[?25h = show cursor, ESC[?25l = hide cursor
        result = strip_ansi(b"\x1b[?25hVisible\x1b[?25l")
        assert "Visible" in result


class TestOSCSequences:
    """Verify OSC (Operating System Command) sequences are removed."""

    def test_osc_title_with_bel(self):
        # ESC ] 0 ; title BEL
        result = strip_ansi(b"\x1b]0;My Terminal Title\x07Content")
        assert "Content" in result
        assert "My Terminal Title" not in result

    def test_osc_title_with_st(self):
        # ESC ] 0 ; title ESC \
        result = strip_ansi(b"\x1b]0;Window Title\x1b\\Content")
        assert "Content" in result
        assert "Window Title" not in result

    def test_osc_hyperlink(self):
        # ESC ] 8 ; ; url BEL text ESC ] 8 ; ; BEL
        result = strip_ansi(b"\x1b]8;;https://example.com\x07Click\x1b]8;;\x07")
        assert "Click" in result
        assert "https://example.com" not in result


class TestCharacterSetAndKeypad:
    """Verify character set selection and keypad mode escapes are removed."""

    def test_character_set_selection(self):
        # ESC ( B = US ASCII, ESC ) 0 = DEC Special Graphics
        result = strip_ansi(b"\x1b(B\x1b)0Text")
        assert "Text" in result
        assert "\x1b" not in result

    def test_keypad_modes(self):
        # ESC = application keypad, ESC > normal keypad
        result = strip_ansi(b"\x1b=\x1b>Keys")
        assert "Keys" in result

    def test_single_char_escape(self):
        # ESC followed by various single chars (e.g., ESC M = reverse index)
        result = strip_ansi(b"\x1bMReversed")
        assert "Reversed" in result


class TestPartialCSICleanup:
    """Verify leftover partial CSI parameters are cleaned up."""

    def test_orphaned_color_params(self):
        # Partial CSI like "38;2;248;242m" without the ESC[
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
        assert "beforeafter" in result

    def test_bell_character(self):
        result = strip_ansi(b"alert\x07here")
        # BEL (0x07) is in range 0x00-0x08, should be stripped
        assert "alerthere" in result

    def test_backspace(self):
        result = strip_ansi(b"back\x08space")
        assert "backspace" in result

    def test_delete_character(self):
        result = strip_ansi(b"del\x7fete")
        assert "delete" in result

    def test_preserved_newline(self):
        result = strip_ansi(b"line1\nline2")
        assert "line1\nline2" in result

    def test_preserved_tab(self):
        result = strip_ansi(b"col1\tcol2")
        # Tab is preserved but spaces collapsed
        assert "col1" in result
        assert "col2" in result

    def test_preserved_carriage_return(self):
        result = strip_ansi(b"old\rnew")
        assert "\r" in result

    def test_escape_without_sequence(self):
        # Bare ESC followed by something not matching any pattern
        # The _ANSI_RE catches ESC + any single char via the "|." branch
        result = strip_ansi(b"\x1b\x1b")
        assert "\x1b" not in result

    def test_vertical_tab_stripped(self):
        # 0x0b = vertical tab, should be stripped
        result = strip_ansi(b"before\x0bafter")
        assert "beforeafter" in result

    def test_form_feed_stripped(self):
        # 0x0c = form feed, should be stripped
        result = strip_ansi(b"before\x0cafter")
        assert "beforeafter" in result


class TestBoxDrawingRemoval:
    """Verify box drawing and decoration characters are removed."""

    def test_horizontal_line(self):
        result = strip_ansi(b"Title\n" + "─".encode() * 40 + b"\nContent")
        assert "Title" in result
        assert "Content" in result

    def test_box_corners(self):
        box = "╭──────╮\n│ text │\n╰──────╯".encode()
        result = strip_ansi(box)
        assert "text" in result

    def test_rounded_corners(self):
        result = strip_ansi("╭╮╯╰".encode())
        stripped = result.strip()
        assert stripped == ""

    def test_pipe_and_double_line(self):
        result = strip_ansi("│║═".encode())
        stripped = result.strip()
        assert stripped == ""

    def test_mixed_decorations(self):
        result = strip_ansi("├── Item ──┤".encode())
        assert "Item" in result

    def test_dashed_line(self):
        result = strip_ansi("╌╌╌╌╌".encode())
        stripped = result.strip()
        assert stripped == ""


class TestWhitespaceCollapsing:
    """Verify multiple spaces and blank lines are collapsed."""

    def test_multiple_spaces_collapsed(self):
        result = strip_ansi(b"word1     word2")
        assert "word1 word2" in result

    def test_tabs_collapsed_to_space(self):
        result = strip_ansi(b"col1\t\t\tcol2")
        assert "col1 col2" in result

    def test_triple_newlines_collapsed(self):
        result = strip_ansi(b"para1\n\n\n\npara2")
        assert "para1\n\npara2" in result

    def test_double_newline_preserved(self):
        result = strip_ansi(b"para1\n\npara2")
        assert "para1\n\npara2" in result


class TestEdgeCases:
    """Verify edge cases and boundary conditions."""

    def test_empty_input(self):
        result = strip_ansi(b"")
        assert result == ""

    def test_only_ansi_codes(self):
        result = strip_ansi(b"\x1b[31m\x1b[1m\x1b[0m")
        assert result.strip() == ""

    def test_only_control_chars(self):
        result = strip_ansi(b"\x00\x01\x02\x03")
        assert result.strip() == ""

    def test_only_box_drawing(self):
        result = strip_ansi("───────".encode())
        assert result.strip() == ""

    def test_mixed_content(self):
        """Realistic terminal output with ANSI codes, box drawing, and control chars."""
        data = (
            b"\x1b[1m\x1b[34m"  # bold blue
            + "╭── Status ".encode()
            + "─".encode() * 10
            + "╮".encode()
            + b"\x1b[0m\n"
            + "│ ".encode()
            + b"\x1b[32m"  # green
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

    def test_utf8_replacement(self):
        # Invalid UTF-8 byte sequence should be replaced, not crash
        result = strip_ansi(b"valid \xff\xfe text")
        assert "valid" in result
        assert "text" in result

    def test_real_world_claude_output(self):
        """Simulates typical Claude Code terminal output."""
        data = (
            b"\x1b[?25l"  # hide cursor
            b"\x1b[1m\x1b[36m"  # bold cyan
            b"Claude"
            b"\x1b[0m"
            b" > I'll read that file for you."
            b"\x1b[?25h"  # show cursor
        )
        result = strip_ansi(data)
        assert "Claude" in result
        assert "read that file" in result
