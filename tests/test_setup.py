"""Tests for the interactive setup wizard."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from cross.setup import (
    KNOWN_AGENTS,
    SHELL_WRAPPER_HEADER,
    _build_env_lines,
    _build_shell_wrappers,
    _check_ollama,
    _detect_agents,
    _detect_shell_rc,
    _parse_provider,
    _strip_ansi,
    run_setup,
)


class TestStripAnsi:
    def test_strips_escape_sequences(self):
        assert _strip_ansi("\x1b[CAIzaSy123") == "AIzaSy123"

    def test_strips_multiple_sequences(self):
        assert _strip_ansi("\x1b[A\x1b[Bhello\x1b[C") == "hello"

    def test_no_ansi_unchanged(self):
        assert _strip_ansi("normal-key-123") == "normal-key-123"

    def test_empty_string(self):
        assert _strip_ansi("") == ""


class TestParseProvider:
    def test_google(self):
        assert _parse_provider("google/gemini-3-flash-preview") == "google"

    def test_anthropic(self):
        assert _parse_provider("anthropic/claude-haiku-4-5") == "anthropic"

    def test_ollama(self):
        assert _parse_provider("ollama/llama3.1:8b") == "ollama"

    def test_cli(self):
        assert _parse_provider("cli/claude") == "cli"

    def test_bare_claude_is_cli(self):
        assert _parse_provider("claude") == "cli"

    def test_no_slash_defaults_to_anthropic(self):
        assert _parse_provider("claude-haiku-4-5") == "anthropic"


class TestDetectAgents:
    @patch("shutil.which")
    def test_finds_installed_agents(self, mock_which):
        mock_which.side_effect = lambda name: f"/usr/bin/{name}" if name == "claude" else None
        agents = _detect_agents()
        assert agents == ["claude"]

    @patch("shutil.which", return_value=None)
    def test_no_agents_found(self, mock_which):
        agents = _detect_agents()
        assert agents == []

    @patch("shutil.which")
    def test_all_agents_found(self, mock_which):
        mock_which.side_effect = lambda name: f"/usr/bin/{name}"
        agents = _detect_agents()
        assert agents == KNOWN_AGENTS

    @patch("shutil.which")
    def test_finds_openclaw(self, mock_which):
        mock_which.side_effect = lambda name: f"/usr/local/bin/{name}" if name == "openclaw" else None
        agents = _detect_agents()
        assert agents == ["openclaw"]

    def test_known_agents_includes_openclaw(self):
        assert "openclaw" in KNOWN_AGENTS


class TestCheckOllama:
    @patch("socket.create_connection")
    def test_ollama_reachable(self, mock_conn):
        mock_sock = MagicMock()
        mock_conn.return_value = mock_sock
        assert _check_ollama() is True
        mock_sock.close.assert_called_once()

    @patch("socket.create_connection", side_effect=OSError("refused"))
    def test_ollama_not_reachable(self, mock_conn):
        assert _check_ollama() is False


class TestDetectShellRc:
    @patch.dict(os.environ, {"SHELL": "/bin/zsh"})
    def test_detects_zshrc(self):
        rc = _detect_shell_rc()
        assert rc is not None
        assert rc.name == ".zshrc"

    @patch.dict(os.environ, {"SHELL": "/bin/bash"})
    def test_detects_bashrc(self):
        rc = _detect_shell_rc()
        assert rc is not None
        assert rc.name == ".bashrc"

    @patch.dict(os.environ, {"SHELL": "/bin/fish"})
    def test_unknown_shell_returns_none(self):
        rc = _detect_shell_rc()
        assert rc is None

    @patch.dict(os.environ, {}, clear=True)
    def test_no_shell_env_returns_none(self):
        rc = _detect_shell_rc()
        assert rc is None


class TestBuildEnvLines:
    def test_llm_enabled_with_key(self):
        lines = _build_env_lines(
            gate_model="google/gemini-3-flash-preview",
            gate_api_key="test-key-123",
            llm_enabled=True,
            slack_bot_token=None,
            slack_app_token=None,
        )
        text = "\n".join(lines)
        assert "CROSS_LLM_GATE_MODEL=google/gemini-3-flash-preview" in text
        assert "CROSS_LLM_SENTINEL_MODEL=google/gemini-3-flash-preview" in text
        assert "CROSS_LLM_GATE_API_KEY=test-key-123" in text
        assert "CROSS_LLM_SENTINEL_API_KEY=test-key-123" in text
        assert "CROSS_LLM_GATE_ENABLED=false" not in text

    def test_llm_disabled(self):
        lines = _build_env_lines(
            gate_model=None,
            gate_api_key=None,
            llm_enabled=False,
            slack_bot_token=None,
            slack_app_token=None,
        )
        text = "\n".join(lines)
        assert "CROSS_LLM_GATE_ENABLED=false" in text
        assert "CROSS_LLM_SENTINEL_ENABLED=false" in text
        assert "CROSS_LLM_GATE_MODEL" not in text

    def test_with_slack_tokens(self):
        lines = _build_env_lines(
            gate_model="anthropic/claude-haiku-4-5",
            gate_api_key="sk-ant-123",
            llm_enabled=True,
            slack_bot_token="xoxb-test",
            slack_app_token="xapp-test",
        )
        text = "\n".join(lines)
        assert "CROSS_SLACK_BOT_TOKEN=xoxb-test" in text
        assert "CROSS_SLACK_APP_TOKEN=xapp-test" in text

    def test_ollama_no_key(self):
        lines = _build_env_lines(
            gate_model="ollama/llama3.1:8b",
            gate_api_key=None,
            llm_enabled=True,
            slack_bot_token=None,
            slack_app_token=None,
        )
        text = "\n".join(lines)
        assert "CROSS_LLM_GATE_MODEL=ollama/llama3.1:8b" in text
        assert "CROSS_LLM_GATE_API_KEY" not in text

    def test_separate_sentinel_model(self):
        lines = _build_env_lines(
            gate_model="google/gemini-3-flash-preview",
            gate_api_key="google-key",
            llm_enabled=True,
            slack_bot_token=None,
            slack_app_token=None,
            sentinel_model="anthropic/claude-haiku-4-5",
            sentinel_api_key="ant-key",
        )
        text = "\n".join(lines)
        assert "CROSS_LLM_GATE_MODEL=google/gemini-3-flash-preview" in text
        assert "CROSS_LLM_SENTINEL_MODEL=anthropic/claude-haiku-4-5" in text
        assert "CROSS_LLM_GATE_API_KEY=google-key" in text
        assert "CROSS_LLM_SENTINEL_API_KEY=ant-key" in text

    def test_sentinel_interval(self):
        lines = _build_env_lines(
            gate_model="google/gemini-3-flash-preview",
            gate_api_key="key",
            llm_enabled=True,
            slack_bot_token=None,
            slack_app_token=None,
            sentinel_interval=120,
        )
        text = "\n".join(lines)
        assert "CROSS_LLM_SENTINEL_INTERVAL_SECONDS=120" in text

    def test_no_sentinel_interval_when_default(self):
        lines = _build_env_lines(
            gate_model="google/gemini-3-flash-preview",
            gate_api_key="key",
            llm_enabled=True,
            slack_bot_token=None,
            slack_app_token=None,
        )
        text = "\n".join(lines)
        assert "INTERVAL" not in text


class TestBuildShellWrappers:
    def test_single_agent(self):
        result = _build_shell_wrappers(["claude"])
        assert SHELL_WRAPPER_HEADER in result
        assert 'claude() { cross wrap -- claude "$@"; }' in result

    def test_multiple_agents(self):
        result = _build_shell_wrappers(["claude", "codex"])
        assert 'claude() { cross wrap -- claude "$@"; }' in result
        assert 'codex() { cross wrap -- codex "$@"; }' in result


# All run_setup tests patch sys.platform to 'linux' to skip the auto-start step,
# unless specifically testing auto-start.

# Input sequence for freeform model flow:
#   gate_model → [api_key] → sentinel_model → interval → email y/n → [email fields]
#   → slack y/n → [slack tokens] → shell_wrappers y/n → auto_update


@patch("cross.setup.sys")
class TestRunSetupDefaultModel:
    """Test pressing Enter to accept defaults throughout."""

    @patch("cross.setup._detect_agents", return_value=["claude"])
    @patch("cross.setup._detect_shell_rc")
    def test_default_model_flow(self, mock_shell_rc, mock_agents, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"
        shell_rc = tmp_path / ".zshrc"
        shell_rc.write_text("# existing content\n")
        mock_shell_rc.return_value = shell_rc

        # "" gate, "none" gate backup, "" sentinel, "" interval,
        # "none" sentinel backup, N email, N slack, Y wrappers, Y perm hook
        inputs = iter(["", "none", "", "", "none", "N", "N", "Y", "Y", "Y"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: "",
            print_fn=output.append,
        )

        assert result["llm_enabled"] is True
        assert result["model"] == "anthropic/claude-code/claude-sonnet-4-6"
        assert result["sentinel_model"] == "anthropic/claude-code/claude-opus-4-6"
        assert result["agents_found"] == ["claude"]

        env_file = cross_dir / ".env"
        assert env_file.exists()
        env_content = env_file.read_text()
        assert "CROSS_LLM_GATE_MODEL=anthropic/claude-code/claude-sonnet-4-6" in env_content
        assert "CROSS_LLM_SENTINEL_MODEL=anthropic/claude-code/claude-opus-4-6" in env_content
        # cli provider should NOT write an API key
        assert "CROSS_LLM_GATE_API_KEY" not in env_content

        assert (cross_dir / "rules.d").is_dir()
        assert result["shell_wrappers_installed"] is True

        full_output = "\n".join(str(o) for o in output)
        assert "Setup complete!" in full_output


@patch("cross.setup.sys")
class TestRunSetupAnthropicModel:
    @patch("cross.setup._detect_agents", return_value=[])
    def test_anthropic_model(self, mock_agents, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"

        # anthropic model, "none" gate backup, "" sentinel (same), "" interval, "none" sentinel backup, N email, N slack
        inputs = iter(["anthropic/claude-haiku-4-5", "none", "", "", "none", "N", "N", "Y"])
        secrets = iter(["sk-ant-test-key"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: next(secrets),
            print_fn=output.append,
        )

        assert result["llm_enabled"] is True
        assert result["model"] == "anthropic/claude-haiku-4-5"

        env_content = (cross_dir / ".env").read_text()
        assert "CROSS_LLM_GATE_MODEL=anthropic/claude-haiku-4-5" in env_content
        assert "CROSS_LLM_GATE_API_KEY=sk-ant-test-key" in env_content


@patch("cross.setup.sys")
class TestRunSetupOpenAIModel:
    @patch("cross.setup._detect_agents", return_value=[])
    def test_openai_model(self, mock_agents, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"

        inputs = iter(["openai/gpt-5-mini", "none", "", "", "none", "N", "N", "Y"])
        secrets = iter(["sk-openai-test"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: next(secrets),
            print_fn=output.append,
        )

        assert result["llm_enabled"] is True
        assert result["model"] == "openai/gpt-5-mini"
        env_content = (cross_dir / ".env").read_text()
        assert "CROSS_LLM_GATE_MODEL=openai/gpt-5-mini" in env_content


@patch("cross.setup.sys")
class TestRunSetupOllamaModel:
    @patch("cross.setup._check_ollama", return_value=True)
    @patch("cross.setup._detect_agents", return_value=[])
    def test_ollama_running(self, mock_agents, mock_ollama, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"

        # ollama gate, "none" gate backup, "" sentinel (same), "" interval, "none" sentinel backup, N email, N slack
        inputs = iter(["ollama/llama3.1:8b", "none", "", "", "none", "N", "N", "Y"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: "",
            print_fn=output.append,
        )

        assert result["llm_enabled"] is True
        assert result["model"] == "ollama/llama3.1:8b"
        full_output = "\n".join(str(o) for o in output)
        assert "Ollama detected" in full_output

    @patch("cross.setup._check_ollama", return_value=False)
    @patch("cross.setup._detect_agents", return_value=[])
    def test_ollama_not_running(self, mock_agents, mock_ollama, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"

        inputs = iter(["ollama/llama3.1:8b", "none", "", "", "none", "N", "N", "Y"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: "",
            print_fn=output.append,
        )

        assert result["llm_enabled"] is True
        assert result["model"] == "ollama/llama3.1:8b"
        full_output = "\n".join(str(o) for o in output)
        assert "could not connect to Ollama" in full_output


@patch("cross.setup.sys")
class TestRunSetupSeparateSentinel:
    """Test choosing a different model for the sentinel."""

    @patch("cross.setup._detect_agents", return_value=[])
    def test_different_sentinel_model_same_provider(self, mock_agents, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"

        # cli/claude gate, "none" gate backup, google sentinel (different provider),
        # 120s, "none" sentinel backup, N email, N slack
        inputs = iter(["", "none", "google/gemini-2.5-flash", "120", "none", "N", "N", "Y"])
        secrets = iter(["google-key"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: next(secrets),
            print_fn=output.append,
        )

        assert result["model"] == "anthropic/claude-code/claude-sonnet-4-6"
        assert result["sentinel_model"] == "google/gemini-2.5-flash"
        assert result["sentinel_interval"] == 120

        env_content = (cross_dir / ".env").read_text()
        assert "CROSS_LLM_GATE_MODEL=anthropic/claude-code/claude-sonnet-4-6" in env_content
        assert "CROSS_LLM_SENTINEL_MODEL=google/gemini-2.5-flash" in env_content
        assert "CROSS_LLM_SENTINEL_INTERVAL_SECONDS=120" in env_content

    @patch("cross.setup._detect_agents", return_value=[])
    def test_different_sentinel_provider_asks_for_key(self, mock_agents, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"

        # cli/claude gate, "none" gate backup, anthropic sentinel (different provider),
        # "" interval, "none" sentinel backup, N email, N slack
        inputs = iter(["", "none", "anthropic/claude-haiku-4-5", "", "none", "N", "N", "Y"])
        secrets = iter(["ant-key"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: next(secrets),
            print_fn=output.append,
        )

        assert result["sentinel_model"] == "anthropic/claude-haiku-4-5"
        env_content = (cross_dir / ".env").read_text()
        assert "CROSS_LLM_SENTINEL_MODEL=anthropic/claude-haiku-4-5" in env_content
        assert "CROSS_LLM_SENTINEL_API_KEY=ant-key" in env_content


@patch("cross.setup.sys")
class TestRunSetupCustomInterval:
    @patch("cross.setup._detect_agents", return_value=[])
    def test_custom_interval(self, mock_agents, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"

        # default gate (cli/claude), "none" gate backup, "" sentinel (same),
        # 300s interval, "none" sentinel backup, N email, N slack
        inputs = iter(["", "none", "", "300", "none", "N", "N", "Y"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: "",
            print_fn=output.append,
        )

        assert result["sentinel_interval"] == 300
        env_content = (cross_dir / ".env").read_text()
        assert "CROSS_LLM_SENTINEL_INTERVAL_SECONDS=300" in env_content

    @patch("cross.setup._detect_agents", return_value=[])
    def test_invalid_interval_uses_default(self, mock_agents, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"

        # default gate (cli/claude), "none" gate backup, "" sentinel (same),
        # "abc" interval, "none" sentinel backup, N email, N slack
        inputs = iter(["", "none", "", "abc", "none", "N", "N", "Y"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: "",
            print_fn=output.append,
        )

        assert result["sentinel_interval"] is None  # None = use config default
        env_content = (cross_dir / ".env").read_text()
        assert "INTERVAL" not in env_content


@patch("cross.setup.sys")
class TestRunSetupNoneLLM:
    @patch("cross.setup._detect_agents", return_value=[])
    def test_none_disables_llm(self, mock_agents, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"

        # "none" skips sentinel + interval prompts entirely
        inputs = iter(["none", "N", "N", "Y"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: "",
            print_fn=output.append,
        )

        assert result["llm_enabled"] is False
        assert result["model"] is None
        env_content = (cross_dir / ".env").read_text()
        assert "CROSS_LLM_GATE_ENABLED=false" in env_content
        assert "CROSS_LLM_SENTINEL_ENABLED=false" in env_content

        full_output = "\n".join(str(o) for o in output)
        assert "disabled" in full_output


@patch("cross.setup.sys")
class TestRunSetupEmptyApiKey:
    @patch("cross.setup._detect_agents", return_value=[])
    def test_empty_api_key_no_env_disables_llm(self, mock_agents, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"

        # Explicit google model, empty key, no env → disables LLM → skips sentinel
        inputs = iter(["google/gemini-3-flash-preview", "N", "N", "Y"])
        secrets = iter([""])  # empty key

        output = []
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("GOOGLE_API_KEY", None)
            result = run_setup(
                cross_dir=cross_dir,
                input_fn=lambda p: next(inputs),
                getpass_fn=lambda p: next(secrets),
                print_fn=output.append,
            )

        assert result["llm_enabled"] is False
        full_output = "\n".join(str(o) for o in output)
        assert "No API key provided" in full_output

    @patch("cross.setup._detect_agents", return_value=[])
    def test_empty_key_falls_back_to_env_var(self, mock_agents, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"

        # Explicit google model, empty key (env fallback), "none" gate backup,
        # "" sentinel, "" interval, "none" sentinel backup, N email, N slack
        inputs = iter(["google/gemini-3-flash-preview", "none", "", "", "none", "N", "N", "Y"])
        secrets = iter([""])  # empty key

        output = []
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "env-key-123"}):
            result = run_setup(
                cross_dir=cross_dir,
                input_fn=lambda p: next(inputs),
                getpass_fn=lambda p: next(secrets),
                print_fn=output.append,
            )

        assert result["llm_enabled"] is True
        full_output = "\n".join(str(o) for o in output)
        assert "Found $GOOGLE_API_KEY" in full_output
        env_content = (cross_dir / ".env").read_text()
        assert "CROSS_LLM_GATE_API_KEY=env-key-123" in env_content


@patch("cross.setup.sys")
class TestRunSetupSlack:
    @patch("cross.setup._detect_agents", return_value=[])
    def test_slack_configured(self, mock_agents, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"

        inputs = iter(["none", "N", "y", "Y"])
        secrets = iter(["xoxb-bot-token", "xapp-app-token"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: next(secrets),
            print_fn=output.append,
        )

        assert result["slack_configured"] is True
        env_content = (cross_dir / ".env").read_text()
        assert "CROSS_SLACK_BOT_TOKEN=xoxb-bot-token" in env_content
        assert "CROSS_SLACK_APP_TOKEN=xapp-app-token" in env_content

    @patch("cross.setup._detect_agents", return_value=[])
    def test_slack_empty_tokens_skipped(self, mock_agents, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"

        inputs = iter(["none", "N", "y", "Y"])
        secrets = iter(["", ""])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: next(secrets),
            print_fn=output.append,
        )

        assert result["slack_configured"] is False
        env_content = (cross_dir / ".env").read_text()
        assert "SLACK" not in env_content


@patch("cross.setup.sys")
class TestRunSetupSlackPreservation:
    @patch("cross.setup._detect_agents", return_value=[])
    def test_existing_slack_preserved_on_rerun(self, mock_agents, mock_sys, tmp_path):
        """Re-running setup preserves Slack tokens when user says Y."""
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"
        cross_dir.mkdir(parents=True)
        (cross_dir / ".env").write_text("CROSS_SLACK_BOT_TOKEN=xoxb-existing\nCROSS_SLACK_APP_TOKEN=xapp-existing\n")

        # "none" gate, N email, "Y" keep slack, "Y" auto-update
        inputs = iter(["none", "N", "Y", "Y"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: "",
            print_fn=output.append,
        )

        assert result["slack_configured"] is True
        env_content = (cross_dir / ".env").read_text()
        assert "CROSS_SLACK_BOT_TOKEN=xoxb-existing" in env_content
        assert "CROSS_SLACK_APP_TOKEN=xapp-existing" in env_content

    @patch("cross.setup._detect_agents", return_value=[])
    def test_existing_slack_replaced_on_rerun(self, mock_agents, mock_sys, tmp_path):
        """Re-running setup allows replacing Slack tokens."""
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"
        cross_dir.mkdir(parents=True)
        (cross_dir / ".env").write_text("CROSS_SLACK_BOT_TOKEN=xoxb-old\nCROSS_SLACK_APP_TOKEN=xapp-old\n")

        # "none" gate, N email, "n" don't keep, "y" configure new, "Y" auto-update
        inputs = iter(["none", "N", "n", "y", "Y"])
        secrets = iter(["xoxb-new", "xapp-new"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: next(secrets),
            print_fn=output.append,
        )

        assert result["slack_configured"] is True
        env_content = (cross_dir / ".env").read_text()
        assert "CROSS_SLACK_BOT_TOKEN=xoxb-new" in env_content
        assert "CROSS_SLACK_APP_TOKEN=xapp-new" in env_content


@patch("cross.setup.sys")
class TestRunSetupShellWrappers:
    @patch("cross.setup._detect_shell_rc")
    @patch("cross.setup._detect_agents", return_value=["claude", "codex"])
    def test_shell_wrappers_installed(self, mock_agents, mock_shell_rc, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"
        shell_rc = tmp_path / ".zshrc"
        shell_rc.write_text("# existing\n")
        mock_shell_rc.return_value = shell_rc

        inputs = iter(["none", "N", "N", "Y", "Y", "Y"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: "",
            print_fn=output.append,
        )

        assert result["shell_wrappers_installed"] is True
        rc_content = shell_rc.read_text()
        assert 'claude() { cross wrap -- claude "$@"; }' in rc_content
        assert 'codex() { cross wrap -- codex "$@"; }' in rc_content

    @patch("cross.setup._detect_shell_rc")
    @patch("cross.setup._detect_agents", return_value=["claude"])
    def test_shell_wrappers_already_present(self, mock_agents, mock_shell_rc, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"
        shell_rc = tmp_path / ".zshrc"
        shell_rc.write_text(f'# existing\n{SHELL_WRAPPER_HEADER}\nclaude() {{ cross wrap -- claude "$@"; }}\n')
        mock_shell_rc.return_value = shell_rc

        inputs = iter(["none", "N", "N", "Y", "Y", "Y"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: "",
            print_fn=output.append,
        )

        assert result["shell_wrappers_installed"] is True
        full_output = "\n".join(str(o) for o in output)
        assert "already present" in full_output

    @patch("cross.setup._detect_shell_rc")
    @patch("cross.setup._detect_agents", return_value=["claude"])
    def test_shell_wrappers_declined(self, mock_agents, mock_shell_rc, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"
        shell_rc = tmp_path / ".zshrc"
        shell_rc.write_text("# existing\n")
        mock_shell_rc.return_value = shell_rc

        inputs = iter(["none", "N", "N", "n", "Y", "Y"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: "",
            print_fn=output.append,
        )

        assert result["shell_wrappers_installed"] is False
        assert shell_rc.read_text() == "# existing\n"

    @patch("cross.setup._detect_shell_rc", return_value=None)
    @patch("cross.setup._detect_agents", return_value=["claude"])
    def test_shell_rc_not_detected(self, mock_agents, mock_shell_rc, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"

        inputs = iter(["none", "N", "N", "Y", "Y", "Y"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: "",
            print_fn=output.append,
        )

        assert result["shell_wrappers_installed"] is False
        full_output = "\n".join(str(o) for o in output)
        assert "Could not detect shell rc file" in full_output


@patch("cross.setup.sys")
class TestRunSetupDefaultRules:
    @patch("cross.setup._detect_agents", return_value=[])
    def test_no_default_yaml_copied(self, mock_agents, mock_sys, tmp_path):
        """Default rules are loaded from the package, not copied to rules.d."""
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"

        inputs = iter(["none", "N", "N", "Y"])

        output = []
        run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: "",
            print_fn=output.append,
        )

        rules_file = cross_dir / "rules.d" / "default.yaml"
        assert not rules_file.exists()

    @patch("cross.setup._detect_agents", return_value=[])
    def test_stale_default_yaml_removed(self, mock_agents, mock_sys, tmp_path):
        """Running setup removes stale default.yaml from prior installations."""
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"
        rules_dir = cross_dir / "rules.d"
        rules_dir.mkdir(parents=True)
        stale = rules_dir / "default.yaml"
        stale.write_text("# old defaults\n")

        inputs = iter(["none", "N", "N", "Y"])

        output = []
        run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: "",
            print_fn=output.append,
        )

        assert not stale.exists()
        full_output = "\n".join(output)
        assert "stale" in full_output.lower()


@patch("cross.setup.sys")
class TestRunSetupCustomModel:
    """Test entering an arbitrary provider/model string."""

    @patch("cross.setup._detect_agents", return_value=[])
    def test_custom_model_string(self, mock_agents, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"

        inputs = iter(["google/gemini-2.5-flash", "none", "", "", "none", "N", "N", "Y"])
        secrets = iter(["my-key"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: next(secrets),
            print_fn=output.append,
        )

        assert result["model"] == "google/gemini-2.5-flash"
        assert result["llm_enabled"] is True
        env_content = (cross_dir / ".env").read_text()
        assert "CROSS_LLM_GATE_MODEL=google/gemini-2.5-flash" in env_content


@patch("cross.setup.sys")
class TestRunSetupClaudeModel:
    """Test entering bare 'claude' (cli/claude provider)."""

    @patch("cross.setup._detect_agents", return_value=[])
    def test_bare_claude_becomes_cli_claude(self, mock_agents, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"

        # "claude" → cli/claude, "none" gate backup, "" sentinel (same),
        # "" interval, "none" sentinel backup, N email, N slack
        inputs = iter(["claude", "none", "", "", "none", "N", "N", "Y"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: "",
            print_fn=output.append,
        )

        assert result["model"] == "anthropic/claude-code/claude-sonnet-4-6"
        assert result["llm_enabled"] is True
        env_content = (cross_dir / ".env").read_text()
        assert "CROSS_LLM_GATE_MODEL=anthropic/claude-code/claude-sonnet-4-6" in env_content
        assert "CROSS_LLM_SENTINEL_MODEL=anthropic/claude-code/claude-opus-4-6" in env_content
        # cli provider should not write API key
        assert "API_KEY" not in env_content

    @patch("cross.setup._detect_agents", return_value=[])
    def test_explicit_cli_claude(self, mock_agents, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"

        # "cli/claude" gate, "none" gate backup, "" sentinel (same),
        # "" interval, "none" sentinel backup, N email, N slack
        inputs = iter(["cli/claude", "none", "", "", "none", "N", "N", "Y"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: "",
            print_fn=output.append,
        )

        assert result["model"] == "anthropic/claude-code/claude-sonnet-4-6"
        assert result["llm_enabled"] is True
        env_content = (cross_dir / ".env").read_text()
        assert "CROSS_LLM_GATE_MODEL=anthropic/claude-code/claude-sonnet-4-6" in env_content


@patch("cross.setup.sys")
class TestRunSetupSummaryOutput:
    @patch("cross.setup._detect_agents", return_value=["claude"])
    @patch("cross.setup._detect_shell_rc", return_value=None)
    def test_summary_includes_key_info(self, mock_shell_rc, mock_agents, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"

        inputs = iter(["none", "N", "N", "n", "Y", "Y"])

        output = []
        run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: "",
            print_fn=output.append,
        )

        full_output = "\n".join(str(o) for o in output)
        assert "Setup complete!" in full_output
        assert "cross daemon" in full_output
        assert "Dashboard" in full_output or "dashboard" in full_output


class TestInstallClaudeCodeHook:
    """Tests for _install_claude_code_hook and _uninstall_claude_code_hook."""

    def test_installs_hook_in_empty_settings(self, tmp_path):
        from cross.setup import _install_claude_code_hook

        settings_file = tmp_path / "settings.json"
        settings_file.write_text("{}")
        output = []
        with patch("cross.setup._CLAUDE_SETTINGS", settings_file):
            result = _install_claude_code_hook(output.append)

        assert result is True
        import json

        settings = json.loads(settings_file.read_text())
        hooks = settings["hooks"]["PreToolUse"]
        assert len(hooks) == 1
        assert "claude_code_hook.py" in hooks[0]["hooks"][0]["command"]

    def test_preserves_existing_settings(self, tmp_path):
        from cross.setup import _install_claude_code_hook

        settings_file = tmp_path / "settings.json"
        import json

        settings_file.write_text(json.dumps({"env": {"FOO": "bar"}, "permissions": {"allow": ["Read"]}}))
        output = []
        with patch("cross.setup._CLAUDE_SETTINGS", settings_file):
            _install_claude_code_hook(output.append)

        settings = json.loads(settings_file.read_text())
        assert settings["env"]["FOO"] == "bar"
        assert settings["permissions"]["allow"] == ["Read"]

    def test_skips_if_already_installed(self, tmp_path):
        from cross.setup import _install_claude_code_hook

        settings_file = tmp_path / "settings.json"
        import json

        existing = {
            "hooks": {
                "PreToolUse": [
                    {"matcher": "", "hooks": [{"type": "command", "command": "python3 /path/to/claude_code_hook.py"}]}
                ]
            }
        }
        settings_file.write_text(json.dumps(existing))
        output = []
        with patch("cross.setup._CLAUDE_SETTINGS", settings_file):
            result = _install_claude_code_hook(output.append)

        assert result is True
        assert any("already installed" in s for s in output)

    def test_creates_settings_dir_if_missing(self, tmp_path):
        from cross.setup import _install_claude_code_hook

        settings_file = tmp_path / "subdir" / "settings.json"
        output = []
        with patch("cross.setup._CLAUDE_SETTINGS", settings_file):
            result = _install_claude_code_hook(output.append)

        assert result is True
        assert settings_file.exists()

    def test_uninstall_removes_hook(self, tmp_path):
        from cross.setup import _uninstall_claude_code_hook

        settings_file = tmp_path / "settings.json"
        import json

        existing = {
            "hooks": {
                "PreToolUse": [
                    {"matcher": "", "hooks": [{"type": "command", "command": "python3 /path/to/claude_code_hook.py"}]}
                ]
            }
        }
        settings_file.write_text(json.dumps(existing))
        output = []
        with patch("cross.setup._CLAUDE_SETTINGS", settings_file):
            result = _uninstall_claude_code_hook(output.append)

        assert result is True
        settings = json.loads(settings_file.read_text())
        assert "hooks" not in settings  # Cleaned up empty hooks

    def test_uninstall_preserves_other_hooks(self, tmp_path):
        from cross.setup import _uninstall_claude_code_hook

        settings_file = tmp_path / "settings.json"
        import json

        existing = {
            "hooks": {
                "PreToolUse": [
                    {"matcher": "Bash", "hooks": [{"type": "command", "command": "other_hook.sh"}]},
                    {"matcher": "", "hooks": [{"type": "command", "command": "python3 /path/to/claude_code_hook.py"}]},
                ]
            }
        }
        settings_file.write_text(json.dumps(existing))
        output = []
        with patch("cross.setup._CLAUDE_SETTINGS", settings_file):
            _uninstall_claude_code_hook(output.append)

        settings = json.loads(settings_file.read_text())
        assert len(settings["hooks"]["PreToolUse"]) == 1
        assert "other_hook" in settings["hooks"]["PreToolUse"][0]["hooks"][0]["command"]


class TestClaudeCodeHookSetupStep:
    """Test the setup wizard step for Claude Desktop hook installation."""

    @patch("cross.setup._install_gate_hook", return_value=True)
    @patch("cross.setup._is_claude_desktop_installed", return_value=True)
    @patch("cross.setup._detect_agents", return_value=[])
    @patch("cross.setup.sys")
    def test_darwin_with_desktop_app_prompts(self, mock_sys, mock_agents, mock_desktop, mock_install, tmp_path):
        mock_sys.platform = "darwin"
        cross_dir = tmp_path / ".cross"

        # none LLM, N notifications, N email, N slack, Y hook, N autostart, Y auto-update
        inputs = iter(["none", "N", "N", "N", "Y", "N", "Y"])
        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: "",
            print_fn=output.append,
        )

        mock_install.assert_called_once()
        assert result.get("claude_code_hook_installed") is True


@patch("cross.setup.sys")
class TestRunSetupEmail:
    @patch("cross.setup._detect_agents", return_value=[])
    def test_email_configured(self, mock_agents, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"

        # none LLM, y email, from, to, smtp host, port, username, password, n imap, N slack, Y auto-update
        inputs = iter(
            [
                "none",
                "y",
                "cross@example.com",
                "admin@example.com",
                "smtp.gmail.com",
                "587",
                "N",
                "N",
                "Y",
            ]
        )
        secrets = iter(["cross@example.com", "secret123"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: next(secrets),
            print_fn=output.append,
        )

        assert result["email_configured"] is True
        env_content = (cross_dir / ".env").read_text()
        assert "CROSS_EMAIL_FROM=cross@example.com" in env_content
        assert "CROSS_EMAIL_TO=admin@example.com" in env_content
        assert "CROSS_EMAIL_SMTP_HOST=smtp.gmail.com" in env_content
        assert "CROSS_EMAIL_SMTP_PORT=587" in env_content
        assert "CROSS_EMAIL_SMTP_USERNAME=cross@example.com" in env_content
        assert "CROSS_EMAIL_SMTP_PASSWORD=secret123" in env_content

    @patch("cross.setup._detect_agents", return_value=[])
    def test_email_with_imap(self, mock_agents, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"

        inputs = iter(
            [
                "none",
                "y",
                "cross@example.com",
                "admin@example.com",
                "smtp.gmail.com",
                "",
                "y",
                "imap.gmail.com",
                "N",
                "Y",
            ]
        )
        secrets = iter(["cross@example.com", "secret123"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: next(secrets),
            print_fn=output.append,
        )

        assert result["email_configured"] is True
        env_content = (cross_dir / ".env").read_text()
        assert "CROSS_EMAIL_IMAP_HOST=imap.gmail.com" in env_content

    @patch("cross.setup._detect_agents", return_value=[])
    def test_email_declined(self, mock_agents, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"

        inputs = iter(["none", "N", "N", "Y"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: "",
            print_fn=output.append,
        )

        assert result["email_configured"] is False
        env_content = (cross_dir / ".env").read_text()
        assert "EMAIL" not in env_content

    @patch("cross.setup._detect_agents", return_value=[])
    def test_email_empty_addresses_skipped(self, mock_agents, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"

        inputs = iter(["none", "y", "", "", "N", "Y"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: "",
            print_fn=output.append,
        )

        assert result["email_configured"] is False
        env_content = (cross_dir / ".env").read_text()
        assert "EMAIL" not in env_content


@patch("cross.setup.sys")
class TestRunSetupEmailPreservation:
    @patch("cross.setup._detect_agents", return_value=[])
    def test_existing_email_preserved_on_rerun(self, mock_agents, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"
        cross_dir.mkdir(parents=True)
        (cross_dir / ".env").write_text("CROSS_EMAIL_FROM=cross@example.com\nCROSS_EMAIL_TO=admin@example.com\n")

        # none LLM, Y keep email, N slack, Y auto-update
        inputs = iter(["none", "Y", "N", "Y"])

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: "",
            print_fn=output.append,
        )

        assert result["email_configured"] is True
        env_content = (cross_dir / ".env").read_text()
        assert "CROSS_EMAIL_FROM=cross@example.com" in env_content
        assert "CROSS_EMAIL_TO=admin@example.com" in env_content

    @patch("cross.setup._detect_agents", return_value=[])
    def test_existing_email_replaced_on_rerun(self, mock_agents, mock_sys, tmp_path):
        mock_sys.platform = "linux"
        cross_dir = tmp_path / ".cross"
        cross_dir.mkdir(parents=True)
        (cross_dir / ".env").write_text("CROSS_EMAIL_FROM=old@example.com\nCROSS_EMAIL_TO=old-admin@example.com\n")

        # none LLM, n don't keep email, y configure new, addresses, smtp, no imap, N slack, Y auto-update
        inputs = iter(
            [
                "none",
                "n",
                "y",
                "new@example.com",
                "new-admin@example.com",
                "",
                "",
                "N",
                "N",
                "Y",
            ]
        )

        output = []
        result = run_setup(
            cross_dir=cross_dir,
            input_fn=lambda p: next(inputs),
            getpass_fn=lambda p: "",
            print_fn=output.append,
        )

        assert result["email_configured"] is True
        env_content = (cross_dir / ".env").read_text()
        assert "CROSS_EMAIL_FROM=new@example.com" in env_content
        assert "CROSS_EMAIL_TO=new-admin@example.com" in env_content


class TestBuildEnvLinesEmail:
    def test_email_config_included(self):
        lines = _build_env_lines(
            gate_model=None,
            gate_api_key=None,
            llm_enabled=False,
            slack_bot_token=None,
            slack_app_token=None,
            email_from="cross@example.com",
            email_to="admin@example.com",
            email_smtp_host="smtp.gmail.com",
            email_smtp_port=587,
            email_smtp_username="cross@example.com",
            email_smtp_password="secret",
            email_imap_host="imap.gmail.com",
        )
        text = "\n".join(lines)
        assert "CROSS_EMAIL_FROM=cross@example.com" in text
        assert "CROSS_EMAIL_TO=admin@example.com" in text
        assert "CROSS_EMAIL_SMTP_HOST=smtp.gmail.com" in text
        assert "CROSS_EMAIL_SMTP_PORT=587" in text
        assert "CROSS_EMAIL_SMTP_USERNAME=cross@example.com" in text
        assert "CROSS_EMAIL_SMTP_PASSWORD=secret" in text
        assert "CROSS_EMAIL_IMAP_HOST=imap.gmail.com" in text

    def test_email_not_included_when_empty(self):
        lines = _build_env_lines(
            gate_model=None,
            gate_api_key=None,
            llm_enabled=False,
            slack_bot_token=None,
            slack_app_token=None,
        )
        text = "\n".join(lines)
        assert "EMAIL" not in text

    def test_email_minimal(self):
        lines = _build_env_lines(
            gate_model=None,
            gate_api_key=None,
            llm_enabled=False,
            slack_bot_token=None,
            slack_app_token=None,
            email_from="cross@example.com",
            email_to="admin@example.com",
        )
        text = "\n".join(lines)
        assert "CROSS_EMAIL_FROM=cross@example.com" in text
        assert "CROSS_EMAIL_TO=admin@example.com" in text
        assert "SMTP_HOST" not in text
        assert "IMAP" not in text


class TestCLISetupRouting:
    @patch("cross.setup.run_setup")
    def test_setup_command_calls_run_setup(self, mock_run_setup):
        from cross.cli import main

        with patch("sys.argv", ["cross", "setup"]):
            main()
        mock_run_setup.assert_called_once()
