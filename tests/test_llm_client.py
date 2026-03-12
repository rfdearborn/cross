"""Tests for cross.llm — LLM client, provider routing, key resolution."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from cross.llm import (
    LLMConfig,
    _reasoning_to_anthropic,
    _reasoning_to_openai,
    close_client,
    complete,
    parse_model_ref,
    resolve_api_key,
    resolve_base_url,
)

# --- parse_model_ref ---


class TestParseModelRef:
    def test_anthropic_model(self):
        assert parse_model_ref("anthropic/claude-haiku-4-5") == ("anthropic", "claude-haiku-4-5")

    def test_openai_model(self):
        assert parse_model_ref("openai/gpt-4o") == ("openai", "gpt-4o")

    def test_no_slash_defaults_anthropic(self):
        assert parse_model_ref("claude-haiku-4-5") == ("anthropic", "claude-haiku-4-5")

    def test_empty_string(self):
        assert parse_model_ref("") == ("", "")

    def test_provider_lowercased(self):
        assert parse_model_ref("Anthropic/claude-haiku-4-5") == ("anthropic", "claude-haiku-4-5")

    def test_strips_whitespace(self):
        assert parse_model_ref("  openai / gpt-4o  ") == ("openai", "gpt-4o")

    def test_slash_only(self):
        # Empty provider and model — falls back to no-slash path
        assert parse_model_ref("/") == ("anthropic", "/")

    def test_multiple_slashes_splits_on_first(self):
        assert parse_model_ref("openrouter/anthropic/claude-sonnet-4-6") == (
            "openrouter",
            "anthropic/claude-sonnet-4-6",
        )


# --- LLMConfig ---


class TestLLMConfig:
    def test_parses_model_on_init(self):
        cfg = LLMConfig(model="openai/gpt-4o")
        assert cfg.provider == "openai"
        assert cfg.model_id == "gpt-4o"

    def test_defaults(self):
        cfg = LLMConfig()
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 256
        assert cfg.reasoning == ""


# --- resolve_api_key ---


class TestResolveApiKey:
    def test_explicit_key(self):
        cfg = LLMConfig(model="anthropic/claude-haiku-4-5", api_key="sk-test")
        assert resolve_api_key(cfg) == "sk-test"

    def test_anthropic_env_fallback(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
        cfg = LLMConfig(model="anthropic/claude-haiku-4-5")
        assert resolve_api_key(cfg) == "sk-from-env"

    def test_openai_env_fallback(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env")
        cfg = LLMConfig(model="openai/gpt-4o")
        assert resolve_api_key(cfg) == "sk-openai-env"

    def test_no_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        cfg = LLMConfig(model="anthropic/claude-haiku-4-5")
        assert resolve_api_key(cfg) is None

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-from-env")
        cfg = LLMConfig(model="anthropic/claude-haiku-4-5", api_key="sk-explicit")
        assert resolve_api_key(cfg) == "sk-explicit"


# --- resolve_base_url ---


class TestResolveBaseUrl:
    def test_explicit_url(self):
        cfg = LLMConfig(model="anthropic/claude-haiku-4-5", base_url="https://custom.api.com/")
        assert resolve_base_url(cfg) == "https://custom.api.com"

    def test_anthropic_default(self):
        cfg = LLMConfig(model="anthropic/claude-haiku-4-5")
        assert resolve_base_url(cfg) == "https://api.anthropic.com"

    def test_openai_default(self):
        cfg = LLMConfig(model="openai/gpt-4o")
        assert resolve_base_url(cfg) == "https://api.openai.com"

    def test_unknown_provider_empty(self):
        cfg = LLMConfig(model="custom/some-model")
        assert resolve_base_url(cfg) == ""


# --- reasoning helpers ---


class TestReasoningMapping:
    def test_anthropic_off(self):
        assert _reasoning_to_anthropic("") == {}

    def test_anthropic_low(self):
        result = _reasoning_to_anthropic("low")
        assert result["thinking"]["budget_tokens"] == 2048

    def test_anthropic_medium(self):
        result = _reasoning_to_anthropic("medium")
        assert result["thinking"]["budget_tokens"] == 8192

    def test_anthropic_high(self):
        result = _reasoning_to_anthropic("high")
        assert result["thinking"]["budget_tokens"] == 32768

    def test_openai_off(self):
        assert _reasoning_to_openai("") == {}

    def test_openai_low(self):
        assert _reasoning_to_openai("low") == {"reasoning_effort": "low"}

    def test_openai_medium(self):
        assert _reasoning_to_openai("medium") == {"reasoning_effort": "medium"}

    def test_openai_high(self):
        assert _reasoning_to_openai("high") == {"reasoning_effort": "high"}


# --- complete() ---


def _mock_response(status_code: int, json_data: dict | None = None, text: str = "") -> httpx.Response:
    """Create a mock httpx.Response with a request set (needed for raise_for_status)."""
    request = httpx.Request("POST", "https://test.com")
    if json_data is not None:
        return httpx.Response(status_code, json=json_data, request=request)
    return httpx.Response(status_code, text=text, request=request)


class TestComplete:
    @pytest.fixture(autouse=True)
    async def _reset_client(self):
        """Ensure clean client state between tests."""
        yield
        await close_client()

    @pytest.mark.anyio
    async def test_no_api_key_returns_none(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        cfg = LLMConfig(model="anthropic/claude-haiku-4-5")
        result = await complete(cfg, system="test", messages=[{"role": "user", "content": "hi"}])
        assert result is None

    @pytest.mark.anyio
    async def test_unsupported_provider_returns_none(self):
        cfg = LLMConfig(model="custom/model", api_key="sk-test")
        result = await complete(cfg, system="test", messages=[{"role": "user", "content": "hi"}])
        assert result is None

    @pytest.mark.anyio
    async def test_anthropic_success(self):
        cfg = LLMConfig(model="anthropic/claude-haiku-4-5", api_key="sk-test")

        mock_response = _mock_response(200, {"content": [{"type": "text", "text": "VERDICT: ALLOW\nThis is safe."}]})
        mock_post = AsyncMock(return_value=mock_response)

        with patch("cross.llm._get_client") as mock_client:
            mock_client.return_value.post = mock_post
            result = await complete(cfg, system="Review this", messages=[{"role": "user", "content": "test"}])

        assert result == "VERDICT: ALLOW\nThis is safe."
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://api.anthropic.com/v1/messages"
        assert call_args[1]["headers"]["x-api-key"] == "sk-test"

    @pytest.mark.anyio
    async def test_openai_success(self):
        cfg = LLMConfig(model="openai/gpt-4o", api_key="sk-test")

        mock_response = _mock_response(200, {"choices": [{"message": {"content": "VERDICT: BLOCK\nDangerous."}}]})
        mock_post = AsyncMock(return_value=mock_response)

        with patch("cross.llm._get_client") as mock_client:
            mock_client.return_value.post = mock_post
            result = await complete(cfg, system="Review this", messages=[{"role": "user", "content": "test"}])

        assert result == "VERDICT: BLOCK\nDangerous."
        call_args = mock_post.call_args
        assert call_args[0][0] == "https://api.openai.com/v1/chat/completions"
        assert "Bearer sk-test" in call_args[1]["headers"]["authorization"]

    @pytest.mark.anyio
    async def test_anthropic_http_error(self):
        cfg = LLMConfig(model="anthropic/claude-haiku-4-5", api_key="sk-test")

        mock_response = _mock_response(429, text="Rate limited")
        mock_post = AsyncMock(return_value=mock_response)

        with patch("cross.llm._get_client") as mock_client:
            mock_client.return_value.post = mock_post
            result = await complete(cfg, system="test", messages=[{"role": "user", "content": "hi"}])

        assert result is None

    @pytest.mark.anyio
    async def test_anthropic_network_error(self):
        cfg = LLMConfig(model="anthropic/claude-haiku-4-5", api_key="sk-test")

        with patch("cross.llm._get_client") as mock_client:
            mock_client.return_value.post = AsyncMock(side_effect=httpx.ConnectError("Connection refused"))
            result = await complete(cfg, system="test", messages=[{"role": "user", "content": "hi"}])

        assert result is None

    @pytest.mark.anyio
    async def test_anthropic_no_text_block(self):
        cfg = LLMConfig(model="anthropic/claude-haiku-4-5", api_key="sk-test")

        mock_response = _mock_response(200, {"content": [{"type": "thinking", "text": "hmm"}]})
        mock_post = AsyncMock(return_value=mock_response)

        with patch("cross.llm._get_client") as mock_client:
            mock_client.return_value.post = mock_post
            result = await complete(cfg, system="test", messages=[{"role": "user", "content": "hi"}])

        assert result is None

    @pytest.mark.anyio
    async def test_reasoning_anthropic_sets_thinking(self):
        cfg = LLMConfig(model="anthropic/claude-haiku-4-5", api_key="sk-test", reasoning="medium")

        mock_response = _mock_response(200, {"content": [{"type": "text", "text": "ok"}]})
        mock_post = AsyncMock(return_value=mock_response)

        with patch("cross.llm._get_client") as mock_client:
            mock_client.return_value.post = mock_post
            await complete(cfg, system="test", messages=[{"role": "user", "content": "hi"}])

        body = mock_post.call_args[1]["json"]
        assert body["thinking"]["type"] == "enabled"
        assert body["thinking"]["budget_tokens"] == 8192
        assert body["temperature"] == 1.0  # required for extended thinking

    @pytest.mark.anyio
    async def test_reasoning_openai_sets_effort(self):
        cfg = LLMConfig(model="openai/gpt-4o", api_key="sk-test", reasoning="high")

        mock_response = _mock_response(200, {"choices": [{"message": {"content": "ok"}}]})
        mock_post = AsyncMock(return_value=mock_response)

        with patch("cross.llm._get_client") as mock_client:
            mock_client.return_value.post = mock_post
            await complete(cfg, system="test", messages=[{"role": "user", "content": "hi"}])

        body = mock_post.call_args[1]["json"]
        assert body["reasoning_effort"] == "high"

    @pytest.mark.anyio
    async def test_custom_base_url(self):
        cfg = LLMConfig(model="anthropic/claude-haiku-4-5", api_key="sk-test", base_url="https://my-proxy.com")

        mock_response = _mock_response(200, {"content": [{"type": "text", "text": "ok"}]})
        mock_post = AsyncMock(return_value=mock_response)

        with patch("cross.llm._get_client") as mock_client:
            mock_client.return_value.post = mock_post
            await complete(cfg, system="test", messages=[{"role": "user", "content": "hi"}])

        assert mock_post.call_args[0][0] == "https://my-proxy.com/v1/messages"

    @pytest.mark.anyio
    async def test_openai_system_message_prepended(self):
        cfg = LLMConfig(model="openai/gpt-4o", api_key="sk-test")

        mock_response = _mock_response(200, {"choices": [{"message": {"content": "ok"}}]})
        mock_post = AsyncMock(return_value=mock_response)

        with patch("cross.llm._get_client") as mock_client:
            mock_client.return_value.post = mock_post
            await complete(cfg, system="Be helpful", messages=[{"role": "user", "content": "hi"}])

        body = mock_post.call_args[1]["json"]
        assert body["messages"][0] == {"role": "system", "content": "Be helpful"}
        assert body["messages"][1] == {"role": "user", "content": "hi"}
