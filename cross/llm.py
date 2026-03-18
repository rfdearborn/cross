"""LLM client — provider-agnostic completion interface.

Supports provider/model naming (e.g. "google/gemini-3-flash-preview", "openai/gpt-4o").
Provider prefix determines API format:
  - cli/*              → CLI subprocess (e.g. cli/claude uses `claude -p`, no API key needed)
  - anthropic/*        → Anthropic Messages API
  - openai/*           → OpenAI Chat Completions API
  - google/*           → Google Gemini (OpenAI-compatible endpoint)
  - ollama/*           → Ollama (OpenAI-compatible endpoint, no API key needed)

Each caller supplies its own api_key, base_url, model. Key resolution:
  explicit api_key → provider env var fallback (e.g. ANTHROPIC_API_KEY, GOOGLE_API_KEY).
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field

import httpx

logger = logging.getLogger("cross.llm")

# Default base URLs per provider
_PROVIDER_BASE_URLS: dict[str, str] = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai",
    "ollama": "http://localhost:11434/v1",
}

# Environment variable fallbacks per provider (tried in order)
_PROVIDER_KEY_ENV_VARS: dict[str, list[str]] = {
    "anthropic": ["ANTHROPIC_API_KEY"],
    "openai": ["OPENAI_API_KEY"],
    "google": ["GOOGLE_API_KEY"],
    "ollama": [],  # no key needed
    "cli": [],  # CLI handles its own auth
}

# Providers that use the OpenAI Chat Completions API format
_OPENAI_COMPATIBLE_PROVIDERS: set[str] = {"openai", "google", "ollama"}

# Lazy singleton httpx client
_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(timeout=60.0)
    return _client


async def close_client() -> None:
    """Shut down the shared httpx client."""
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None


@dataclass
class LLMConfig:
    """Per-component LLM configuration."""

    model: str = ""  # provider/model format, e.g. "anthropic/claude-haiku-4-5"
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.0
    max_tokens: int = 256
    reasoning: str = ""  # "", "low", "medium", "high"

    # Parsed from model string (set in __post_init__)
    provider: str = field(default="", init=False)
    model_id: str = field(default="", init=False)

    def __post_init__(self):
        self.provider, self.model_id = parse_model_ref(self.model)


def parse_model_ref(model: str) -> tuple[str, str]:
    """Parse 'provider/model' into (provider, model_id). Default provider: anthropic.

    Special case: 'anthropic/claude-code/model' routes through the CLI
    provider (claude -p --model) using the user's Claude subscription.
    """
    if not model:
        return ("", "")
    # anthropic/claude-code/model → cli provider with model spec
    if model.lower().startswith("anthropic/claude-code/"):
        model_name = model.split("/", 2)[2].strip()
        return ("cli", model_name)
    parts = model.split("/", 1)
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        return (parts[0].strip().lower(), parts[1].strip())
    # No slash — assume anthropic
    return ("anthropic", model.strip())


def resolve_api_key(config: LLMConfig) -> str | None:
    """Resolve API key: explicit → provider env var fallback."""
    if config.api_key:
        return config.api_key

    env_vars = _PROVIDER_KEY_ENV_VARS.get(config.provider, [])
    for var in env_vars:
        value = os.environ.get(var, "").strip()
        if value:
            return value

    return None


def resolve_base_url(config: LLMConfig) -> str:
    """Resolve base URL: explicit → provider default."""
    if config.base_url:
        return config.base_url.rstrip("/")
    return _PROVIDER_BASE_URLS.get(config.provider, "")


def _reasoning_to_anthropic(level: str) -> dict:
    """Map reasoning level to Anthropic extended thinking parameters."""
    budgets = {"low": 2048, "medium": 8192, "high": 32768}
    budget = budgets.get(level, 0)
    if budget:
        return {"thinking": {"type": "enabled", "budget_tokens": budget}}
    return {}


def _reasoning_to_openai(level: str) -> dict:
    """Map reasoning level to OpenAI reasoning_effort parameter."""
    if level in ("low", "medium", "high"):
        return {"reasoning_effort": level}
    return {}


async def complete(
    config: LLMConfig,
    system: str,
    messages: list[dict],
    timeout_s: float = 30.0,
) -> str | None:
    """Send a completion request. Returns the text response, or None on error."""
    # CLI provider uses subprocess — no API key or base URL needed
    if config.provider == "cli":
        return await _complete_cli(config, system, messages, timeout_s)

    api_key = resolve_api_key(config)
    if not api_key:
        if config.provider == "ollama":
            # Ollama doesn't require an API key — use a placeholder
            api_key = "ollama"
        else:
            logger.warning(f"No API key available for provider '{config.provider}'")
            return None

    base_url = resolve_base_url(config)
    if not base_url:
        logger.warning(f"No base URL for provider '{config.provider}'")
        return None

    if config.provider == "anthropic":
        return await _complete_anthropic(config, api_key, base_url, system, messages, timeout_s)
    elif config.provider in _OPENAI_COMPATIBLE_PROVIDERS:
        return await _complete_openai(config, api_key, base_url, system, messages, timeout_s)
    else:
        logger.warning(f"Unsupported provider: '{config.provider}'")
        return None


async def _complete_anthropic(
    config: LLMConfig,
    api_key: str,
    base_url: str,
    system: str,
    messages: list[dict],
    timeout_s: float,
) -> str | None:
    """Call the Anthropic Messages API."""
    url = f"{base_url}/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    body: dict = {
        "model": config.model_id,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "messages": messages,
    }
    if system:
        body["system"] = system

    # Extended thinking
    reasoning_params = _reasoning_to_anthropic(config.reasoning)
    if reasoning_params:
        body.update(reasoning_params)
        # Extended thinking requires temperature=1 on Anthropic
        body["temperature"] = 1.0
        # max_tokens must exceed budget_tokens
        budget = reasoning_params["thinking"]["budget_tokens"]
        if body["max_tokens"] <= budget:
            body["max_tokens"] = budget + config.max_tokens

    client = _get_client()
    try:
        resp = await client.post(url, headers=headers, json=body, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()

        # Extract text from content blocks
        for block in data.get("content", []):
            if block.get("type") == "text":
                return block["text"]

        logger.warning(f"Anthropic response had no text block: {data}")
        return None
    except httpx.HTTPStatusError as e:
        logger.warning(f"Anthropic API error {e.response.status_code}: {e.response.text[:200]}")
        return None
    except Exception as e:
        logger.warning(f"Anthropic request failed: {e}")
        return None


async def _complete_openai(
    config: LLMConfig,
    api_key: str,
    base_url: str,
    system: str,
    messages: list[dict],
    timeout_s: float,
) -> str | None:
    """Call the OpenAI Chat Completions API."""
    url = f"{base_url}/chat/completions"
    headers = {
        "authorization": f"Bearer {api_key}",
        "content-type": "application/json",
    }

    # Build messages list with system message prepended
    oai_messages = []
    if system:
        oai_messages.append({"role": "system", "content": system})
    oai_messages.extend(messages)

    body: dict = {
        "model": config.model_id,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "messages": oai_messages,
    }

    # Reasoning effort
    reasoning_params = _reasoning_to_openai(config.reasoning)
    if reasoning_params:
        body.update(reasoning_params)

    client = _get_client()
    try:
        resp = await client.post(url, headers=headers, json=body, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()

        choices = data.get("choices", [])
        if choices:
            return choices[0].get("message", {}).get("content")

        logger.warning(f"OpenAI response had no choices: {data}")
        return None
    except httpx.HTTPStatusError as e:
        logger.warning(f"OpenAI API error {e.response.status_code}: {e.response.text[:200]}")
        return None
    except Exception as e:
        logger.warning(f"OpenAI request failed: {e}")
        return None


def _build_cli_prompt(system: str, messages: list[dict]) -> str:
    """Flatten system prompt + messages into a single prompt string for CLI mode."""
    parts: list[str] = []
    if system:
        parts.append(f"System: {system}")
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(f"{role.capitalize()}: {content}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(f"{role.capitalize()}: {block['text']}")
    return "\n\n".join(parts)


async def _complete_cli(
    config: LLMConfig,
    system: str,
    messages: list[dict],
    timeout_s: float,
) -> str | None:
    """Run a CLI command (e.g. ``claude -p``) as a subprocess and return its stdout.

    The subprocess environment is sanitised: ``CROSS_*`` env vars are removed
    and ``ANTHROPIC_BASE_URL`` is set to the real Anthropic API to avoid
    recursive proxying through cross.
    """
    prompt = _build_cli_prompt(system, messages)

    # model_id is either "claude" (bare, uses CC default) or a model name
    # like "claude-sonnet-4-6" (from anthropic/claude-code/claude-sonnet-4-6)
    model_id = config.model_id
    cmd = "claude"
    model_flag = model_id if model_id != "claude" else None

    # Build a clean env: strip CROSS_* vars, point at real Anthropic API
    # to avoid recursive proxying through cross.
    #
    # By default, also strip ANTHROPIC_API_KEY so claude -p uses the
    # user's Max/Pro subscription instead of making billed API calls.
    # Claude Code headless mode has a bug where it uses the API key even
    # when "use custom API key" is set to false in config.
    # See: https://github.com/anthropics/claude-code/issues/33996
    from cross.config import settings as cross_settings

    strip_keys = {"ANTHROPIC_API_KEY"} if cross_settings.cli_strip_anthropic_api_key else set()
    env = {k: v for k, v in os.environ.items() if not k.startswith("CROSS_") and k not in strip_keys}
    env["ANTHROPIC_BASE_URL"] = "https://api.anthropic.com"

    model_desc = f" --model {model_flag}" if model_flag else ""
    logger.debug(f"CLI invoke: {cmd} -p{model_desc} (prompt length {len(prompt)})")

    cli_args = [cmd, "-p", prompt]
    if model_flag:
        cli_args = [cmd, "--model", model_flag, "-p", prompt]

    try:
        proc = await asyncio.create_subprocess_exec(
            *cli_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)

        if proc.returncode != 0:
            err_text = stderr.decode(errors="replace").strip()[:200] if stderr else "(no stderr)"
            logger.warning(f"CLI '{cmd}' exited with code {proc.returncode}: {err_text}")
            return None

        text = stdout.decode(errors="replace").strip() if stdout else ""
        if not text:
            logger.warning(f"CLI '{cmd}' returned empty output")
            return None

        return text

    except FileNotFoundError:
        logger.warning(f"CLI command not found: '{cmd}'")
        return None
    except asyncio.TimeoutError:
        logger.warning(f"CLI '{cmd}' timed out after {timeout_s}s")
        try:
            proc.kill()  # type: ignore[possibly-undefined]
        except Exception:
            pass
        return None
    except Exception as e:
        logger.warning(f"CLI '{cmd}' failed: {e}")
        return None
