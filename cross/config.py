from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    config_dir: str = "~/.cross"
    anthropic_base_url: str = "https://api.anthropic.com"  # upstream API target the proxy forwards to
    listen_host: str = "127.0.0.1"
    listen_port: int = 2767
    log_file: str = "data/cross.log"  # relative to cwd
    cli_strip_anthropic_api_key: bool = True  # strip ANTHROPIC_API_KEY from claude -p env (workaround for CC#33996)

    # Auto-updating
    auto_update_enabled: bool = True
    auto_update_interval_hours: int = 24

    # Native desktop notifications (macOS, via terminal-notifier)
    native_notifications_enabled: bool = False

    # Slack
    slack_bot_token: str = ""
    slack_app_token: str = ""
    slack_channel_base: str = "cross"  # base channel name
    slack_channel_append_project: bool = False  # append project name to channel
    slack_channel_append_user: bool = True  # append Slack username to channel

    # Gating (denylist)
    gating_enabled: bool = True
    rules_dir: str = "~/.cross/rules.d"

    # LLM gate — synchronous review of denylist-flagged calls
    llm_gate_enabled: bool = True
    llm_gate_model: str = "anthropic/claude-code/claude-sonnet-4-6"
    llm_gate_api_key: str = ""  # optional; falls back to provider default
    llm_gate_base_url: str = ""  # optional; falls back to provider default
    llm_gate_temperature: float = 0.0
    llm_gate_max_tokens: int = 256
    llm_gate_reasoning: str = ""  # off by default for speed
    llm_gate_timeout_ms: float = 30000  # max wait for LLM response before falling back
    llm_gate_threshold: str = "escalate"  # min denylist action to trigger LLM review
    llm_gate_context_tools: int = 3  # number of recent tool calls to include in LLM review
    llm_gate_justification: bool = False  # include explanation after verdict (costs tokens)
    llm_gate_shadow: bool = False  # LLM decides but human makes the final call
    gate_approval_timeout: int = 300  # seconds to wait for human approval on escalation
    gate_max_retries: int = 3  # max retry attempts when a gate blocks a tool call

    # LLM sentinel — async periodic review of all activity
    llm_sentinel_enabled: bool = True
    llm_sentinel_model: str = "anthropic/claude-code/claude-opus-4-6"
    llm_sentinel_api_key: str = ""
    llm_sentinel_base_url: str = ""
    llm_sentinel_temperature: float = 0.0
    llm_sentinel_max_tokens: int = 1024
    llm_sentinel_reasoning: str = "medium"  # extended thinking level (sentinel benefits from deeper reasoning)
    llm_sentinel_interval_seconds: int = 60  # how often the sentinel reviews recent activity

    # Env file load order (earlier files take precedence)
    model_config = {
        "env_prefix": "CROSS_",
        "env_file": ("~/.cross/local.env", "~/.cross/.env"),
        "extra": "ignore",
    }


settings = Settings()
