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

    # Email
    email_from: str = ""  # sender address (e.g. cross@example.com)
    email_to: str = ""  # recipient address for notifications
    email_smtp_host: str = "localhost"
    email_smtp_port: int = 587
    email_smtp_ssl: bool = False  # use SMTP_SSL (port 465)
    email_smtp_starttls: bool = True  # use STARTTLS (port 587)
    email_smtp_username: str = ""
    email_smtp_password: str = ""
    email_imap_host: str = ""  # IMAP host for inbound replies (empty = outbound only)
    email_imap_port: int = 993
    email_imap_ssl: bool = True
    email_imap_username: str = ""  # falls back to smtp_username
    email_imap_password: str = ""  # falls back to smtp_password
    email_imap_poll_interval: int = 30  # seconds between IMAP checks

    # Slack
    slack_bot_token: str = ""
    slack_app_token: str = ""
    slack_channel_base: str = "cross"  # base channel name
    slack_channel_append_project: bool = False  # append project name to channel
    slack_channel_append_user: bool = True  # append Slack username to channel

    # Custom instructions (included in gate + sentinel prompts)
    custom_instructions_file: str = "~/.cross/instructions.md"

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
    llm_gate_context_tools: int = 3  # number of recent tool calls to include in LLM review
    llm_gate_context_turns: int = 5  # number of recent conversation turns to include in LLM review
    llm_gate_context_chars_per_turn: int = 300  # max characters per conversation turn
    llm_gate_context_intent_chars: int = 500  # max characters for user intent extraction
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
