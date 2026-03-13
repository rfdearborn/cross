from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_base_url: str = "https://api.anthropic.com"
    listen_host: str = "127.0.0.1"
    listen_port: int = 8080
    log_file: str = "data/cross.log"

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
    llm_gate_model: str = "anthropic/claude-haiku-4-5"
    llm_gate_api_key: str = ""
    llm_gate_base_url: str = ""
    llm_gate_temperature: float = 0.0
    llm_gate_max_tokens: int = 256
    llm_gate_reasoning: str = ""  # off by default for speed
    llm_gate_timeout_ms: float = 30000
    llm_gate_threshold: str = "block"  # min denylist action to trigger LLM review
    llm_gate_context_tools: int = 3  # number of recent tool calls to include in LLM review
    llm_gate_justification: bool = False  # include explanation after verdict (costs tokens)
    gate_approval_timeout: int = 300  # seconds to wait for human approval on escalation

    # LLM sentinel — async periodic review of all activity
    llm_sentinel_enabled: bool = True
    llm_sentinel_model: str = "anthropic/claude-sonnet-4-6"
    llm_sentinel_api_key: str = ""
    llm_sentinel_base_url: str = ""
    llm_sentinel_temperature: float = 0.0
    llm_sentinel_max_tokens: int = 1024
    llm_sentinel_reasoning: str = "medium"
    llm_sentinel_interval_seconds: int = 60

    model_config = {"env_prefix": "CROSS_", "env_file": ".env"}


settings = Settings()
