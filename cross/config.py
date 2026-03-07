from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_base_url: str = "https://api.anthropic.com"
    listen_host: str = "0.0.0.0"
    listen_port: int = 8080
    log_file: str = "data/cross.log"

    # Slack
    slack_bot_token: str = ""
    slack_app_token: str = ""
    slack_channel_prefix: str = ""  # optional org prefix

    model_config = {"env_prefix": "CROSS_", "env_file": ".env"}


settings = Settings()
