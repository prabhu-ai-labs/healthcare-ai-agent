from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings, read from environment variables (prefix ``HCA_``) or a ``.env`` file.

    The Anthropic API key is read by the SDK itself from ``ANTHROPIC_API_KEY``.
    """

    model_config = SettingsConfigDict(env_prefix="HCA_", env_file=".env", extra="ignore")

    model: str = "claude-opus-5-5"
    chat_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    extraction_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    max_tokens: int = 16000
    max_tool_iterations: int = 10
    # Server-side refusal fallback: re-runs a declined request on Anthropic's
    # recommended fallback model inside the same API call.
    enable_refusal_fallback: bool = True


@lru_cache
def get_settings() -> Settings:
    return Settings()
