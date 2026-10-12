"""Shared Claude client construction and request options."""

from typing import Any

import anthropic

from .config import Settings

FALLBACK_BETA = "server-side-fallback-2026-07-01"


def build_client() -> anthropic.Anthropic:
    # Credentials resolve from the environment (ANTHROPIC_API_KEY, or an `ant auth login` profile).
    return anthropic.Anthropic()


def request_options(settings: Settings, effort: str) -> dict[str, Any]:
    """Options common to every Messages call made by the agent."""
    options: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "output_config": {"effort": effort},
    }
    if settings.enable_refusal_fallback:
        options["betas"] = [FALLBACK_BETA]
        options["fallbacks"] = "default"
    return options


class RefusalError(RuntimeError):
    """Raised when Claude (and any fallback model) declines a request."""

    def __init__(self, category: str | None, explanation: str | None):
        self.category = category
        self.explanation = explanation
        super().__init__(f"Request declined (category={category}): {explanation}")


def raise_if_refused(response: Any) -> None:
    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise RefusalError(
            getattr(details, "category", None), getattr(details, "explanation", None)
        )
