"""Chat-model factory and API error handling shared by the agent modules."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables import Runnable

from src.config import Settings

logger = logging.getLogger("tunescout.llm")

_REQUEST_TIMEOUT_SECONDS = 90
_MAX_RETRIES = 3
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"  # automatic retries with backoff for transient API errors


class LLMError(Exception):
    """An LLM call failed. ``fatal`` errors (bad key, no quota) stop the whole run."""

    def __init__(self, message: str, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


def get_chat_model(settings: Settings, temperature: float | None = None) -> BaseChatModel:
    settings.require_llm()
    temperature = settings.llm_temperature if temperature is None else temperature
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=temperature,
            max_retries=_MAX_RETRIES,
            timeout=_REQUEST_TIMEOUT_SECONDS,
            max_tokens=8192,
        )
    from langchain_openai import ChatOpenAI

    if settings.llm_provider == "gemini":
        # Google exposes Gemini through an OpenAI-compatible endpoint (free tier available).
        return ChatOpenAI(
            model=settings.gemini_model,
            api_key=settings.gemini_api_key,
            base_url=GEMINI_OPENAI_BASE_URL,
            temperature=temperature,
            max_retries=_MAX_RETRIES,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=temperature,
        max_retries=_MAX_RETRIES,
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )


def friendly_api_error(exc: Exception, service: str = "the AI provider") -> tuple[str, bool]:
    """Translate an API exception into (user-facing message, is_fatal)."""
    text = f"{type(exc).__name__}: {exc}"
    if "AuthenticationError" in text or "401" in text or "invalid x-api-key" in text:
        return f"{service} rejected the API key. Check the key in your .env file.", True
    lowered = text.lower()
    if any(phrase in lowered for phrase in ("insufficient_quota", "credit balance", "no credits", "billing")):
        return f"Your {service} account has no remaining quota or credit. Check your billing.", True
    if "NotFoundError" in text or "model_not_found" in text:
        return f"{service} does not recognise the configured model name. Check your .env file.", True
    if "RateLimitError" in text or "429" in text:
        return f"{service} rate limit reached. Wait a minute and try again.", False
    if "503" in text or "UNAVAILABLE" in text or "overloaded" in lowered or "high demand" in lowered:
        return f"{service} is overloaded right now. Wait a minute and try again.", False
    if "APIConnectionError" in text or "Timeout" in text or "ConnectError" in text:
        return f"Could not reach {service}. Check your internet connection and try again.", False
    return f"{service} request failed: {exc}", False


def invoke_structured(chain: Runnable, inputs: dict[str, Any], task: str, service: str) -> dict:
    """Run a ``with_structured_output(..., include_raw=True)`` chain.

    Returns the dict with keys ``raw``, ``parsed`` and ``parsing_error``.
    API failures are logged with a traceback and re-raised as LLMError.
    """
    try:
        return chain.invoke(inputs)
    except Exception as exc:
        logger.exception("LLM API error during %s", task)
        message, fatal = friendly_api_error(exc, service)
        raise LLMError(message, fatal=fatal) from exc


def service_name(settings: Settings) -> str:
    return {"openai": "OpenAI", "anthropic": "Anthropic", "gemini": "Google Gemini"}[settings.llm_provider]
