"""Groq LLM client wrapper with retry, exponential backoff, and model fallback."""

from __future__ import annotations

import os
import re
import time
from abc import ABC, abstractmethod
from typing import Any

from dotenv import load_dotenv
from groq import Groq

from t_edu_agent.config import LLMConfig


class LLMServiceError(RuntimeError):
    """User-safe error raised when LLM completion fails across retries and fallbacks."""


class BaseLLMClient(ABC):
    """Abstract chat completion client for agent orchestration and unit tests."""

    @abstractmethod
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        preferred_model: str | None = None,
    ) -> tuple[str, str]:
        """Run chat completion and return (assistant_text, model_used)."""


def _is_transient_error(exc: Exception) -> bool:
    """Detect rate-limit, timeout, or transient upstream 5xx failures."""
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int) and status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    msg = str(exc).lower()
    return any(
        token in msg
        for token in ("rate limit", "429", "timeout", "timed out", "temporarily", "overloaded", "connection")
    )


def _is_json_mode_unsupported(exc: Exception) -> bool:
    """Detect provider error when a specific model rejects response_format=json_object."""
    status_code = getattr(exc, "status_code", None)
    msg = str(exc).lower()
    return status_code == 400 and ("response_format" in msg or "json" in msg)


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _clean_completion(raw_text: str | None) -> str:
    """Strip hidden reasoning blocks and outer whitespace from model output."""
    return _THINK_BLOCK_RE.sub("", raw_text or "").strip()


class GroqLLMClient(BaseLLMClient):
    """Production Groq SDK client supporting retries and model fallback without exposing keys."""

    def __init__(
        self,
        cfg: LLMConfig,
        client: Any | None = None,
        backoff_base_sec: float = 0.5,
    ) -> None:
        self.cfg = cfg
        self.backoff_base_sec = backoff_base_sec
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        load_dotenv(override=False)
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise LLMServiceError(
                "Не найден ключ GROQ_API_KEY. Добавьте его в файл .env или переменные окружения."
            )
        self._client = Groq(api_key=api_key, timeout=float(self.cfg.timeout_sec), max_retries=0)
        return self._client

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        preferred_model: str | None = None,
    ) -> tuple[str, str]:
        client = self._get_client()
        raw_models = (
            [preferred_model, self.cfg.primary_model, *self.cfg.fallback_models]
            if preferred_model
            else [self.cfg.primary_model, *self.cfg.fallback_models]
        )
        models = list(dict.fromkeys(m for m in raw_models if m))
        temp = self.cfg.temperature if temperature is None else temperature
        toks = self.cfg.max_tokens if max_tokens is None else max_tokens

        last_error: Exception | None = None
        failed_models: list[str] = []
        for model_name in models:
            use_json_format = json_mode
            model_error: Exception | None = None
            for attempt in range(self.cfg.max_retries + 1):
                kwargs: dict[str, Any] = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": temp,
                    "max_tokens": toks,
                }
                if use_json_format:
                    kwargs["response_format"] = {"type": "json_object"}

                try:
                    response = client.chat.completions.create(**kwargs)
                    cleaned = _clean_completion(response.choices[0].message.content)
                    if not cleaned:
                        raise RuntimeError("empty_completion")
                    return cleaned, model_name
                except Exception as exc:
                    last_error = exc
                    model_error = exc
                    if use_json_format and _is_json_mode_unsupported(exc):
                        use_json_format = False
                        continue
                    if attempt < self.cfg.max_retries and _is_transient_error(exc):
                        delay = self.backoff_base_sec * (2**attempt)
                        if delay > 0:
                            time.sleep(delay)
                        continue
                    break
            if model_error is not None:
                failed_models.append(f"{model_name}:{type(model_error).__name__}")

        details = (
            ", ".join(failed_models)
            if failed_models
            else (type(last_error).__name__ if last_error is not None else "UnknownError")
        )
        raise LLMServiceError(
            f"Не удалось получить ответ от языковой модели ({details}). Попробуйте повторить запрос."
        )
