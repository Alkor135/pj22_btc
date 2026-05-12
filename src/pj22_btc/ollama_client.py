"""Минимальный клиент локального Ollama API для исследовательских пайплайнов."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


DETERMINISTIC_OLLAMA_OPTIONS = {
    "temperature": 0,
    "top_p": 1,
    "top_k": 1,
    "seed": 42,
}


class OllamaAPIError(RuntimeError):
    """Ошибка вызова локального Ollama API."""


@dataclass(frozen=True)
class OllamaClient:
    """HTTP-клиент для `/api/generate` локального Ollama."""

    base_url: str = "http://localhost:11434"

    def generate(
        self,
        *,
        model: str,
        prompt: str,
        keep_alive: str | None = None,
        timeout_seconds: int = 60,
    ) -> str:
        """Возвращает текст ответа модели для одного prompt."""
        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": DETERMINISTIC_OLLAMA_OPTIONS,
        }
        if keep_alive:
            payload["keep_alive"] = keep_alive

        try:
            response = requests.post(
                f"{self.base_url.rstrip('/')}/api/generate",
                json=payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise OllamaAPIError(f"Ollama request failed: {exc}") from exc
        except ValueError as exc:
            raise OllamaAPIError("Ollama returned invalid JSON") from exc

        return str(data.get("response") or "").strip()
