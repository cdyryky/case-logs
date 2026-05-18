from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "gemma4:latest"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_NUM_CTX = 8192


class LLMClientError(RuntimeError):
    pass


class LLMResponseError(LLMClientError):
    pass


@dataclass(frozen=True)
class LLMSettings:
    base_url: str = DEFAULT_OLLAMA_BASE_URL
    model: str = DEFAULT_OLLAMA_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    num_ctx: int = DEFAULT_NUM_CTX

    @classmethod
    def from_env(cls) -> "LLMSettings":
        timeout_raw = os.environ.get("ACGME_LLM_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
        num_ctx_raw = os.environ.get("ACGME_LLM_NUM_CTX", str(DEFAULT_NUM_CTX))
        try:
            timeout = float(timeout_raw)
        except ValueError:
            timeout = DEFAULT_TIMEOUT_SECONDS
        try:
            num_ctx = int(num_ctx_raw)
        except ValueError:
            num_ctx = DEFAULT_NUM_CTX
        return cls(
            base_url=os.environ.get("ACGME_LLM_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/"),
            model=os.environ.get("ACGME_LLM_MODEL", DEFAULT_OLLAMA_MODEL),
            timeout_seconds=timeout,
            num_ctx=max(num_ctx, DEFAULT_NUM_CTX),
        )


class OllamaClient:
    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or LLMSettings.from_env()

    def generate_json(self, prompt: str, schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        payload = {
            "model": self.settings.model,
            "prompt": prompt,
            "stream": False,
            "format": schema,
            "options": {"num_ctx": self.settings.num_ctx},
        }
        try:
            response = httpx.post(
                f"{self.settings.base_url}/api/generate",
                json=payload,
                timeout=self.settings.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMClientError(f"Ollama request failed: {exc}") from exc

        try:
            raw = response.json()
        except json.JSONDecodeError as exc:
            raise LLMResponseError("Ollama returned non-JSON response envelope.") from exc

        content = raw.get("response")
        if not isinstance(content, str) or not content.strip():
            raise LLMResponseError("Ollama response envelope did not contain a JSON response string.")
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMResponseError(f"Ollama returned invalid JSON content: {exc}") from exc
        if not isinstance(parsed, dict):
            raise LLMResponseError("Ollama JSON content must be an object.")
        return parsed, raw


def ollama_health(settings: LLMSettings | None = None) -> dict[str, Any]:
    settings = settings or LLMSettings.from_env()
    result: dict[str, Any] = {
        "base_url": settings.base_url,
        "model": settings.model,
        "timeout_seconds": settings.timeout_seconds,
        "num_ctx": settings.num_ctx,
        "reachable": False,
        "model_available": False,
        "error": "",
    }
    try:
        response = httpx.get(f"{settings.base_url}/api/tags", timeout=1.0)
        response.raise_for_status()
        tags = response.json()
    except Exception as exc:
        result["error"] = str(exc)
        return result
    models = [str(item.get("name") or "") for item in tags.get("models") or [] if isinstance(item, dict)]
    result["reachable"] = True
    result["model_available"] = settings.model in models
    return result
