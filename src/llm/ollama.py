from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests


class LLMConfigurationError(RuntimeError):
    """Raised when the LLM client is called without usable configuration."""


class LLMClientError(RuntimeError):
    """Raised when the Ollama-compatible API returns an unusable response."""


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


class OllamaClient:
    def __init__(
        self,
        api_base: str,
        model: str,
        api_key: str | None = None,
        timeout_seconds: int = 90,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds

    @property
    def is_configured(self) -> bool:
        return bool(self.api_base and self.model and not self.model.startswith("replace-with-"))

    def chat(self, messages: list[ChatMessage]) -> str:
        if not self.is_configured:
            raise LLMConfigurationError(
                "Set OLLAMA_MODEL in .env before asking the model to generate answers."
            )

        response = requests.post(
            f"{self.api_base}/api/chat",
            json={
                "model": self.model,
                "messages": [message.__dict__ for message in messages],
                "stream": False,
            },
            headers=self._headers(),
            timeout=self.timeout_seconds,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise LLMClientError(f"Ollama API request failed: {response.text}") from exc

        data = response.json()
        content = _extract_content(data)
        if not content:
            raise LLMClientError("Ollama API response did not include answer content.")
        return content.strip()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def _extract_content(data: dict[str, Any]) -> str | None:
    message = data.get("message")
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    response = data.get("response")
    if isinstance(response, str):
        return response
    return None
