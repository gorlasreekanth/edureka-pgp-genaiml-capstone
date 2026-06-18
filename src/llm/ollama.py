from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

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
        return self.configuration_issue is None

    @property
    def configuration_issue(self) -> str | None:
        if not self.api_base:
            return "Set OLLAMA_API_BASE before asking the model to generate answers."
        if not self.model or self.model.startswith("replace-with-"):
            return "Set OLLAMA_MODEL in .env before asking the model to generate answers."
        if self.uses_ollama_cloud and not self.api_key:
            return "Set OLLAMA_API_KEY in .env to use Ollama Cloud."
        return None

    @property
    def uses_ollama_cloud(self) -> bool:
        parsed = urlparse(self.api_base if "://" in self.api_base else f"https://{self.api_base}")
        host = parsed.hostname or ""
        return host == "ollama.com" or host.endswith(".ollama.com")

    def chat(self, messages: list[ChatMessage]) -> str:
        if self.configuration_issue:
            raise LLMConfigurationError(self.configuration_issue)

        try:
            response = requests.post(
                self._chat_url(),
                json={
                    "model": self.model,
                    "messages": [message.__dict__ for message in messages],
                    "stream": False,
                },
                headers=self._headers(),
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = getattr(exc.response, "text", None) or str(exc)
            raise LLMClientError(f"Ollama API request failed: {detail}") from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise LLMClientError("Ollama API response was not valid JSON.") from exc
        content = _extract_content(data)
        if not content:
            raise LLMClientError("Ollama API response did not include answer content.")
        return content.strip()

    def _chat_url(self) -> str:
        if self.api_base.endswith("/api"):
            return f"{self.api_base}/chat"
        return f"{self.api_base}/api/chat"

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
