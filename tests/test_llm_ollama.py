from __future__ import annotations

import pytest
import requests

from src.llm.ollama import ChatMessage, LLMConfigurationError, OllamaClient


class FakeResponse:
    def __init__(self, payload: dict[str, object], text: str = "OK") -> None:
        self.payload = payload
        self.text = text

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return self.payload


def test_ollama_cloud_requires_api_key() -> None:
    client = OllamaClient(api_base="https://ollama.com", model="gpt-oss:120b")

    assert client.is_configured is False
    assert client.configuration_issue == "Set OLLAMA_API_KEY in .env to use Ollama Cloud."
    with pytest.raises(LLMConfigurationError, match="OLLAMA_API_KEY"):
        client.chat([ChatMessage(role="user", content="Hello")])


def test_ollama_cloud_posts_to_direct_api_with_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(
        url: str,
        json: dict[str, object],
        headers: dict[str, str],
        timeout: int,
    ) -> FakeResponse:
        captured.update({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse({"message": {"content": "Grounded cloud answer"}})

    monkeypatch.setattr(requests, "post", fake_post)
    client = OllamaClient(
        api_base="https://ollama.com",
        api_key="test-key",
        model="gpt-oss:120b",
        timeout_seconds=12,
    )

    answer = client.chat([ChatMessage(role="user", content="Answer from context")])

    assert answer == "Grounded cloud answer"
    assert captured["url"] == "https://ollama.com/api/chat"
    assert captured["timeout"] == 12
    assert captured["headers"] == {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-key",
    }
    assert captured["json"] == {
        "model": "gpt-oss:120b",
        "messages": [{"role": "user", "content": "Answer from context"}],
        "stream": False,
    }


def test_local_ollama_does_not_require_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_post(
        url: str,
        json: dict[str, object],
        headers: dict[str, str],
        timeout: int,
    ) -> FakeResponse:
        captured.update({"url": url, "json": json, "headers": headers, "timeout": timeout})
        return FakeResponse({"message": {"content": "Grounded local answer"}})

    monkeypatch.setattr(requests, "post", fake_post)
    client = OllamaClient(api_base="http://localhost:11434", model="llama3.1")

    answer = client.chat([ChatMessage(role="user", content="Answer from context")])

    assert client.is_configured is True
    assert answer == "Grounded local answer"
    assert captured["url"] == "http://localhost:11434/api/chat"
    assert captured["headers"] == {"Content-Type": "application/json"}
