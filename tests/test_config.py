from __future__ import annotations

from pathlib import Path

from src.config import AppConfig


def test_cloud_config_requires_api_key_for_llm_generation() -> None:
    config = _build_config(
        ollama_api_base="https://ollama.com",
        ollama_api_key=None,
        ollama_model="gpt-oss:120b",
    )

    assert config.uses_ollama_cloud is True
    assert config.has_configured_llm is False
    assert config.llm_configuration_issue == "Set OLLAMA_API_KEY in .env to use Ollama Cloud."


def test_local_config_allows_missing_api_key() -> None:
    config = _build_config(
        ollama_api_base="http://localhost:11434",
        ollama_api_key=None,
        ollama_model="llama3.1",
    )

    assert config.uses_ollama_cloud is False
    assert config.has_configured_llm is True
    assert config.llm_configuration_issue is None


def _build_config(
    ollama_api_base: str,
    ollama_api_key: str | None,
    ollama_model: str,
) -> AppConfig:
    return AppConfig(
        ollama_api_base=ollama_api_base,
        ollama_api_key=ollama_api_key,
        ollama_model=ollama_model,
        embedding_model="local-hash",
        chroma_path=Path("ignored"),
        chroma_collection="test",
        chunk_size=200,
        chunk_overlap=20,
        retrieval_top_k=3,
        min_relevance_score=0.25,
    )
