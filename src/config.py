from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from src.validation import (
    DEFAULT_MAX_FILE_SIZE_MB,
    DEFAULT_MAX_QUESTION_CHARS,
    DEFAULT_MIN_QUESTION_CHARS,
    ValidationLimits,
)


@dataclass(frozen=True)
class AppConfig:
    """Runtime settings loaded from environment variables."""

    ollama_api_base: str
    ollama_api_key: str | None
    ollama_model: str
    embedding_model: str
    chroma_path: Path
    chroma_collection: str
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    min_relevance_score: float
    max_file_size_mb: int = DEFAULT_MAX_FILE_SIZE_MB
    min_question_chars: int = DEFAULT_MIN_QUESTION_CHARS
    max_question_chars: int = DEFAULT_MAX_QUESTION_CHARS
    ollama_timeout_seconds: int = 120

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv()
        return cls(
            ollama_api_base=_get_str("OLLAMA_API_BASE", "https://ollama.com").rstrip("/"),
            ollama_api_key=_get_optional_str("OLLAMA_API_KEY"),
            ollama_model=_get_str("OLLAMA_MODEL", "replace-with-your-ollama-model"),
            ollama_timeout_seconds=_get_int("OLLAMA_TIMEOUT_SECONDS", 120),
            embedding_model=_get_str("EMBEDDING_MODEL", "local-hash"),
            chroma_path=Path(_get_str("CHROMA_PATH", "chroma_db")),
            chroma_collection=_get_str("CHROMA_COLLECTION", "document_qa"),
            chunk_size=_get_int("CHUNK_SIZE", 900),
            chunk_overlap=_get_int("CHUNK_OVERLAP", 120),
            retrieval_top_k=_get_int("RETRIEVAL_TOP_K", 4),
            min_relevance_score=_get_float("MIN_RELEVANCE_SCORE", 0.25),
            max_file_size_mb=_get_int("MAX_FILE_SIZE_MB", DEFAULT_MAX_FILE_SIZE_MB),
            min_question_chars=_get_int("MIN_QUESTION_CHARS", DEFAULT_MIN_QUESTION_CHARS),
            max_question_chars=_get_int("MAX_QUESTION_CHARS", DEFAULT_MAX_QUESTION_CHARS),
        )

    @property
    def validation_limits(self) -> ValidationLimits:
        return ValidationLimits(
            max_file_size_mb=self.max_file_size_mb,
            min_question_chars=self.min_question_chars,
            max_question_chars=self.max_question_chars,
        )

    @property
    def has_configured_llm(self) -> bool:
        return self.llm_configuration_issue is None

    @property
    def uses_ollama_cloud(self) -> bool:
        return _is_ollama_cloud_base(self.ollama_api_base)

    @property
    def ollama_runtime_label(self) -> str:
        if self.uses_ollama_cloud:
            return "Ollama Cloud"
        return "local/custom Ollama"

    @property
    def llm_configuration_issue(self) -> str | None:
        if self.ollama_model.startswith("replace-with-"):
            return "Set OLLAMA_MODEL in .env before asking the LLM to generate final answers."
        if self.uses_ollama_cloud and not self.ollama_api_key:
            return "Set OLLAMA_API_KEY in .env to use Ollama Cloud."
        return None


def _get_str(name: str, default: str) -> str:
    value = os.getenv(name, default).strip()
    if not value:
        raise ValueError(f"{name} cannot be empty.")
    return value


def _get_optional_str(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    value = value.strip()
    if not value or value.startswith("replace-with-"):
        return None
    return value


def _get_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero.")
    return value


def _get_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number.") from exc
    if value < 0:
        raise ValueError(f"{name} cannot be negative.")
    return value


def _is_ollama_cloud_base(api_base: str) -> bool:
    parsed = urlparse(api_base if "://" in api_base else f"https://{api_base}")
    host = parsed.hostname or ""
    return host == "ollama.com" or host.endswith(".ollama.com")
