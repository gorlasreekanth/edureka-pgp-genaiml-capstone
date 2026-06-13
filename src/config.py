from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


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

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv()
        return cls(
            ollama_api_base=_get_str("OLLAMA_API_BASE", "https://ollama.com").rstrip("/"),
            ollama_api_key=_get_optional_str("OLLAMA_API_KEY"),
            ollama_model=_get_str("OLLAMA_MODEL", "replace-with-your-ollama-model"),
            embedding_model=_get_str(
                "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            chroma_path=Path(_get_str("CHROMA_PATH", "chroma_db")),
            chroma_collection=_get_str("CHROMA_COLLECTION", "document_qa"),
            chunk_size=_get_int("CHUNK_SIZE", 900),
            chunk_overlap=_get_int("CHUNK_OVERLAP", 120),
            retrieval_top_k=_get_int("RETRIEVAL_TOP_K", 4),
            min_relevance_score=_get_float("MIN_RELEVANCE_SCORE", 0.25),
        )

    @property
    def has_configured_llm(self) -> bool:
        return not self.ollama_model.startswith("replace-with-")


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
