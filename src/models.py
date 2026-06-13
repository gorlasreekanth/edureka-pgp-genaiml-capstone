from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Metadata = dict[str, Any]


@dataclass(frozen=True)
class SourceDocument:
    text: str
    metadata: Metadata

    @property
    def source_name(self) -> str:
        return str(self.metadata.get("source_name", "unknown"))


@dataclass(frozen=True)
class TextChunk:
    id: str
    text: str
    metadata: Metadata


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    metadata: Metadata
    relevance_score: float
    distance: float | None = None


@dataclass(frozen=True)
class IndexResult:
    document_count: int
    chunk_count: int
    indexed_chunk_count: int
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class QueryResult:
    question: str
    answer: str
    sources: list[RetrievedChunk]
    warnings: list[str]
    used_llm: bool
