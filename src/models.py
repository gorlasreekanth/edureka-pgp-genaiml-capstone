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
    used_source_indices: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "used_source_indices",
            normalize_source_indices(self.used_source_indices, len(self.sources)),
        )

    @property
    def used_sources(self) -> list[RetrievedChunk]:
        return [
            self.sources[index - 1]
            for index in self.used_source_indices
            if 1 <= index <= len(self.sources)
        ]


def normalize_source_indices(indices: list[int], source_count: int) -> list[int]:
    valid_indices = []
    for index in indices:
        if 1 <= index <= source_count and index not in valid_indices:
            valid_indices.append(index)
    return valid_indices
