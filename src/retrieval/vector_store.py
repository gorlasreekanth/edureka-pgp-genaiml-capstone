from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models import RetrievedChunk, TextChunk
from src.rag.embeddings import EmbeddingProvider


class ChromaVectorStore:
    def __init__(
        self,
        path: Path,
        collection_name: str,
        embedding_provider: EmbeddingProvider,
    ) -> None:
        import chromadb
        from chromadb.config import Settings

        self.path = path
        self.collection_name = collection_name
        self.embedding_provider = embedding_provider
        self.client = chromadb.PersistentClient(
            path=str(path),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def clear(self) -> None:
        if self.collection_name in self.client.list_collections():
            self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_chunks(self, chunks: list[TextChunk]) -> int:
        if not chunks:
            return 0

        embeddings = self.embedding_provider.embed_documents([chunk.text for chunk in chunks])
        self.collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[_sanitize_metadata(chunk.metadata) for chunk in chunks],
            embeddings=embeddings,
        )
        return len(chunks)

    def query(self, question: str, top_k: int) -> list[RetrievedChunk]:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question cannot be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        available_count = self.collection.count()
        if available_count == 0:
            return []

        query_embedding = self.embedding_provider.embed_query(normalized_question)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, available_count),
            include=["documents", "metadatas", "distances"],
        )

        ids = _first_result_list(results, "ids")
        documents = _first_result_list(results, "documents")
        metadatas = _first_result_list(results, "metadatas")
        distances = _first_result_list(results, "distances")

        retrieved: list[RetrievedChunk] = []
        for index, chunk_id in enumerate(ids):
            distance = float(distances[index]) if index < len(distances) else None
            retrieved.append(
                RetrievedChunk(
                    id=str(chunk_id),
                    text=str(documents[index]) if index < len(documents) else "",
                    metadata=dict(metadatas[index]) if index < len(metadatas) else {},
                    relevance_score=_distance_to_relevance(distance),
                    distance=distance,
                )
            )
        return retrieved


def _first_result_list(results: dict[str, Any], key: str) -> list[Any]:
    value = results.get(key) or [[]]
    return list(value[0] or [])


def _distance_to_relevance(distance: float | None) -> float:
    if distance is None:
        return 0.0
    if distance <= 1:
        return max(0.0, min(1.0, 1.0 - distance))
    return 1.0 / (1.0 + distance)


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    sanitized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if isinstance(value, str | int | float | bool):
            sanitized[key] = value
        elif value is not None:
            sanitized[key] = str(value)
    return sanitized
