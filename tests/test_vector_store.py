from __future__ import annotations

from src.models import TextChunk
from src.retrieval.vector_store import ChromaVectorStore


class FakeEmbeddingProvider:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        del text
        return [1.0, 0.0, 0.0]


def test_chroma_clear_resets_collection_with_current_api(tmp_path) -> None:
    store = ChromaVectorStore(
        path=tmp_path / "chroma",
        collection_name="document_qa_test",
        embedding_provider=FakeEmbeddingProvider(),
    )
    store.upsert_chunks(
        [
            TextChunk(
                id="chunk-1",
                text="Customer churn improved after onboarding changes.",
                metadata={"source_name": "notes.txt", "chunk_index": 1},
            )
        ]
    )

    store.clear()
    results = store.query("What improved?", top_k=3)

    assert results == []
