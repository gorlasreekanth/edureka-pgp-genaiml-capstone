import pytest

from src.rag.embeddings import LocalHashEmbeddings, create_embedding_provider


def test_local_hash_embeddings_are_normalized_and_deterministic() -> None:
    embeddings = LocalHashEmbeddings(dimensions=32)

    first = embeddings.embed_query("Customer churn improved")
    second = embeddings.embed_query("Customer churn improved")

    assert first == second
    assert len(first) == 32
    assert sum(value * value for value in first) == pytest.approx(1.0)


def test_embedding_factory_uses_local_hash_mode() -> None:
    provider = create_embedding_provider("local-hash")

    assert isinstance(provider, LocalHashEmbeddings)
