from src.ingestion.loaders import load_uploaded_file
from src.rag.chunking import chunk_documents, recursive_chunk_text


def test_recursive_chunk_text_keeps_overlap() -> None:
    text = "Alpha beta gamma. Delta epsilon zeta. Eta theta iota."

    chunks = recursive_chunk_text(text, chunk_size=28, chunk_overlap=8)

    assert len(chunks) > 1
    assert all(len(chunk) <= 28 for chunk in chunks)
    assert "Delta" in chunks[1]


def test_chunk_documents_adds_stable_metadata() -> None:
    documents = load_uploaded_file(
        "notes.txt",
        b"Revenue increased in Q1.\n\nCustomer churn decreased after onboarding changes.",
    )

    chunks = chunk_documents(documents, chunk_size=35, chunk_overlap=5)

    assert chunks
    assert chunks[0].metadata["source_name"] == "notes.txt"
    assert chunks[0].metadata["doc_type"] == "txt"
    assert chunks[0].metadata["chunk_index"] == 1
    assert chunks[0].id
