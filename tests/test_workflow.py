from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.agents import AnswerAgent, RetrievalAgent, ValidationAgent
from src.config import AppConfig
from src.llm.ollama import ChatMessage
from src.models import QueryResult, RetrievedChunk, TextChunk
from src.rag.workflow import DocumentQAWorkflow


class FakeVectorStore:
    def __init__(self) -> None:
        self.chunks: list[TextChunk] = []

    def clear(self) -> None:
        self.chunks = []

    def upsert_chunks(self, chunks: list[TextChunk]) -> int:
        self.chunks.extend(chunks)
        return len(chunks)

    def query(self, question: str, top_k: int) -> list[RetrievedChunk]:
        del question
        return [
            RetrievedChunk(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata,
                relevance_score=0.92,
                distance=0.08,
            )
            for chunk in self.chunks[:top_k]
        ]


@dataclass
class FakeChatClient:
    configured: bool
    response: str = "The source says the launch risk is schedule delay. [Source 1]\n\nUsed sources: 1"

    @property
    def is_configured(self) -> bool:
        return self.configured

    @property
    def configuration_issue(self) -> str | None:
        if self.configured:
            return None
        return "Set OLLAMA_MODEL in .env before asking the model to generate answers."

    def chat(self, messages: list[ChatMessage]) -> str:
        assert "Document context" in messages[-1].content
        return self.response


def test_workflow_indexes_txt_and_returns_answer() -> None:
    vector_store = FakeVectorStore()
    workflow = _build_workflow(vector_store, FakeChatClient(configured=True))

    index_result = workflow.index_files(
        [("risk.txt", b"The main launch risk is schedule delay.")],
        reset=True,
    )
    query_result = workflow.ask("What is the launch risk?")

    assert index_result.indexed_chunk_count == 1
    assert query_result.used_llm is True
    assert "schedule delay" in query_result.answer
    assert query_result.used_source_indices == [1]
    assert query_result.used_sources == query_result.sources[:1]
    assert query_result.sources[0].metadata["source_name"] == "risk.txt"


def test_workflow_shows_retrieval_when_llm_is_placeholder() -> None:
    vector_store = FakeVectorStore()
    workflow = _build_workflow(vector_store, FakeChatClient(configured=False))

    workflow.index_files([("notes.txt", b"Customer churn improved after onboarding.")])
    query_result = workflow.ask("What improved?")

    assert query_result.used_llm is False
    assert query_result.sources
    assert query_result.used_source_indices == [1]
    assert "retrieval-only answer" in query_result.answer
    assert "Customer churn improved after onboarding" in query_result.answer
    assert any("LLM" in warning for warning in query_result.warnings)


def test_workflow_allows_top_k_override() -> None:
    vector_store = FakeVectorStore()
    workflow = _build_workflow(vector_store, FakeChatClient(configured=False))

    workflow.index_files(
        [
            ("first.txt", b"First document discusses churn."),
            ("second.txt", b"Second document discusses revenue."),
        ]
    )
    query_result = workflow.ask("What do the documents discuss?", top_k=1)

    assert len(query_result.sources) == 1


def test_workflow_tracks_sources_declared_by_llm() -> None:
    vector_store = FakeVectorStore()
    workflow = _build_workflow(
        vector_store,
        FakeChatClient(
            configured=True,
            response="Revenue is the relevant topic. [Source 2]\n\nUsed sources: 2",
        ),
    )

    workflow.index_files(
        [
            ("first.txt", b"First document discusses churn."),
            ("second.txt", b"Second document discusses revenue."),
        ]
    )
    query_result = workflow.ask("Which document mentions revenue?", top_k=2)

    assert query_result.used_source_indices == [2]
    assert len(query_result.used_sources) == 1
    assert query_result.used_sources[0].metadata["source_name"] == "second.txt"
    assert "Used sources:" not in query_result.answer


def test_query_result_normalizes_used_source_indices() -> None:
    sources = [
        RetrievedChunk(
            id="chunk-1",
            text="First source",
            metadata={"source_name": "first.txt"},
            relevance_score=0.9,
        ),
        RetrievedChunk(
            id="chunk-2",
            text="Second source",
            metadata={"source_name": "second.txt"},
            relevance_score=0.8,
        ),
    ]

    result = QueryResult(
        question="Which sources?",
        answer="Both sources apply.",
        sources=sources,
        warnings=[],
        used_llm=True,
        used_source_indices=[2, 99, 2, 1],
    )

    assert result.used_source_indices == [2, 1]
    assert result.used_sources == [sources[1], sources[0]]


def _build_workflow(
    vector_store: FakeVectorStore,
    chat_client: FakeChatClient,
) -> DocumentQAWorkflow:
    config = AppConfig(
        ollama_api_base="https://example.test",
        ollama_api_key=None,
        ollama_model="test-model",
        embedding_model="test-embedding",
        chroma_path=Path("ignored"),
        chroma_collection="test",
        chunk_size=200,
        chunk_overlap=20,
        retrieval_top_k=3,
        min_relevance_score=0.25,
    )
    return DocumentQAWorkflow(
        config=config,
        vector_store=vector_store,
        retrieval_agent=RetrievalAgent(vector_store),
        answer_agent=AnswerAgent(chat_client),
        validation_agent=ValidationAgent(),
    )
