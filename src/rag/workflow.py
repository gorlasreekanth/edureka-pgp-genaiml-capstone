from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from src.agents import AnswerAgent, QueryPlannerAgent, RetrievalAgent, ValidationAgent
from src.config import AppConfig
from src.ingestion import DocumentLoadError, load_uploaded_file
from src.llm import OllamaClient
from src.models import IndexResult, QueryResult, TextChunk
from src.rag.chunking import chunk_documents
from src.rag.embeddings import create_embedding_provider
from src.retrieval import ChromaVectorStore


ProgressCallback = Callable[[str], None]


class IndexableVectorStore(Protocol):
    def clear(self) -> None:
        ...

    def upsert_chunks(self, chunks: list[TextChunk]) -> int:
        ...


class DocumentQAWorkflow:
    def __init__(
        self,
        config: AppConfig,
        vector_store: IndexableVectorStore,
        retrieval_agent: RetrievalAgent,
        answer_agent: AnswerAgent,
        validation_agent: ValidationAgent,
    ) -> None:
        self.config = config
        self.vector_store = vector_store
        self.query_planner = QueryPlannerAgent()
        self.retrieval_agent = retrieval_agent
        self.answer_agent = answer_agent
        self.validation_agent = validation_agent

    @classmethod
    def from_config(cls, config: AppConfig) -> "DocumentQAWorkflow":
        embeddings = create_embedding_provider(config.embedding_model)
        vector_store = ChromaVectorStore(
            path=config.chroma_path,
            collection_name=config.chroma_collection,
            embedding_provider=embeddings,
        )
        llm_client = OllamaClient(
            api_base=config.ollama_api_base,
            api_key=config.ollama_api_key,
            model=config.ollama_model,
        )
        return cls(
            config=config,
            vector_store=vector_store,
            retrieval_agent=RetrievalAgent(vector_store),
            answer_agent=AnswerAgent(llm_client),
            validation_agent=ValidationAgent(),
        )

    def index_files(
        self,
        files: list[tuple[str, bytes]],
        reset: bool = True,
        progress_callback: ProgressCallback | None = None,
    ) -> IndexResult:
        if not files:
            raise ValueError("Upload at least one document before indexing.")

        _report(progress_callback, "Parsing uploaded files...")
        documents = []
        errors: list[str] = []
        for file_name, content in files:
            try:
                documents.extend(load_uploaded_file(file_name, content))
            except DocumentLoadError as exc:
                errors.append(str(exc))

        if not documents:
            _report(progress_callback, "No extractable document text was found.")
            return IndexResult(
                document_count=0,
                chunk_count=0,
                indexed_chunk_count=0,
                errors=errors,
            )

        _report(progress_callback, f"Chunking {len(documents)} parsed document sections...")
        chunks = chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        if reset:
            _report(progress_callback, "Resetting the local vector collection...")
            self.vector_store.clear()
        _report(
            progress_callback,
            f"Embedding and storing {len(chunks)} chunks using `{self.config.embedding_model}`...",
        )
        indexed_count = self.vector_store.upsert_chunks(chunks)
        _report(progress_callback, f"Indexed {indexed_count} chunks.")
        return IndexResult(
            document_count=len(documents),
            chunk_count=len(chunks),
            indexed_chunk_count=indexed_count,
            errors=errors,
        )

    def ask(self, question: str, top_k: int | None = None) -> QueryResult:
        plan = self.query_planner.plan(question, top_k or self.config.retrieval_top_k)
        sources = self.retrieval_agent.retrieve(plan)
        draft = self.answer_agent.answer(plan, sources)
        validation = self.validation_agent.validate(
            sources=sources,
            draft=draft,
            min_relevance_score=self.config.min_relevance_score,
        )
        return QueryResult(
            question=plan.question,
            answer=draft.answer,
            sources=sources,
            warnings=validation.warnings,
            used_llm=draft.used_llm,
        )


def _report(progress_callback: ProgressCallback | None, message: str) -> None:
    if progress_callback:
        progress_callback(message)
