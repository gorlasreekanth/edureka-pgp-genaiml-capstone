from __future__ import annotations

from typing import Protocol

from src.agents import AnswerAgent, QueryPlannerAgent, RetrievalAgent, ValidationAgent
from src.config import AppConfig
from src.ingestion import DocumentLoadError, load_uploaded_file
from src.llm import OllamaClient
from src.models import IndexResult, QueryResult, TextChunk
from src.rag.chunking import chunk_documents
from src.rag.embeddings import SentenceTransformerEmbeddings
from src.retrieval import ChromaVectorStore


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
        embeddings = SentenceTransformerEmbeddings(config.embedding_model)
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

    def index_files(self, files: list[tuple[str, bytes]], reset: bool = True) -> IndexResult:
        if not files:
            raise ValueError("Upload at least one document before indexing.")

        documents = []
        errors: list[str] = []
        for file_name, content in files:
            try:
                documents.extend(load_uploaded_file(file_name, content))
            except DocumentLoadError as exc:
                errors.append(str(exc))

        if not documents:
            return IndexResult(
                document_count=0,
                chunk_count=0,
                indexed_chunk_count=0,
                errors=errors,
            )

        chunks = chunk_documents(
            documents,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        if reset:
            self.vector_store.clear()
        indexed_count = self.vector_store.upsert_chunks(chunks)
        return IndexResult(
            document_count=len(documents),
            chunk_count=len(chunks),
            indexed_chunk_count=indexed_count,
            errors=errors,
        )

    def ask(self, question: str) -> QueryResult:
        plan = self.query_planner.plan(question, self.config.retrieval_top_k)
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
