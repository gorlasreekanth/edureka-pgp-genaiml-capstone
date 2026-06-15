from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from src.agents.query_planner import QueryPlan
from src.llm.ollama import ChatMessage
from src.models import RetrievedChunk


class ChatClient(Protocol):
    @property
    def is_configured(self) -> bool:
        ...

    def chat(self, messages: list[ChatMessage]) -> str:
        ...


@dataclass(frozen=True)
class AnswerDraft:
    answer: str
    used_llm: bool
    warnings: list[str] = field(default_factory=list)


class AnswerAgent:
    def __init__(self, llm_client: ChatClient) -> None:
        self.llm_client = llm_client

    def answer(self, plan: QueryPlan, sources: list[RetrievedChunk]) -> AnswerDraft:
        if not sources:
            return AnswerDraft(
                answer=(
                    "I could not find enough relevant context in the indexed documents to answer that."
                ),
                used_llm=False,
                warnings=["No relevant document chunks were retrieved."],
            )

        if not self.llm_client.is_configured:
            return AnswerDraft(
                answer=_build_retrieval_only_answer(sources),
                used_llm=False,
                warnings=[
                    "LLM settings are placeholders, so this is a retrieval-only answer from matching source chunks."
                ],
            )

        prompt = _build_grounded_prompt(plan.question, sources)
        answer = self.llm_client.chat(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "You answer questions using only the provided document context. "
                        "If the context does not contain the answer, say that clearly. "
                        "Keep the answer concise and cite source numbers when useful."
                    ),
                ),
                ChatMessage(role="user", content=prompt),
            ]
        )
        return AnswerDraft(answer=answer, used_llm=True)


def _build_grounded_prompt(question: str, sources: list[RetrievedChunk]) -> str:
    context_blocks = []
    for index, source in enumerate(sources, start=1):
        label = _source_label(source)
        context_blocks.append(
            f"[Source {index}: {label}; score={source.relevance_score:.2f}]\n{source.text}"
        )

    return (
        "Question:\n"
        f"{question}\n\n"
        "Document context:\n"
        f"{'\n\n'.join(context_blocks)}\n\n"
        "Answer from the context above. If the context is weak or missing, say so instead of guessing."
    )


def _build_retrieval_only_answer(sources: list[RetrievedChunk]) -> str:
    lines = [
        "This is a retrieval-only answer because LLM generation is not configured yet. Vector search found these relevant passages:",
        "",
    ]
    for index, source in enumerate(sources[:3], start=1):
        lines.append(
            f"{index}. {_source_label(source)} (score {source.relevance_score:.2f}): "
            f"{_snippet(source.text)}"
        )
    lines.extend(
        [
            "",
            "Use these passages as the provisional answer and check the source sections below for full context.",
        ]
    )
    return "\n".join(lines)


def _snippet(text: str, max_length: int = 450) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."


def _source_label(source: RetrievedChunk) -> str:
    metadata = source.metadata
    parts = [str(metadata.get("source_name", "unknown source"))]
    if metadata.get("page"):
        parts.append(f"page {metadata['page']}")
    if metadata.get("sheet"):
        parts.append(f"sheet {metadata['sheet']}")
    if metadata.get("row_range"):
        parts.append(f"rows {metadata['row_range']}")
    if metadata.get("chunk_index"):
        parts.append(f"chunk {metadata['chunk_index']}")
    return ", ".join(parts)
