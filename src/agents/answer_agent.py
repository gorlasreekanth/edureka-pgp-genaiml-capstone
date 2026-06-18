from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol

from src.agents.query_planner import QueryPlan
from src.llm.ollama import ChatMessage
from src.models import RetrievedChunk, normalize_source_indices


class ChatClient(Protocol):
    @property
    def is_configured(self) -> bool:
        ...

    @property
    def configuration_issue(self) -> str | None:
        ...

    def chat(self, messages: list[ChatMessage]) -> str:
        ...


@dataclass(frozen=True)
class AnswerDraft:
    answer: str
    used_llm: bool
    used_source_indices: list[int] = field(default_factory=list)
    source_citation_missing: bool = False
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
            issue = self.llm_client.configuration_issue or "LLM generation is not configured yet."
            return AnswerDraft(
                answer=_build_retrieval_only_answer(sources),
                used_llm=False,
                used_source_indices=list(range(1, min(len(sources), 3) + 1)),
                warnings=[
                    f"{issue} This is a retrieval-only answer from matching source chunks."
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
                        "Keep the answer concise and cite source numbers like [Source 1]. "
                        "End every response with exactly one final line: "
                        "'Used sources: 1, 2' or 'Used sources: none'. "
                        "List only the source numbers you actually relied on."
                    ),
                ),
                ChatMessage(role="user", content=prompt),
            ]
        )
        answer_text, used_source_indices, citation_missing = _extract_answer_and_used_sources(
            answer,
            source_count=len(sources),
        )
        return AnswerDraft(
            answer=answer_text,
            used_llm=True,
            used_source_indices=used_source_indices,
            source_citation_missing=citation_missing,
        )


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
        "Answer from the context above. If the context is weak or missing, say so instead of guessing.\n"
        "Use inline citations such as [Source 1] for facts from the context. "
        "End with a final line in this exact format: Used sources: 1, 2"
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


_USED_SOURCES_LINE_RE = re.compile(
    r"(?im)^\s*(?:used\s+sources?|sources\s+used)\s*:\s*(?P<value>.*?)\s*$"
)
_INLINE_SOURCE_RE = re.compile(r"(?i)\[?\bsource\s*#?\s*(\d+)\]?")


def _extract_answer_and_used_sources(
    raw_answer: str,
    source_count: int,
) -> tuple[str, list[int], bool]:
    match = None
    for candidate in _USED_SOURCES_LINE_RE.finditer(raw_answer):
        match = candidate

    if match:
        used_value = match.group("value").strip()
        answer_text = (raw_answer[: match.start()] + raw_answer[match.end() :]).strip()
        if _means_no_sources(used_value):
            return answer_text, [], False
        return answer_text, normalize_source_indices(_extract_numbers(used_value), source_count), False

    inline_sources = [
        int(match.group(1))
        for match in _INLINE_SOURCE_RE.finditer(raw_answer)
        if match.group(1).isdigit()
    ]
    return raw_answer.strip(), normalize_source_indices(inline_sources, source_count), not inline_sources


def _means_no_sources(value: str) -> bool:
    normalized = value.strip().lower().rstrip(".")
    return normalized in {"none", "no sources", "n/a", "not applicable"}


def _extract_numbers(value: str) -> list[int]:
    return [int(match) for match in re.findall(r"\d+", value)]


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
