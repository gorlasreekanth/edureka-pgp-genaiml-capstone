from __future__ import annotations

import hashlib
import json

from src.models import SourceDocument, TextChunk


SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def chunk_documents(
    documents: list[SourceDocument],
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")

    chunks: list[TextChunk] = []
    for document in documents:
        text_chunks = recursive_chunk_text(document.text, chunk_size, chunk_overlap)
        for chunk_index, text in enumerate(text_chunks, start=1):
            metadata = {
                **document.metadata,
                "chunk_index": chunk_index,
                "chunk_count": len(text_chunks),
            }
            chunks.append(
                TextChunk(
                    id=_stable_chunk_id(text, metadata),
                    text=text,
                    metadata=metadata,
                )
            )
    return chunks


def recursive_chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []

    base_chunks = _split_recursively(stripped, chunk_size, SEPARATORS)
    if chunk_overlap == 0 or len(base_chunks) <= 1:
        return base_chunks

    chunks = [base_chunks[0]]
    for next_chunk in base_chunks[1:]:
        overlap = chunks[-1][-chunk_overlap:].strip()
        candidate = f"{overlap} {next_chunk}".strip() if overlap else next_chunk
        if len(candidate) > chunk_size:
            available_overlap = chunk_size - len(next_chunk) - 1
            if available_overlap > 0:
                candidate = f"{overlap[-available_overlap:]} {next_chunk}".strip()
            else:
                candidate = next_chunk[-chunk_size:]
        chunks.append(candidate)
    return chunks


def _split_recursively(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    if not separators:
        return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]

    separator = separators[0]
    if separator == "":
        return [text[index : index + chunk_size] for index in range(0, len(text), chunk_size)]

    if separator not in text:
        return _split_recursively(text, chunk_size, separators[1:])

    chunks: list[str] = []
    current = ""

    for piece in (part.strip() for part in text.split(separator) if part.strip()):
        candidate = f"{current}{separator}{piece}" if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = ""

        if len(piece) > chunk_size:
            chunks.extend(_split_recursively(piece, chunk_size, separators[1:]))
        else:
            current = piece

    if current:
        chunks.append(current)

    return chunks


def _stable_chunk_id(text: str, metadata: dict[str, object]) -> str:
    payload = {
        "text": text,
        "metadata": metadata,
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return digest[:20]
