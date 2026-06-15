from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


class EmbeddingProvider(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


def create_embedding_provider(model_name: str) -> EmbeddingProvider:
    normalized_model_name = model_name.strip().lower()
    if normalized_model_name in {"local-hash", "hash", "local"}:
        return LocalHashEmbeddings()
    return SentenceTransformerEmbeddings(model_name)


class LocalHashEmbeddings:
    """Small no-download embedding provider for local demos and tests."""

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero.")
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        normalized_text = text.strip()
        if not normalized_text:
            raise ValueError("Query text cannot be empty.")
        return self._embed(normalized_text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = _tokens(text)
        if not tokens:
            return vector

        features = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
        for feature in features:
            digest = hashlib.sha1(feature.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return vector
        return [value / magnitude for value in vector]


class SentenceTransformerEmbeddings:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._load_model().encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vectors = self.embed_documents([text])
        if not vectors:
            raise ValueError("Query text cannot be empty.")
        return vectors[0]

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())
