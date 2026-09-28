"""Embedding providers with lazy model loading."""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Any

from .config import Settings


class EmbeddingError(RuntimeError):
    """Raised when text cannot be embedded."""


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class Embedder:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any = None

    @property
    def dimension(self) -> int:
        return self.settings.embedding_dim

    def _hash_embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimension
            tokens = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
            if not tokens:
                tokens = [text]
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:8], "big") % self.dimension
                vector[index] += 1.0
            vectors.append(_normalize(vector))
        return vectors

    def _load_flag_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import FlagAutoModel
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise EmbeddingError("FlagEmbedding이 설치되어 있지 않습니다.") from exc

        kwargs: dict[str, Any] = {"use_fp16": self.settings.embedding_use_fp16}
        if self.settings.embedding_device and self.settings.embedding_device != "auto":
            kwargs["devices"] = [self.settings.embedding_device]
        if self.settings.embedding_model.lower().endswith("bge-m3"):
            kwargs["model_class"] = "encoder-only-m3"

        try:
            self._model = FlagAutoModel.from_finetuned(self.settings.embedding_model, **kwargs)
        except Exception as exc:
            raise EmbeddingError(
                f"임베딩 모델을 불러오지 못했습니다: {self.settings.embedding_model}. "
                "모델 경로, Hugging Face 접근, GPU/CPU 설정을 확인하세요."
            ) from exc
        return self._model

    @staticmethod
    def _as_lists(value: Any) -> list[list[float]]:
        if isinstance(value, dict):
            value = value.get("dense_vecs")
            if value is None:
                raise ValueError("임베딩 결과에 dense_vecs가 없습니다.")
        if hasattr(value, "tolist"):
            value = value.tolist()
        return [[float(item) for item in row] for row in value]

    def _flag_embed(self, texts: list[str], query: bool) -> list[list[float]]:
        model = self._load_flag_model()
        try:
            method_name = "encode_queries" if query else "encode_corpus"
            method = getattr(model, method_name, None) or getattr(model, "encode")
            result = method(texts, batch_size=self.settings.embedding_batch_size)
            vectors = self._as_lists(result)
        except Exception as exc:
            raise EmbeddingError("임베딩 계산에 실패했습니다.") from exc

        if not vectors or any(len(vector) != self.dimension for vector in vectors):
            actual = len(vectors[0]) if vectors else 0
            raise EmbeddingError(
                f"임베딩 차원이 설정과 다릅니다. 설정={self.dimension}, 실제={actual}. "
                "EMBEDDING_DIM을 모델에 맞게 조정하세요."
            )
        return [_normalize(vector) for vector in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.settings.embedding_provider == "hash":
            return self._hash_embed(texts)
        if self.settings.embedding_provider == "flag":
            return self._flag_embed(texts, query=False)
        raise EmbeddingError(f"지원하지 않는 EMBEDDING_PROVIDER입니다: {self.settings.embedding_provider}")

    def embed_query(self, text: str) -> list[float]:
        if self.settings.embedding_provider == "hash":
            return self._hash_embed([text])[0]
        if self.settings.embedding_provider == "flag":
            return self._flag_embed([text], query=True)[0]
        raise EmbeddingError(f"지원하지 않는 EMBEDDING_PROVIDER입니다: {self.settings.embedding_provider}")


@lru_cache(maxsize=4)
def get_embedder(settings: Settings) -> Embedder:
    return Embedder(settings)
