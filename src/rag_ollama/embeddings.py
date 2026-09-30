# =============================================================================
# 파일명: embeddings.py
# 경로: src/rag_ollama/embeddings.py
# 목적: FlagEmbedding과 결정적 hash 임베딩을 동일 인터페이스로 제공함
# 작성자: AI전략팀
# 작성일: 2026-09-30
# 수정일: 2026-09-30
# =============================================================================

"""FlagEmbedding과 결정적 hash 임베딩을 동일 인터페이스로 제공함"""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Any

from .config import Settings


class EmbeddingError(RuntimeError):
    """텍스트를 임베딩 벡터로 변환하지 못했음을 나타내는 예외임"""


def _normalize(vector: list[float]) -> list[float]:
    """검색 비교를 위해 벡터를 L2 단위 길이로 정규화함"""
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class Embedder:
    """설정에 따라 FlagEmbedding 또는 결정적 hash 임베딩을 제공함"""

    def __init__(self, settings: Settings) -> None:
        """설정을 보관하고 무거운 임베딩 모델은 최초 사용까지 지연함"""
        self.settings = settings
        self._model: Any = None

    @property
    def dimension(self) -> int:
        """저장소 벡터 차원과 맞춰야 하는 현재 임베딩 차원을 반환함"""
        return self.settings.embedding_dim

    def _hash_embed(self, texts: list[str]) -> list[list[float]]:
        """의미 검색이 아닌 연결·테스트용 결정적 hash 벡터를 생성함"""
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
        """FlagEmbedding 모델을 최초 사용 시 한 번만 지연 로딩함

        Raises:
            EmbeddingError: 의존성·모델 경로·실행 장치 문제로 로딩하지 못할 때 발생함
        """
        if self._model is not None:
            return self._model
        try:
            from FlagEmbedding import FlagAutoModel
        except ImportError as exc:  # NOTE: 선언된 의존성이 테스트 환경에서 누락된 경우만 해당함
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
        """FlagEmbedding 반환값을 중첩된 Python float 목록으로 정규화함"""
        if isinstance(value, dict):
            value = value.get("dense_vecs")
            if value is None:
                raise ValueError("임베딩 결과에 dense_vecs가 없습니다.")
        if hasattr(value, "tolist"):
            value = value.tolist()
        return [[float(item) for item in row] for row in value]

    def _flag_embed(self, texts: list[str], query: bool) -> list[list[float]]:
        """FlagEmbedding으로 문서 또는 질문 벡터를 계산하고 차원을 검증함

        Args:
            texts: 임베딩할 텍스트 목록임
            query: 질문 전용 인코더 사용 여부임

        Returns:
            list[list[float]]: 길이가 설정 차원과 일치하는 정규화 벡터임

        Raises:
            EmbeddingError: 모델 호출 실패나 차원 불일치가 있을 때 발생함
        """
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
        """문서 텍스트 목록을 검색 저장용 임베딩으로 변환함

        Raises:
            EmbeddingError: 설정된 임베딩 제공자를 사용할 수 없을 때 발생함
        """
        if self.settings.embedding_provider == "hash":
            return self._hash_embed(texts)
        if self.settings.embedding_provider == "flag":
            return self._flag_embed(texts, query=False)
        raise EmbeddingError(f"지원하지 않는 EMBEDDING_PROVIDER입니다: {self.settings.embedding_provider}")

    def embed_query(self, text: str) -> list[float]:
        """질문 한 건을 문서 검색과 호환되는 임베딩으로 변환함

        Raises:
            EmbeddingError: 설정된 임베딩 제공자를 사용할 수 없을 때 발생함
        """
        if self.settings.embedding_provider == "hash":
            return self._hash_embed([text])[0]
        if self.settings.embedding_provider == "flag":
            return self._flag_embed([text], query=True)[0]
        raise EmbeddingError(f"지원하지 않는 EMBEDDING_PROVIDER입니다: {self.settings.embedding_provider}")


@lru_cache(maxsize=4)
def get_embedder(settings: Settings) -> Embedder:
    """동일한 불변 설정에 대해 임베더 인스턴스를 재사용함"""
    return Embedder(settings)
