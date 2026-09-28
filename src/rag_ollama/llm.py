"""OpenAI-compatible chat completion client.

The service can point this at a hosted API or a local server that exposes a
`/v1/chat/completions`-compatible endpoint. It is optional: retrieval still
works when no LLM is configured.
"""

from __future__ import annotations

from typing import Any

import httpx

from .config import Settings


class LLMError(RuntimeError):
    """Raised when the generation backend fails."""


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @property
    def configured(self) -> bool:
        return bool(self.settings.llm_base_url and self.settings.llm_model)

    def complete(self, question: str, context: str) -> str | None:
        if not self.configured:
            return None

        base_url = self.settings.llm_base_url.rstrip("/")  # type: ignore[union-attr]
        endpoint = base_url if base_url.endswith("/chat/completions") else f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.llm_api_key}"

        system_prompt = (
            "당신은 근거 기반 RAG 도우미다. 제공된 CONTEXT만 사용해 답변하라. "
            "근거가 부족하면 모른다고 말하고 추측하지 마라. "
            "답변에 사용한 근거 번호를 [1], [2] 형식으로 표시하라."
        )
        payload: dict[str, Any] = {
            "model": self.settings.llm_model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"},
            ],
        }
        try:
            response = httpx.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.settings.llm_timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return str(content).strip()
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError("LLM 응답을 받지 못했습니다. LLM_BASE_URL/LLM_MODEL을 확인하세요.") from exc
