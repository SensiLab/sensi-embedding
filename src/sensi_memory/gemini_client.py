from __future__ import annotations

import random
import time
from typing import Any

import numpy as np
from google import genai
from google.genai import types

from sensi_memory.config import Settings


class EmbeddingError(RuntimeError):
    """Raised when embedding generation fails."""


class RateLimitExceededError(EmbeddingError):
    """Raised when the Gemini API remains rate-limited after retries."""


class GeminiEmbedder:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = genai.Client(api_key=settings.gemini_api_key)

    def embed_document_texts(self, texts: list[str]) -> list[list[float]]:
        return self._embed_contents(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query_text(self, text: str) -> list[float]:
        embeddings = self._embed_contents([text], task_type="RETRIEVAL_QUERY")
        return embeddings[0]

    def embed_image(self, image_bytes: bytes, mime_type: str, text: str | None) -> list[float]:
        parts: list[types.Part] = []
        if text:
            parts.append(types.Part(text=text))
        parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
        content = types.Content(parts=parts)
        embeddings = self._embed_contents([content], task_type="RETRIEVAL_DOCUMENT")
        return embeddings[0]

    def _embed_contents(self, contents: list[Any], task_type: str) -> list[list[float]]:
        delay = self._settings.base_retry_delay_seconds

        for attempt in range(self._settings.max_retries):
            try:
                response = self._client.models.embed_content(
                    model=self._settings.embedding_model,
                    contents=contents,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self._settings.embedding_dimensions,
                    ),
                )
                return [self._normalize_embedding(item.values) for item in response.embeddings]
            except Exception as exc:  # pragma: no cover - SDK-specific exception types vary.
                if not _is_rate_limited(exc):
                    raise EmbeddingError(str(exc)) from exc
                if attempt == self._settings.max_retries - 1:
                    raise RateLimitExceededError(str(exc)) from exc
                sleep_seconds = delay + random.uniform(0, delay)
                time.sleep(sleep_seconds)
                delay *= 2

        raise RateLimitExceededError("Gemini embedding retries exhausted.")

    def _normalize_embedding(self, values: list[float]) -> list[float]:
        if self._settings.embedding_dimensions == 3072:
            return values

        array = np.array(values, dtype=np.float32)
        norm = np.linalg.norm(array)
        if norm == 0:
            return values
        normalized = array / norm
        return normalized.tolist()


def _is_rate_limited(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True

    code = getattr(exc, "code", None)
    if code == 429:
        return True

    message = str(exc).lower()
    return "429" in message or "resource exhausted" in message or "rate limit" in message
