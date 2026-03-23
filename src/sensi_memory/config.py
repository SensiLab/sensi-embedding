from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    gemini_api_key: str
    chroma_path: Path = Path("./local_storage")
    chroma_collection: str = "sensi_memories"
    embedding_model: str = "gemini-embedding-2-preview"
    embedding_dimensions: int = 768
    default_top_k: int = 5
    max_retries: int = 4
    base_retry_delay_seconds: float = 1.0
    max_text_chunk_chars: int = 2_000

    @classmethod
    def from_env(cls) -> "Settings":
        gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required.")

        chroma_path = Path(os.getenv("SENSI_CHROMA_PATH", "./local_storage"))
        chroma_collection = os.getenv("SENSI_CHROMA_COLLECTION", "sensi_memories")
        embedding_model = os.getenv(
            "SENSI_EMBEDDING_MODEL",
            "gemini-embedding-2-preview",
        )
        embedding_dimensions = int(os.getenv("SENSI_EMBEDDING_DIMENSIONS", "768"))
        default_top_k = int(os.getenv("SENSI_DEFAULT_TOP_K", "5"))

        return cls(
            gemini_api_key=gemini_api_key,
            chroma_path=chroma_path,
            chroma_collection=chroma_collection,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            default_top_k=default_top_k,
        )
