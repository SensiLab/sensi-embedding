from __future__ import annotations

from typing import Any

from sensi_memory.chroma_store import ChromaMemoryStore
from sensi_memory.config import Settings
from sensi_memory.gemini_client import GeminiEmbedder
from sensi_memory.models import (
    ImageIngestRequest,
    IngestMetadata,
    SearchResponse,
    StoredRecord,
    TextIngestRequest,
    generate_document_id,
)
from sensi_memory.normalization import normalize_image_request, normalize_text_request


class MemoryService:
    def __init__(
        self,
        *,
        settings: Settings,
        embedder: GeminiEmbedder,
        store: ChromaMemoryStore,
    ) -> None:
        """Wire together the embedder and store with their shared settings."""
        self._settings = settings
        self._embedder = embedder
        self._store = store

    @classmethod
    def from_settings(cls, settings: Settings) -> "MemoryService":
        """Construct a MemoryService with a Gemini embedder and ChromaDB store from settings."""
        return cls(
            settings=settings,
            embedder=GeminiEmbedder(settings),
            store=ChromaMemoryStore(settings),
        )

    def ingest_text(
        self,
        text: str,
        *,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        document_id: str | None = None,
        chunk: bool = True,
    ) -> list[StoredRecord]:
        """Chunk, embed, and store text; returns one StoredRecord per chunk."""
        request = TextIngestRequest(
            text=text,
            metadata=IngestMetadata(tags=tags or [], attributes=metadata or {}),
            document_id=document_id or generate_document_id(),
            chunk=chunk,
        )
        normalized_chunks = normalize_text_request(request, self._settings)
        documents = [item.document for item in normalized_chunks]
        embeddings = self._embedder.embed_document_texts(documents)
        return self._store.upsert_records(
            ids=[item.record_id for item in normalized_chunks],
            documents=documents,
            metadatas=[item.metadata for item in normalized_chunks],
            embeddings=embeddings,
        )

    def ingest_image(
        self,
        image_path: str,
        *,
        text: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        document_id: str | None = None,
    ) -> StoredRecord:
        """Embed a PNG or JPEG image (with optional caption) and store it as a single record."""
        request = ImageIngestRequest(
            image_path=image_path,
            text=text,
            metadata=IngestMetadata(tags=tags or [], attributes=metadata or {}),
            document_id=document_id or generate_document_id(),
        )
        normalized_image = normalize_image_request(request)
        embedding = self._embedder.embed_image(
            image_bytes=normalized_image.image_bytes,
            mime_type=normalized_image.mime_type,
            text=request.text,
        )
        [record] = self._store.upsert_records(
            ids=[normalized_image.record_id],
            documents=[normalized_image.document],
            metadatas=[normalized_image.metadata],
            embeddings=[embedding],
        )
        return record

    def search_text(
        self,
        text: str,
        *,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> SearchResponse:
        """Embed the query and retrieve the top-k most similar records from the store."""
        if not text.strip():
            raise ValueError("Search text cannot be empty.")
        query_embedding = self._embedder.embed_query_text(text.strip())
        return self._store.query(
            embedding=query_embedding,
            top_k=top_k or self._settings.default_top_k,
            metadata_filter=metadata_filter,
        )
