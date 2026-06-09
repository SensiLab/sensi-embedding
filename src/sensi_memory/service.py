from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sensi_memory.chroma_store import ChromaMemoryStore
from sensi_memory.config import Settings
from sensi_memory.gemini_client import GeminiEmbedder
from sensi_memory.models import (
    EmbeddingResult,
    EmbeddingsResponse,
    ImageIngestRequest,
    SearchResponse,
    StoredRecord,
    TextIngestRequest,
    generate_document_id,
)
from sensi_memory.normalization import normalize_image_request, normalize_text_request


@dataclass
class GraphNode:
    id: str
    source_path: str | None
    object_path: str | None


@dataclass
class GraphEdge:
    source: str
    target: str
    distance: float


@dataclass
class GraphData:
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class MemoryService:
    def __init__(
        self,
        *,
        settings: Settings,
        embedder: GeminiEmbedder,
        store: ChromaMemoryStore) -> None:
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
        sender: str,
        tags: list[str],
        metadata: dict[str, Any] | None = None,
        document_id: str | None = None,
        chunk: bool = True) -> list[StoredRecord]:
        """Chunk, embed, and store text; returns one StoredRecord per chunk."""

        request = TextIngestRequest(
            text=text,
            sender=sender,
            tags=tags,
            metadata=metadata or {},
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
        sender: str,
        tags: list[str],
        source_path: str,
        object_path: str | None = None,
        metadata: dict[str, Any] | None = None,
        document_id: str | None = None) -> StoredRecord:
        """Embed a PNG or JPEG image (with optional caption) and store it as a single record."""

        request = ImageIngestRequest(
            image_path=image_path,
            text=text,
            sender=sender,
            tags=tags,
            source_path=source_path,
            object_path=object_path,
            metadata=metadata or {},
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
        metadata_filter: dict[str, Any] | None = None) -> SearchResponse:
        """Embed the query and retrieve the top-k most similar records from the store."""

        if not text.strip():
            raise ValueError("Search text cannot be empty.")
        query_embedding = self._embedder.embed_query_text(text.strip())
        return self._store.query(
            embedding=query_embedding,
            top_k=top_k or self._settings.default_top_k,
            metadata_filter=metadata_filter,
        )

    def search_similar_by_id(
        self,
        record_id: str,
        *,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> SearchResponse:
        """Find the top-k records most similar to the stored embedding for record_id."""
        embedding = self._store.get_embedding_by_id(record_id)
        if embedding is None:
            raise ValueError(f"Record not found: {record_id}")
        effective_k = top_k or self._settings.default_top_k
        response = self._store.query(
            embedding=embedding,
            top_k=effective_k + 1,
            metadata_filter=metadata_filter,
        )
        hits = [h for h in response.hits if h.id != record_id][:effective_k]
        return SearchResponse(hits=hits)

    def search_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        *,
        top_k: int | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> SearchResponse:
        """Embed an image and retrieve the top-k most similar records without storing it."""
        embedding = self._embedder.embed_image(
            image_bytes=image_bytes,
            mime_type=mime_type,
            text=None,
        )
        return self._store.query(
            embedding=embedding,
            top_k=top_k or self._settings.default_top_k,
            metadata_filter=metadata_filter,
        )

    def count(self) -> int:
        """Return the total number of records in the store."""
        return self._store.count()

    def get_records_by_sender_since(self, sender: str, days: int) -> list[StoredRecord]:
        """Return all records for the given sender created within the last `days` days."""
        return self._store.get_records_by_sender_since(sender, days)

    def get_all_senders(self) -> list[str]:
        """Return a sorted, deduplicated list of every sender in the store."""
        return self._store.get_all_senders()

    def export_all(self) -> list[StoredRecord]:
        """Return all stored records without embeddings."""
        return self._store.get_all_records()

    def get_embeddings_for_records(self, record_ids: list[str]) -> EmbeddingsResponse:
        """Return the stored embedding vector for each requested record ID."""
        embeddings_map = self._store.get_embeddings_by_ids(record_ids)
        return EmbeddingsResponse(
            results=[EmbeddingResult(id=rid, embedding=embeddings_map[rid]) for rid in record_ids]
        )

    def get_graph_data(self, threshold: float = 0.4) -> GraphData:
        """Return image nodes and similarity edges for all stored image records.

        Edges connect pairs whose cosine distance is below threshold.
        """
        all_records, all_embeddings = self._store.get_all_with_embeddings()

        image_records = [r for r, e in zip(all_records, all_embeddings) if r.modality == "image"]
        image_embeddings = [e for r, e in zip(all_records, all_embeddings) if r.modality == "image"]

        nodes = [
            GraphNode(id=r.id, source_path=r.source_path, object_path=r.object_path)
            for r in image_records
        ]

        edges: list[GraphEdge] = []
        if len(image_embeddings) >= 2:
            matrix = np.array(image_embeddings, dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            normed = matrix / np.maximum(norms, 1e-10)
            distances = 1.0 - (normed @ normed.T)
            i_idx, j_idx = np.where(np.triu(distances < threshold, k=1))
            edges = [
                GraphEdge(
                    source=image_records[int(i)].id,
                    target=image_records[int(j)].id,
                    distance=float(distances[int(i), int(j)]),
                )
                for i, j in zip(i_idx, j_idx)
            ]

        return GraphData(nodes=nodes, edges=edges)
