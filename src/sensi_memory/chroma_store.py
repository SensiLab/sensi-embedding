from __future__ import annotations

from typing import Any

import chromadb

from sensi_memory.config import Settings
from sensi_memory.models import SearchHit, SearchResponse, StoredRecord


class ChromaMemoryStore:
    def __init__(self, settings: Settings) -> None:
        """Open (or create) the ChromaDB persistent collection at the configured path."""
        self._client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert_records(
        self,
        *,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]],
        embeddings: list[list[float]]) -> list[StoredRecord]:
        """Write or overwrite records in the collection and return them as StoredRecord objects."""

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        return [
            StoredRecord(
                id=record_id,
                document_id=str(metadata["document_id"]),
                modality=metadata["modality"],
                document=document,
                metadata=metadata,
            )
            for record_id, document, metadata in zip(ids, documents, metadatas, strict=True)
        ]

    def query(
        self,
        *,
        embedding: list[float],
        top_k: int,
        metadata_filter: dict[str, Any] | None = None) -> SearchResponse:
        """Find the top_k records nearest to the given embedding, with optional metadata filtering."""

        query_kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        
        if metadata_filter:
            query_kwargs["where"] = metadata_filter

        response = self._collection.query(**query_kwargs)

        hits: list[SearchHit] = []
        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        ids = response.get("ids", [[]])[0]

        for record_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=False,
        ):
            hits.append(
                SearchHit(
                    id=record_id,
                    document=document,
                    metadata=metadata or {},
                    distance=float(distance),
                )
            )

        return SearchResponse(hits=hits)

    def get_embedding_by_id(self, record_id: str) -> list[float] | None:
        """Return the stored embedding vector for a single record ID, or None if not found."""
        result = self._collection.get(ids=[record_id], include=["embeddings"])
        embeddings = result.get("embeddings")
        if embeddings is None or len(embeddings) == 0:
            return None
        return list(embeddings[0])

    def get_all_records(self) -> list[StoredRecord]:
        """Return every record in the collection, excluding embeddings."""
        result = self._collection.get(include=["documents", "metadatas"])
        records = []
        for record_id, document, metadata in zip(
            result["ids"], result["documents"], result["metadatas"], strict=False
        ):
            metadata = metadata or {}
            records.append(
                StoredRecord(
                    id=record_id,
                    document_id=str(metadata.get("document_id", "")),
                    modality=metadata.get("modality", ""),
                    document=document,
                    metadata=metadata,
                )
            )
        return records
