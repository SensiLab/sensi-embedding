from __future__ import annotations

from typing import Any

import chromadb

from sensi_memory.config import Settings
from sensi_memory.models import SearchHit, SearchResponse, StoredRecord

_RESERVED_KEYS = {"document_id", "sender", "modality", "tags", "date", "source_path", "filename"}


def _split_tags(raw: str) -> list[str]:
    return [t for t in raw.split(",") if t] if raw else []


def _strip_reserved(meta: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in meta.items() if k not in _RESERVED_KEYS}


def _record_from_parts(record_id: str, document: str, metadata: dict[str, Any]) -> StoredRecord:
    return StoredRecord(
        id=record_id,
        document_id=str(metadata.get("document_id", "")),
        sender=str(metadata.get("sender", "")),
        modality=metadata.get("modality", "text"),
        tags=_split_tags(str(metadata.get("tags", ""))),
        date=str(metadata.get("date", "")),
        source_path=metadata.get("source_path") or None,
        document=document,
        metadata=_strip_reserved(metadata),
    )


def _hit_from_parts(record_id: str, document: str, metadata: dict[str, Any], distance: float) -> SearchHit:
    return SearchHit(
        id=record_id,
        sender=str(metadata.get("sender", "")),
        modality=metadata.get("modality", "text"),
        tags=_split_tags(str(metadata.get("tags", ""))),
        date=str(metadata.get("date", "")),
        source_path=metadata.get("source_path") or None,
        document=document,
        metadata=_strip_reserved(metadata),
        distance=distance,
    )


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
            _record_from_parts(record_id, document, metadata)
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

        documents = response.get("documents", [[]])[0]
        metadatas = response.get("metadatas", [[]])[0]
        distances = response.get("distances", [[]])[0]
        ids = response.get("ids", [[]])[0]

        hits = [
            _hit_from_parts(record_id, document, metadata or {}, float(distance))
            for record_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=False
            )
        ]

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
        return [
            _record_from_parts(record_id, document, metadata or {})
            for record_id, document, metadata in zip(
                result["ids"], result["documents"], result["metadatas"], strict=False
            )
        ]

    def get_all_with_embeddings(self) -> tuple[list[StoredRecord], list[list[float]]]:
        """Return every record in the collection together with their raw embedding vectors."""
        result = self._collection.get(include=["documents", "metadatas", "embeddings"])
        records = []
        embeddings: list[list[float]] = []
        for record_id, document, metadata, embedding in zip(
            result["ids"], result["documents"], result["metadatas"], result["embeddings"], strict=False
        ):
            records.append(_record_from_parts(record_id, document, metadata or {}))
            embeddings.append(list(embedding))
        return records, embeddings
