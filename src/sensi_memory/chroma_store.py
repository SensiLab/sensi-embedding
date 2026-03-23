from __future__ import annotations

from typing import Any

import chromadb

from sensi_memory.config import Settings
from sensi_memory.models import SearchHit, SearchResponse, StoredRecord


class ChromaMemoryStore:
    def __init__(self, settings: Settings) -> None:
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
        embeddings: list[list[float]],
    ) -> list[StoredRecord]:
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
        metadata_filter: dict[str, Any] | None = None,
    ) -> SearchResponse:
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
