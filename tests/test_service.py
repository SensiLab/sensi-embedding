from pathlib import Path

import pytest

from sensi_memory.config import Settings
from sensi_memory.models import SearchResponse, StoredRecord
from sensi_memory.service import MemoryService

_RESERVED = {"document_id", "sender", "modality", "tags", "date", "source_path", "filename"}


class StubEmbedder:
    def __init__(self) -> None:
        self.document_texts: list[list[str]] = []
        self.query_texts: list[str] = []
        self.image_calls: list[tuple[bytes, str, str | None]] = []

    def embed_document_texts(self, texts: list[str]) -> list[list[float]]:
        self.document_texts.append(texts)
        return [[0.1, 0.2] for _ in texts]

    def embed_query_text(self, text: str) -> list[float]:
        self.query_texts.append(text)
        return [0.3, 0.4]

    def embed_image(self, image_bytes: bytes, mime_type: str, text: str | None) -> list[float]:
        self.image_calls.append((image_bytes, mime_type, text))
        return [0.5, 0.6]


class StubStore:
    def __init__(self) -> None:
        self.upsert_calls: list[dict[str, object]] = []
        self.query_calls: list[dict[str, object]] = []

    def upsert_records(self, **kwargs: object) -> list[StoredRecord]:
        self.upsert_calls.append(kwargs)
        ids = kwargs["ids"]
        documents = kwargs["documents"]
        metadatas = kwargs["metadatas"]
        return [
            StoredRecord(
                id=record_id,
                document_id=str(metadata.get("document_id", "")),
                sender=str(metadata.get("sender", "")),
                modality=metadata.get("modality", "text"),
                tags=[t for t in str(metadata.get("tags", "")).split(",") if t],
                date=str(metadata.get("date", "")),
                source_path=metadata.get("source_path") or None,
                document=document,
                metadata={k: v for k, v in metadata.items() if k not in _RESERVED},
            )
            for record_id, document, metadata in zip(ids, documents, metadatas, strict=True)
        ]

    def query(self, **kwargs: object) -> SearchResponse:
        self.query_calls.append(kwargs)
        return SearchResponse()


def test_ingest_text_embeds_and_stores_chunks() -> None:
    settings = Settings(gemini_api_key="test-key", max_text_chunk_chars=5)
    embedder = StubEmbedder()
    store = StubStore()
    service = MemoryService(settings=settings, embedder=embedder, store=store)

    records = service.ingest_text(
        "alpha\nbeta",
        sender="test-sender",
        tags=["memory"],
        document_id="doc-1",
    )

    assert len(records) == 2
    assert records[0].sender == "test-sender"
    assert records[0].tags == ["memory"]
    assert records[0].modality == "text"
    assert "sender" not in records[0].metadata
    assert "tags" not in records[0].metadata
    assert embedder.document_texts == [["alpha", "beta"]]
    assert store.upsert_calls[0]["ids"] == ["doc-1:chunk:0", "doc-1:chunk:1"]


def test_ingest_image_embeds_and_stores(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png")
    settings = Settings(gemini_api_key="test-key")
    embedder = StubEmbedder()
    store = StubStore()
    service = MemoryService(settings=settings, embedder=embedder, store=store)

    record = service.ingest_image(
        str(image_path),
        text="a diagram",
        sender="test-sender",
        tags=["diagram"],
        source_path="/original/images/sample.png",
        document_id="img-1",
    )

    assert record.id == "img-1"
    assert record.sender == "test-sender"
    assert record.tags == ["diagram"]
    assert record.modality == "image"
    assert "sender" not in record.metadata
    assert embedder.image_calls[0][1] == "image/png"
    assert store.upsert_calls[0]["documents"] == ["a diagram"]


def test_search_text_embeds_query_and_hits_store() -> None:
    settings = Settings(gemini_api_key="test-key", default_top_k=7)
    embedder = StubEmbedder()
    store = StubStore()
    service = MemoryService(settings=settings, embedder=embedder, store=store)

    response = service.search_text("find this")

    assert response.hits == []
    assert embedder.query_texts == ["find this"]
    assert store.query_calls[0]["top_k"] == 7


def test_search_text_rejects_empty_text() -> None:
    settings = Settings(gemini_api_key="test-key")
    service = MemoryService(settings=settings, embedder=StubEmbedder(), store=StubStore())

    with pytest.raises(ValueError):
        service.search_text("   ")
