from typing import Any

from sensi_memory.chroma_store import ChromaMemoryStore
from sensi_memory.config import Settings


class FakeCollection:
    def __init__(self) -> None:
        self.upsert_kwargs: dict[str, Any] | None = None
        self.query_kwargs: dict[str, Any] | None = None

    def upsert(self, **kwargs: Any) -> None:
        self.upsert_kwargs = kwargs

    def query(self, **kwargs: Any) -> dict[str, list[list[Any]]]:
        self.query_kwargs = kwargs
        return {
            "ids": [["rec-1"]],
            "documents": [["hello world"]],
            "metadatas": [[{"document_id": "doc-1", "modality": "text"}]],
            "distances": [[0.12]],
        }


class FakeClient:
    def __init__(self, collection: FakeCollection) -> None:
        self.collection = collection
        self.path: str | None = None

    def get_or_create_collection(self, *, name: str, metadata: dict[str, Any]) -> FakeCollection:
        assert name == "test_collection"
        assert metadata == {"hnsw:space": "cosine"}
        return self.collection


def test_chroma_store_upserts_and_queries(monkeypatch) -> None:
    collection = FakeCollection()

    def fake_persistent_client(*, path: str) -> FakeClient:
        client = FakeClient(collection)
        client.path = path
        return client

    monkeypatch.setattr("sensi_memory.chroma_store.chromadb.PersistentClient", fake_persistent_client)

    settings = Settings(
        gemini_api_key="test-key",
        chroma_collection="test_collection",
    )
    store = ChromaMemoryStore(settings)

    records = store.upsert_records(
        ids=["rec-1"],
        documents=["hello world"],
        metadatas=[{"document_id": "doc-1", "modality": "text"}],
        embeddings=[[0.1, 0.2]],
    )
    response = store.query(embedding=[0.1, 0.2], top_k=3, metadata_filter={"modality": "text"})

    assert records[0].id == "rec-1"
    assert collection.upsert_kwargs is not None
    assert collection.query_kwargs is not None
    assert collection.query_kwargs["where"] == {"modality": "text"}
    assert response.hits[0].id == "rec-1"