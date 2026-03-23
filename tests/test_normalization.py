from pathlib import Path

import pytest

from sensi_memory.config import Settings
from sensi_memory.models import ImageIngestRequest, IngestMetadata, TextIngestRequest
from sensi_memory.normalization import chunk_text, normalize_image_request, normalize_text_request


def test_chunk_text_splits_large_paragraphs() -> None:
    chunks = chunk_text("abcdefghij", max_chars=4)

    assert chunks == ["abcd", "efgh", "ij"]


def test_normalize_text_request_adds_chunk_metadata() -> None:
    settings = Settings(gemini_api_key="test-key", max_text_chunk_chars=5)
    request = TextIngestRequest(
        text="alpha\nbeta",
        metadata=IngestMetadata(tags=["note"], attributes={"user_id": "u1"}),
        document_id="doc-1",
    )

    chunks = normalize_text_request(request, settings)

    assert [chunk.record_id for chunk in chunks] == ["doc-1:chunk:0", "doc-1:chunk:1"]
    assert chunks[0].metadata["chunk_count"] == 2
    assert chunks[0].metadata["tags"] == ["note"]
    assert chunks[0].metadata["user_id"] == "u1"


def test_normalize_image_request_reads_png(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"fake-png")

    request = ImageIngestRequest(
        image_path=str(image_path),
        text="diagram",
        document_id="img-1",
    )

    normalized = normalize_image_request(request)

    assert normalized.record_id == "img-1"
    assert normalized.mime_type == "image/png"
    assert normalized.document == "diagram"
    assert normalized.metadata["source_path"] == str(image_path.resolve())


def test_normalize_image_request_rejects_unsupported_type(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.gif"
    image_path.write_bytes(b"GIF89a")

    request = ImageIngestRequest(image_path=str(image_path))

    with pytest.raises(ValueError):
        normalize_image_request(request)
