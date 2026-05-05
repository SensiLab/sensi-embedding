from __future__ import annotations

import mimetypes
from dataclasses import dataclass
from pathlib import Path

from sensi_memory.config import Settings
from sensi_memory.models import (
    ImageIngestRequest,
    Modality,
    TextIngestRequest,
    build_base_metadata,
)

SUPPORTED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
}


@dataclass(frozen=True, slots=True)
class NormalizedTextChunk:
    record_id: str
    document: str
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class NormalizedImageInput:
    record_id: str
    document: str
    metadata: dict[str, object]
    image_bytes: bytes
    mime_type: str


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into paragraph-aligned chunks no longer than max_chars characters."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("Text input cannot be empty.")
    if len(stripped) <= max_chars:
        return [stripped]

    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for paragraph in stripped.splitlines():
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        paragraph_length = len(paragraph)
        separator_length = 1 if current_chunk else 0
        if current_length + paragraph_length + separator_length <= max_chars:
            current_chunk.append(paragraph)
            current_length += paragraph_length + separator_length
            continue

        if current_chunk:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_length = 0

        if paragraph_length <= max_chars:
            current_chunk.append(paragraph)
            current_length = paragraph_length
            continue

        start = 0
        while start < paragraph_length:
            chunks.append(paragraph[start : start + max_chars])
            start += max_chars

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks


def normalize_text_request(
    request: TextIngestRequest,
    settings: Settings,
) -> list[NormalizedTextChunk]:
    """Convert a TextIngestRequest into a list of NormalizedTextChunk records with per-chunk metadata."""
    chunks = (
        chunk_text(request.text, settings.max_text_chunk_chars)
        if request.chunk
        else [request.text.strip()]
    )
    normalized_chunks: list[NormalizedTextChunk] = []

    for index, chunk in enumerate(chunks):
        metadata = build_base_metadata(
            document_id=request.document_id,
            modality=Modality.TEXT,
            mime_type="text/plain",
            metadata=request.metadata,
        )
        metadata["chunk_index"] = index
        metadata["chunk_count"] = len(chunks)
        normalized_chunks.append(
            NormalizedTextChunk(
                record_id=f"{request.document_id}:chunk:{index}",
                document=chunk,
                metadata=metadata,
            )
        )

    return normalized_chunks


def normalize_image_request(request: ImageIngestRequest) -> NormalizedImageInput:
    """Validate the image path, read its bytes, and return a NormalizedImageInput ready for embedding."""
    image_path = Path(request.image_path).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not image_path.is_file():
        raise ValueError(f"Image path is not a file: {image_path}")

    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
        raise ValueError(
            "Unsupported image type. Supported formats are PNG and JPEG."
        )

    image_bytes = image_path.read_bytes()
    document = request.text.strip() if request.text else image_path.name
    metadata = build_base_metadata(
        document_id=request.document_id,
        modality=Modality.IMAGE,
        mime_type=mime_type,
        metadata=request.metadata,
    )
    metadata["source_path"] = str(image_path)
    metadata["filename"] = image_path.name

    return NormalizedImageInput(
        record_id=request.document_id,
        document=document,
        metadata=metadata,
        image_bytes=image_bytes,
        mime_type=mime_type,
    )