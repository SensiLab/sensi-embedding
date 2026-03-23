from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"


class IngestMetadata(BaseModel):
    tags: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


def generate_document_id() -> str:
    return uuid4().hex


class TextIngestRequest(BaseModel):
    text: str
    metadata: IngestMetadata = Field(default_factory=IngestMetadata)
    document_id: str = Field(default_factory=generate_document_id)
    chunk: bool = True


class ImageIngestRequest(BaseModel):
    image_path: str
    text: str | None = None
    metadata: IngestMetadata = Field(default_factory=IngestMetadata)
    document_id: str = Field(default_factory=generate_document_id)


class StoredRecord(BaseModel):
    id: str
    document_id: str
    modality: Modality
    document: str
    metadata: dict[str, Any]


class SearchHit(BaseModel):
    id: str
    document: str
    metadata: dict[str, Any]
    distance: float


class SearchResponse(BaseModel):
    hits: list[SearchHit] = Field(default_factory=list)


def build_base_metadata(
    *,
    document_id: str,
    modality: Modality,
    mime_type: str | None,
    metadata: IngestMetadata,
) -> dict[str, Any]:
    base_metadata: dict[str, Any] = {
        "document_id": document_id,
        "modality": modality.value,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tags": list(metadata.tags),
    }
    if mime_type:
        base_metadata["mime_type"] = mime_type
    if metadata.attributes:
        base_metadata.update(metadata.attributes)
    return base_metadata