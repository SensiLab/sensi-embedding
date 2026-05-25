from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class Modality(str, Enum):
    TEXT = "text"
    IMAGE = "image"


def generate_document_id() -> str:
    """Generate a random hex document ID using UUID4."""
    return uuid4().hex


class TextIngestRequest(BaseModel):
    text: str
    sender: str
    tags: list[str]
    metadata: dict[str, Any] = Field(default_factory=dict)
    document_id: str = Field(default_factory=generate_document_id)
    chunk: bool = True


class ImageIngestRequest(BaseModel):
    image_path: str
    text: str | None = None
    sender: str
    tags: list[str]
    source_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    document_id: str = Field(default_factory=generate_document_id)


class StoredRecord(BaseModel):
    id: str
    document_id: str
    sender: str
    modality: Modality
    tags: list[str]
    date: str
    source_path: str | None = None
    document: str
    metadata: dict[str, Any]


class SearchHit(BaseModel):
    id: str
    sender: str
    modality: Modality
    tags: list[str]
    date: str
    source_path: str | None = None
    document: str
    metadata: dict[str, Any]
    distance: float


class SearchResponse(BaseModel):
    hits: list[SearchHit] = Field(default_factory=list)


def build_base_metadata(
    *,
    document_id: str,
    sender: str,
    modality: Modality,
    mime_type: str | None,
    tags: list[str],
    attributes: dict[str, Any],
) -> dict[str, Any]:
    """Build the standard metadata dict shared by all stored records, merging custom attributes."""
    base: dict[str, Any] = {
        "document_id": document_id,
        "sender": sender,
        "modality": modality.value,
        "date": datetime.now(timezone.utc).isoformat(),
        "tags": ",".join(tags),
    }
    if mime_type:
        base["mime_type"] = mime_type
    base.update(attributes)
    return base
