"""FastAPI HTTP server for sensi-memory, exposing ingest and search over REST."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from sensi_memory.config import Settings
from sensi_memory.gemini_client import EmbeddingError
from sensi_memory.models import EmbeddingsResponse, SearchResponse, StoredRecord
from sensi_memory.service import MemoryService


class GraphNodeResponse(BaseModel):
    id: str
    source_path: str | None
    object_path: str | None


class GraphEdgeResponse(BaseModel):
    source: str
    target: str
    distance: float


class GraphResponse(BaseModel):
    nodes: list[GraphNodeResponse]
    edges: list[GraphEdgeResponse]


class TextIngestRequest(BaseModel):
    """Request body for the POST /ingest/text endpoint."""

    text: str
    sender: str
    tags: list[str]
    metadata: dict[str, Any] | None = Field(default=None)
    document_id: str | None = Field(default=None)
    chunk: bool = Field(default=True)


class SearchRequest(BaseModel):
    """Request body for the POST /search endpoint."""

    text: str
    top_k: int | None = Field(default=None, ge=1)
    sender: str | None = Field(default=None)
    modality: str | None = Field(default=None)
    date_from: datetime | None = Field(default=None)
    date_to: datetime | None = Field(default=None)
    metadata_filter: dict[str, Any] | None = Field(default=None)


def _iso(dt: datetime) -> str:
    """Normalise a datetime to UTC and return an ISO 8601 string matching stored date format."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt.isoformat()


def _build_filter(
    sender: str | None,
    modality: str | None,
    date_from: datetime | None,
    date_to: datetime | None,
    metadata_filter: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Merge sender/modality/date-range constraints with an optional caller-supplied filter using $and."""
    conditions: list[dict[str, Any]] = []
    if sender is not None:
        conditions.append({"sender": {"$eq": sender}})
    if modality is not None:
        conditions.append({"modality": {"$eq": modality}})
    if date_from is not None:
        conditions.append({"date": {"$gte": _iso(date_from)}})
    if date_to is not None:
        conditions.append({"date": {"$lte": _iso(date_to)}})
    if metadata_filter:
        conditions.append(metadata_filter)
    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize a shared MemoryService on startup and clean up on shutdown."""
    settings = Settings.from_env()
    app.state.service = MemoryService.from_settings(settings)
    yield


app = FastAPI(title="Sensi Memory API", version="0.1.0", lifespan=lifespan)


def _service() -> MemoryService:
    """Return the shared MemoryService instance stored on the app state."""
    return app.state.service


@app.get("/health", status_code=status.HTTP_200_OK)
def health() -> dict[str, str]:
    """Return a simple liveness check response."""
    return {"status": "ok"}


@app.get("/count", status_code=status.HTTP_200_OK)
def count() -> dict[str, int]:
    """Return the total number of records stored in the database."""
    return {"count": _service().count()}


@app.get("/senders", status_code=status.HTTP_200_OK)
def senders() -> dict[str, list[str]]:
    """Return a sorted, deduplicated list of every sender value present in the database."""
    return {"senders": _service().get_all_senders()}


@app.get("/records", response_model=list[StoredRecord], status_code=status.HTTP_200_OK)
def records_by_sender(
    sender: str = Query(..., description="Sender name to filter by"),
    days: int = Query(default=7, ge=1, description="Number of past days to include"),
) -> list[StoredRecord]:
    """Return all records for a given sender created within the last N days."""
    return _service().get_records_by_sender_since(sender, days)


@app.post("/ingest/text", response_model=list[StoredRecord], status_code=status.HTTP_200_OK)
def ingest_text(body: TextIngestRequest) -> list[StoredRecord]:
    """Embed and store text in the vector database, optionally chunking large inputs."""
    try:
        return _service().ingest_text(
            body.text,
            sender=body.sender,
            tags=body.tags,
            metadata=body.metadata,
            document_id=body.document_id,
            chunk=body.chunk,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except EmbeddingError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


@app.post("/ingest/image", response_model=StoredRecord, status_code=status.HTTP_200_OK)
async def ingest_image(
    file: UploadFile = File(...),
    sender: str = Form(...),
    tags: str = Form(...),
    source_path: str = Form(...),
    text: str | None = Form(default=None),
    object_path: str | None = Form(default=None),
    metadata: str | None = Form(default=None),
    document_id: str | None = Form(default=None),
) -> StoredRecord:
    """Embed and store an uploaded image, writing it to a tempfile for path-based normalization.

    sender: who or what is ingesting this record.
    tags: comma-separated string (e.g. "photo,nature").
    source_path: original path or URI of the source screenshot/image as known to the sender (required).
    object_path: path to an artifact extracted from the source image, if stored alongside it (optional).
    metadata: JSON object string (e.g. '{"source": "camera"}').
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    metadata_dict: dict[str, Any] | None = None
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
            if not isinstance(metadata_dict, dict):
                raise ValueError("metadata must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    original_filename = file.filename or "upload"
    suffix = ("." + original_filename.rsplit(".", 1)[-1]) if "." in original_filename else ""
    image_bytes = await file.read()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        return _service().ingest_image(
            tmp_path,
            text=text,
            sender=sender,
            tags=tag_list,
            source_path=source_path,
            object_path=object_path,
            metadata=metadata_dict,
            document_id=document_id,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except EmbeddingError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))
    finally:
        os.unlink(tmp_path)


@app.post("/search", response_model=SearchResponse, status_code=status.HTTP_200_OK)
def search(body: SearchRequest) -> SearchResponse:
    """Search the vector database for records semantically similar to the query text."""
    try:
        return _service().search_text(
            body.text,
            top_k=body.top_k,
            metadata_filter=_build_filter(body.sender, body.modality, body.date_from, body.date_to, body.metadata_filter),
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except EmbeddingError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


class EmbeddingsRequest(BaseModel):
    """Request body for the POST /embeddings endpoint."""

    record_ids: list[str] = Field(min_length=1)


class SimilarRequest(BaseModel):
    """Request body for the POST /similar endpoint."""

    record_id: str
    top_k: int | None = Field(default=None, ge=1)
    metadata_filter: dict[str, Any] | None = Field(default=None)


ALLOWED_IMAGE_SEARCH_TYPES = {"image/jpeg", "image/png"}


@app.post("/search/image", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def search_image(
    file: UploadFile = File(...),
    top_k: int = Form(default=5, ge=1),
    sender: str | None = Form(default=None),
    modality: str | None = Form(default=None),
    date_from: str | None = Form(default=None),
    date_to: str | None = Form(default=None),
    metadata_filter: str | None = Form(default=None),
) -> SearchResponse:
    """Embed an uploaded image and search for similar records without storing the image."""
    if file.content_type not in ALLOWED_IMAGE_SEARCH_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only JPEG and PNG images are supported",
        )
    filter_dict: dict[str, Any] | None = None
    if metadata_filter:
        try:
            filter_dict = json.loads(metadata_filter)
            if not isinstance(filter_dict, dict):
                raise ValueError("metadata_filter must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    parsed_date_from: datetime | None = None
    parsed_date_to: datetime | None = None
    try:
        if date_from:
            parsed_date_from = datetime.fromisoformat(date_from)
        if date_to:
            parsed_date_to = datetime.fromisoformat(date_to)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=f"Invalid date: {exc}")

    image_bytes = await file.read()
    try:
        return _service().search_image(
            image_bytes,
            file.content_type,
            top_k=top_k,
            metadata_filter=_build_filter(sender, modality, parsed_date_from, parsed_date_to, filter_dict),
        )
    except EmbeddingError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


@app.post("/embeddings", response_model=EmbeddingsResponse, status_code=status.HTTP_200_OK)
def get_embeddings(body: EmbeddingsRequest) -> EmbeddingsResponse:
    """Return the stored embedding vector for each requested record ID.

    Missing records are included in the response with `embedding: null`.
    """
    return _service().get_embeddings_for_records(body.record_ids)


@app.post("/similar", response_model=SearchResponse, status_code=status.HTTP_200_OK)
def similar(body: SimilarRequest) -> SearchResponse:
    """Find records with embeddings nearest to the stored embedding for the given record ID."""
    try:
        return _service().search_similar_by_id(
            body.record_id,
            top_k=body.top_k,
            metadata_filter=body.metadata_filter,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))


@app.get("/export/csv", status_code=status.HTTP_200_OK)
def export_csv(include_embeddings: bool = Query(default=False)) -> StreamingResponse:
    """Return all stored records as a CSV file. Pass include_embeddings=true to include embedding vectors."""
    if include_embeddings:
        records, embeddings_list = _service().export_all_with_embeddings()
    else:
        records = _service().export_all()
        embeddings_list = None

    all_meta_keys: list[str] = []
    seen: set[str] = set()
    for r in records:
        for k in r.metadata:
            if k not in seen:
                seen.add(k)
                all_meta_keys.append(k)

    base_fields = ["id", "document_id", "sender", "modality", "tags", "date", "source_path", "object_path", "document"]
    fieldnames = base_fields + all_meta_keys + (["embedding"] if include_embeddings else [])

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for i, r in enumerate(records):
        row: dict[str, Any] = {
            "id": r.id,
            "document_id": r.document_id,
            "sender": r.sender,
            "modality": r.modality,
            "tags": ",".join(r.tags),
            "date": r.date,
            "source_path": r.source_path or "",
            "object_path": r.object_path or "",
            "document": r.document,
            **r.metadata,
        }
        if include_embeddings and embeddings_list is not None:
            row["embedding"] = json.dumps(embeddings_list[i])
        writer.writerow(row)

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=\"export.csv\""},
    )


@app.get("/graph", response_model=GraphResponse, status_code=status.HTTP_200_OK)
def graph(threshold: float = Query(default=0.4, ge=0.0, le=2.0)) -> GraphResponse:
    """Return all image nodes and similarity edges where cosine distance is below threshold."""
    data = _service().get_graph_data(threshold)
    return GraphResponse(
        nodes=[GraphNodeResponse(id=n.id, source_path=n.source_path, object_path=n.object_path) for n in data.nodes],
        edges=[GraphEdgeResponse(source=e.source, target=e.target, distance=e.distance) for e in data.edges],
    )


def main() -> None:
    """Start the uvicorn server on 0.0.0.0:8000."""
    uvicorn.run("sensi_memory.http_server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
