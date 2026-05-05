"""FastAPI HTTP server for sensi-memory, exposing ingest and search over REST."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from sensi_memory.config import Settings
from sensi_memory.gemini_client import EmbeddingError
from sensi_memory.models import SearchResponse, StoredRecord
from sensi_memory.service import MemoryService


class TextIngestRequest(BaseModel):
    """Request body for the POST /ingest/text endpoint."""

    text: str
    tags: list[str] | None = Field(default=None)
    metadata: dict[str, Any] | None = Field(default=None)
    document_id: str | None = Field(default=None)
    chunk: bool = Field(default=True)


class SearchRequest(BaseModel):
    """Request body for the POST /search endpoint."""

    text: str
    top_k: int | None = Field(default=None, ge=1)
    metadata_filter: dict[str, Any] | None = Field(default=None)


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


@app.post("/ingest/text", response_model=list[StoredRecord], status_code=status.HTTP_200_OK)
def ingest_text(body: TextIngestRequest) -> list[StoredRecord]:
    """Embed and store text in the vector database, optionally chunking large inputs."""
    try:
        return _service().ingest_text(
            body.text,
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
    text: str | None = Form(default=None),
    tags: str | None = Form(default=None),
    metadata: str | None = Form(default=None),
    document_id: str | None = Form(default=None),
) -> StoredRecord:
    """Embed and store an uploaded image, writing it to a tempfile for path-based normalization.

    tags: comma-separated string (e.g. "photo,nature").
    metadata: JSON object string (e.g. '{"source": "camera"}').
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
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
            tags=tag_list,
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
            metadata_filter=body.metadata_filter,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    except EmbeddingError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc))


def main() -> None:
    """Start the uvicorn server on 0.0.0.0:8000."""
    uvicorn.run("sensi_memory.http_server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
