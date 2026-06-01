from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

SENSI_HTTP_URL = os.environ.get("SENSI_HTTP_URL", "http://localhost:8000")
STATIC_DIR = Path(__file__).parent / "static"
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}


class SearchRequest(BaseModel):
    text: str
    top_k: int = Field(default=5, ge=1)
    metadata_filter: dict[str, Any] | None = Field(default=None)


class SimilarRequest(BaseModel):
    record_id: str
    top_k: int = Field(default=5, ge=1)
    metadata_filter: dict[str, Any] | None = Field(default=None)


app = FastAPI(title="Sensi Visualisation", version="0.1.0")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text())


@app.post("/api/search")
async def search(body: SearchRequest) -> dict[str, Any]:
    formatted = f"task: search result | query: {body.text}"
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{SENSI_HTTP_URL}/search",
                json={"text": formatted, "top_k": body.top_k, "metadata_filter": body.metadata_filter},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=exc.response.text,
            )
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Could not reach search service: {exc}",
            )
    return response.json()


@app.post("/api/similar")
async def similar(body: SimilarRequest) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{SENSI_HTTP_URL}/similar",
                json={"record_id": body.record_id, "top_k": body.top_k, "metadata_filter": body.metadata_filter},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=exc.response.text,
            )
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Could not reach search service: {exc}",
            )
    return response.json()


@app.post("/api/search-by-image")
async def search_by_image(
    file: UploadFile = File(...),
    top_k: int = Form(default=5, ge=1),
    metadata_filter: str | None = Form(default=None),
) -> dict[str, Any]:
    image_bytes = await file.read()
    form_data: dict[str, Any] = {"top_k": top_k}
    if metadata_filter is not None:
        form_data["metadata_filter"] = metadata_filter
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{SENSI_HTTP_URL}/search/image",
                files={"file": (file.filename, image_bytes, file.content_type)},
                data=form_data,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=exc.response.text,
            )
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Could not reach search service: {exc}",
            )
    return response.json()


@app.get("/api/graph")
async def graph(threshold: float = Query(default=0.4, ge=0.0, le=2.0)) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.get(
                f"{SENSI_HTTP_URL}/graph",
                params={"threshold": threshold},
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise HTTPException(
                status_code=exc.response.status_code,
                detail=exc.response.text,
            )
        except httpx.RequestError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Could not reach search service: {exc}",
            )
    return response.json()


@app.get("/api/image")
async def serve_image(path: str = Query(...)) -> FileResponse:
    image_path = Path(path)
    if not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image")

    return FileResponse(image_path, media_type=mime_type)
