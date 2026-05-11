# Sensi Memory

Sensi Memory is a Python package for agent-style ingestion and retrieval of multimodal embeddings. It accepts text or image inputs, embeds them with Gemini, and stores them in a local ChromaDB collection.

## Features

- Ingest text and images via HTTP API, MCP tools, CLI, or Python
- Generate embeddings with Gemini's multimodal embedding model
- Persist vectors and metadata in a ChromaDB collection
- Run similarity search with structured results and optional metadata filtering
- Deploy with Docker — HTTP REST API and MCP SSE server share one database

## Interfaces

| Interface | Transport | Best for |
|---|---|---|
| HTTP REST API (port 8000) | HTTP/JSON | Scripts, services, non-LLM clients |
| MCP SSE server (port 8001) | HTTP+SSE | LLM clients (Claude Desktop, Cursor, VS Code) |
| CLI (`sensi-memory`) | stdin/stdout | One-off commands and shell scripts |
| Python API | In-process | Direct integration in Python code |

---

## Docker (recommended)

Requires Docker with the Compose plugin. Copy `.env.example` to `.env` and add your `GEMINI_API_KEY`.

```bash
cp .env.example .env
# edit .env and set GEMINI_API_KEY

docker compose up --build
```

This starts two containers that share a single ChromaDB volume:

- **`sensi-http`** on port 8000 — REST API for scripts and non-LLM clients
- **`sensi-mcp`** on port 8001 — MCP SSE server for LLM clients

See [API.md](API.md) for full endpoint reference.

### Connect an LLM client to the MCP server

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "sensi-memory": {
      "url": "http://localhost:8001/sse"
    }
  }
}
```

**Cursor / VS Code** — same URL, configured in each client's MCP settings. Restart the client after saving.

---

## Local setup

Use Python 3.11 or newer. The macOS system interpreter at `/usr/bin/python3` is often Python 3.9 and is not supported.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e .[dev]
```

Copy `.env.example` to `.env` and set `GEMINI_API_KEY`.

> If you see an `onnxruntime` error during `chromadb` installation, verify the active interpreter is Python 3.11+ with `python -V`.

### HTTP server

```bash
sensi-memory-http
```

### MCP server

```bash
sensi-memory-mcp
```

### CLI

```bash
sensi-memory ingest-text --text "Paris is the capital of France"
sensi-memory ingest-image --image ./example.png --text "A skyline photo"
sensi-memory search --text "capital city in Europe"
```

### Python

```python
from sensi_memory import MemoryService, Settings

service = MemoryService.from_settings(Settings.from_env())

records = service.ingest_text("Paris is the capital of France")
results = service.search_text("capital city in Europe")
```

---

## Configuration

All settings are read from environment variables (or a `.env` file).

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | required | Google Gemini API key |
| `SENSI_CHROMA_PATH` | `./local_storage` | Directory for ChromaDB persistence |
| `SENSI_CHROMA_COLLECTION` | `sensi_memories` | Collection name |
| `SENSI_EMBEDDING_MODEL` | `gemini-embedding-2` | Gemini embedding model |
| `SENSI_EMBEDDING_DIMENSIONS` | `768` | Output dimensionality |
| `SENSI_DEFAULT_TOP_K` | `5` | Default number of search results |

When running via Docker, `SENSI_CHROMA_PATH` is automatically set to `/app/local_storage` (the mounted volume). You do not need to set it manually.

---

## Notes

- Image records store their source path and metadata in ChromaDB, not the raw binary. When ingesting images via the MCP server inside Docker, the path must be accessible from within the container. For remote image uploads, use the HTTP API's `POST /ingest/image` endpoint instead.
- The ChromaDB SQLite backend serializes writes, so a single-worker server process is used. Do not increase uvicorn workers without switching to a ChromaDB HTTP client.
