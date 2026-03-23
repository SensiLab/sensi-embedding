# Sensi Memory

Sensi Memory is a small Python package for agent-style ingestion and retrieval of multimodal embeddings.
It accepts text or image inputs, embeds them with Gemini, and stores them in a local ChromaDB collection.

## Features

- Ingest text and image inputs
- Generate embeddings with Gemini's multimodal embedding model
- Persist vectors and metadata in a local ChromaDB collection
- Run similarity search with structured results suitable for LLM agents
- Use either Python APIs or a small CLI

## Setup

Use Python 3.11 or newer. The macOS system interpreter at `/usr/bin/python3` is often Python 3.9 and is not supported by this project.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .[dev]
```

Set the environment variables documented in [.env.example](.env.example).

If you see an `onnxruntime` resolution error during `chromadb` installation, verify that the active interpreter is Python 3.11+:

```bash
python -V
which python
```

Chroma requires `onnxruntime>=1.14.1`, and current `onnxruntime` wheels target modern Python versions. Using the system Python 3.9 on macOS will typically fail or resolve incorrectly.

## CLI Examples

```bash
sensi-memory ingest-text --text "Paris is the capital of France"
sensi-memory ingest-image --image ./example.png --text "A skyline photo"
sensi-memory search --text "capital city in Europe"
```

## Python Example

```python
from sensi_memory import MemoryService, Settings

service = MemoryService.from_settings(Settings.from_env())

record = service.ingest_text("Paris is the capital of France")
results = service.search_text("capital city in Europe")
```

## MCP Server (Claude, Cursor, VS Code Copilot)

Sensi Memory exposes `ingest_text`, `ingest_image`, and `search_memory` as MCP tools for use with Claude Desktop, Cursor, and other agent hosts.

### Run the MCP server

```bash
# First, install the optional mcp dependency
pip install mcp

# Start the MCP server
python -m sensi_memory.mcp_server
```

### Claude Desktop Configuration

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "sensi-memory": {
      "command": "python",
      "args": ["-m", "sensi_memory.mcp_server"],
      "cwd": "/Users/stephenkrol/Sensi"
    }
  }
}
```

Replace `/Users/stephenkrol/Sensi` with your actual project path. Restart Claude Desktop after updating.

### Cursor Configuration

Edit `.cursor/config/mcp_servers.json` (or `.cursor/mcp_config.json` depending on version):

```json
{
  "mcpServers": {
    "sensi-memory": {
      "command": "python",
      "args": ["-m", "sensi_memory.mcp_server"],
      "cwd": "/path/to/Sensi"
    }
  }
}
```

### VS Code Copilot Configuration

Edit `.vscode/settings.json`:

```json
{
  "github.copilot.chat.tools": {
    "sensi-memory": {
      "command": "python",
      "args": ["-m", "sensi_memory.mcp_server"],
      "cwd": "${workspaceFolder}"
    }
  }
}
```

## Notes

- The Gemini API key must be provided through `GEMINI_API_KEY`.
- The code assumes local Chroma persistence and does not manage external database provisioning.
- Image records store their source path and metadata in Chroma, not the raw binary payload.
- The MCP server inherits environment variables from your shell. Make sure `GEMINI_API_KEY` is available to the MCP process (e.g., sourced from `.env`).