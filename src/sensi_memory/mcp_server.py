"""MCP server for sensi-memory, exposing ingest and search as tools."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from sensi_memory.config import Settings
from sensi_memory.service import MemoryService

mcp = FastMCP("sensi-memory")
service = MemoryService.from_settings(Settings.from_env())


@mcp.tool()
def ingest_text(text: str, tags: str = "") -> str:
    """Store text in long-term memory.

    Args:
        text: The text to store.
        tags: Comma-separated tags to associate with the record (optional).

    Returns:
        A confirmation message with the record ID(s) stored.
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    records = service.ingest_text(text, tags=tag_list)
    ids = ", ".join(r.id for r in records)
    return f"Stored {len(records)} record(s) with IDs: {ids}"


@mcp.tool()
def ingest_image(image_path: str, description: str = "") -> str:
    """Store an image in long-term memory.

    Args:
        image_path: Path to a PNG or JPEG image file.
        description: Optional text description of the image.

    Returns:
        A confirmation message with the record ID.
    """
    record = service.ingest_image(image_path, text=description or None)
    return f"Stored image record: {record.id}\nSource: {record.metadata.get('source_path', image_path)}\nModality: {record.metadata.get('modality')}"


@mcp.tool()
def search_memory(query: str, top_k: int = 5) -> str:
    """Search long-term memory for information relevant to a query.

    Args:
        query: The search query.
        top_k: Maximum number of results to return (default 5).

    Returns:
        Formatted list of matching memory records, or a message if no matches.
    """
    if not query.strip():
        return "Search query cannot be empty."
    
    response = service.search_text(query, top_k=top_k)
    if not response.hits:
        return f"No relevant memories found for query: '{query}'"
    
    lines = [f"Found {len(response.hits)} match(es) for: '{query}'\n"]
    for i, hit in enumerate(response.hits, 1):
        modality = hit.metadata.get("modality", "unknown")
        mime = hit.metadata.get("mime_type", "")
        mime_str = f" ({mime})" if mime else ""
        distance = hit.distance
        lines.append(f"[{i}] [{modality}{mime_str}] {hit.document} (distance: {distance:.3f})")
    
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run()


def main() -> None:
    """Entry point for the MCP server."""
    mcp.run()
