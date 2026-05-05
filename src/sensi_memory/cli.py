from __future__ import annotations

import argparse
import json

from sensi_memory.config import Settings
from sensi_memory.service import MemoryService


def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser for all sensi-memory CLI subcommands."""
    parser = argparse.ArgumentParser(prog="sensi-memory")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_text = subparsers.add_parser("ingest-text")
    ingest_text.add_argument("--text", required=True)
    ingest_text.add_argument("--no-chunk", action="store_true")
    ingest_text.add_argument("--tags", nargs="*", default=[])
    ingest_text.add_argument("--metadata", default="{}")
    ingest_text.add_argument("--document-id")

    ingest_image = subparsers.add_parser("ingest-image")
    ingest_image.add_argument("--image", required=True)
    ingest_image.add_argument("--text")
    ingest_image.add_argument("--tags", nargs="*", default=[])
    ingest_image.add_argument("--metadata", default="{}")
    ingest_image.add_argument("--document-id")

    search = subparsers.add_parser("search")
    search.add_argument("--text", required=True)
    search.add_argument("--top-k", type=int)
    search.add_argument("--where", default="{}")

    return parser


def main() -> None:
    """Parse CLI arguments and dispatch to ingest-text, ingest-image, or search, printing JSON output."""
    parser = build_parser()
    args = parser.parse_args()

    service = MemoryService.from_settings(Settings.from_env())

    if args.command == "ingest-text":
        records = service.ingest_text(
            args.text,
            metadata=_parse_json_dict(args.metadata),
            tags=args.tags,
            document_id=args.document_id,
            chunk=not args.no_chunk,
        )
        print(json.dumps([record.model_dump() for record in records], indent=2))
        return

    if args.command == "ingest-image":
        record = service.ingest_image(
            args.image,
            text=args.text,
            metadata=_parse_json_dict(args.metadata),
            tags=args.tags,
            document_id=args.document_id,
        )
        print(json.dumps(record.model_dump(), indent=2))
        return

    if args.command == "search":
        response = service.search_text(
            args.text,
            top_k=args.top_k,
            metadata_filter=_parse_json_dict(args.where),
        )
        print(json.dumps(response.model_dump(), indent=2))
        return

    parser.error(f"Unsupported command: {args.command}")


def _parse_json_dict(value: str) -> dict[str, object]:
    """Parse a JSON string and raise ValueError if it is not a JSON object."""
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")
    return parsed


if __name__ == "__main__":
    main()