from __future__ import annotations

import argparse
import json

from bootstrap import ensure_project_root_on_path

ensure_project_root_on_path()

from src.api.service import TeleRAGService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a TeleRAG knowledge base from downloaded communications standards.")
    parser.add_argument("--download-first", action="store_true", help="Download configured standards targets before indexing.")
    parser.add_argument("--source-org", action="append", dest="source_orgs", help="Filter by source org, e.g. 3GPP or ITU-R.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N standards sources.")
    parser.add_argument("--no-persist", action="store_true", help="Build the knowledge base without saving the vector index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = TeleRAGService()
    result = service.index_standards(
        download_first=args.download_first,
        source_orgs=args.source_orgs,
        limit=args.limit,
        persist=not args.no_persist,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
