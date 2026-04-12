from __future__ import annotations

import argparse
import json
from pathlib import Path

from bootstrap import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path()

from src.standards.downloader import build_download_jobs, download_jobs, load_targets


DEFAULT_TARGETS_PATH = PROJECT_ROOT / "config" / "standards_targets.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "raw"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "data" / "raw" / "standards" / "download_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a curated communications standards corpus for TeleRAG.")
    parser.add_argument("--targets-file", type=Path, default=DEFAULT_TARGETS_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--source-org", action="append", dest="source_orgs", help="Filter by source org, e.g. 3GPP")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N expanded download jobs.")
    parser.add_argument("--dry-run", action="store_true", help="Expand targets and write a manifest without downloading files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = load_targets(args.targets_file)
    if args.source_orgs:
        allowed = {item.lower() for item in args.source_orgs}
        targets = [item for item in targets if item["source_org"].lower() in allowed]

    jobs = build_download_jobs(targets)
    manifest = download_jobs(
        jobs,
        output_root=args.output_root,
        manifest_path=args.manifest_path,
        dry_run=args.dry_run,
        limit=args.limit,
    )

    summary = {
        "job_count": manifest["job_count"],
        "downloaded_bytes": manifest["downloaded_bytes"],
        "manifest_path": str(args.manifest_path),
        "dry_run": args.dry_run,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
