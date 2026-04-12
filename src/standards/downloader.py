from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Iterable
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; TeleRAG/1.0; +https://example.com/teleRAG)",
}


@dataclass(slots=True)
class DownloadJob:
    target_name: str
    source_org: str
    url: str
    destination_dir: str
    filename: str
    kind: str
    release: str | None = None
    series: str | None = None


class DirectoryListingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return

        attr_map = dict(attrs)
        href = attr_map.get("href")
        if href:
            self.links.append(href)


def load_targets(targets_path: str | Path) -> list[dict]:
    path = Path(targets_path)
    return json.loads(path.read_text(encoding="utf-8"))


def fetch_text(url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> str:
    request = Request(url, headers=DEFAULT_HEADERS)
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="ignore")


def fetch_bytes(url: str, timeout: int = DEFAULT_TIMEOUT_SECONDS) -> bytes:
    request = Request(url, headers=DEFAULT_HEADERS)
    with urlopen(request, timeout=timeout) as response:
        return response.read()


def build_download_jobs(
    targets: Iterable[dict],
    fetch_text_func: Callable[[str], str] | None = None,
) -> list[DownloadJob]:
    fetch_listing = fetch_text_func or fetch_text
    jobs: list[DownloadJob] = []

    for target in targets:
        kind = target.get("kind", "file")
        if kind == "directory":
            listing_html = fetch_listing(target["url"])
            jobs.extend(_build_directory_jobs(target, listing_html))
            continue

        if kind == "file":
            jobs.append(_make_job(target, target["url"], target.get("filename")))
            continue

        raise ValueError(f"Unsupported download target kind: {kind}")

    return jobs


def download_jobs(
    jobs: Iterable[DownloadJob],
    output_root: str | Path,
    manifest_path: str | Path,
    *,
    dry_run: bool = False,
    limit: int | None = None,
    fetch_bytes_func: Callable[[str], bytes] | None = None,
) -> dict:
    fetch_file = fetch_bytes_func or fetch_bytes
    output_root_path = Path(output_root)
    manifest_path_obj = Path(manifest_path)
    records: list[dict] = []
    total_downloaded_bytes = 0

    selected_jobs = list(jobs)
    if limit is not None:
        selected_jobs = selected_jobs[:limit]

    for job in selected_jobs:
        destination = output_root_path / job.destination_dir / job.filename
        destination.parent.mkdir(parents=True, exist_ok=True)

        if destination.exists():
            status = "skipped_existing"
            size_bytes = destination.stat().st_size
        elif dry_run:
            status = "planned"
            size_bytes = None
        else:
            content = fetch_file(job.url)
            destination.write_bytes(content)
            size_bytes = len(content)
            total_downloaded_bytes += size_bytes
            status = "downloaded"

        record = {
            **asdict(job),
            "destination": str(destination),
            "size_bytes": size_bytes,
            "status": status,
        }
        records.append(record)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root_path),
        "job_count": len(selected_jobs),
        "downloaded_bytes": total_downloaded_bytes,
        "records": records,
    }
    manifest_path_obj.parent.mkdir(parents=True, exist_ok=True)
    manifest_path_obj.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def _build_directory_jobs(target: dict, listing_html: str) -> list[DownloadJob]:
    suffixes = tuple(s.lower() for s in target.get("file_suffixes", [".zip"]))
    parser = DirectoryListingParser()
    parser.feed(listing_html)

    jobs: list[DownloadJob] = []
    seen_urls: set[str] = set()
    for href in parser.links:
        resolved = urljoin(target["url"].rstrip("/") + "/", href)
        parsed = urlparse(resolved)
        filename = Path(parsed.path).name
        if not filename:
            continue
        if not filename.lower().endswith(suffixes):
            continue
        if resolved in seen_urls:
            continue
        seen_urls.add(resolved)
        jobs.append(_make_job(target, resolved, filename))

    jobs.sort(key=lambda item: item.filename)
    return jobs


def _make_job(target: dict, url: str, filename: str | None = None) -> DownloadJob:
    parsed = urlparse(url)
    resolved_filename = filename or Path(parsed.path).name
    if not resolved_filename:
        raise ValueError(f"Could not infer filename from URL: {url}")

    return DownloadJob(
        target_name=target["name"],
        source_org=target["source_org"],
        url=url,
        destination_dir=target["destination_dir"],
        filename=resolved_filename,
        kind=target.get("kind", "file"),
        release=target.get("release"),
        series=target.get("series"),
    )
