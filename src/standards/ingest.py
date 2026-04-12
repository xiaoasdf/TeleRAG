from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import subprocess
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree


logger = logging.getLogger(__name__)

SUPPORTED_SOURCE_SUFFIXES = {".zip", ".pdf", ".txt", ".md", ".doc", ".docx"}
DIRECT_INDEXABLE_SUFFIXES = {".pdf", ".txt", ".md"}
ZIP_ENTRY_PRIORITY = {
    ".docx": 4,
    ".doc": 4,
    ".pdf": 3,
    ".txt": 2,
    ".md": 1,
}


def collect_standard_sources(standards_root: str | Path, source_orgs: Iterable[str] | None = None) -> list[Path]:
    root = Path(standards_root)
    if not root.exists():
        return []

    requested = _normalize_source_orgs(source_orgs)
    sources = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.suffix.lower() not in SUPPORTED_SOURCE_SUFFIXES:
            continue
        if requested and not _matches_source_org(path, requested):
            continue
        sources.append(path)
    return sources


def stage_standard_sources(
    standards_root: str | Path,
    index_ready_root: str | Path,
    state_path: str | Path,
    *,
    source_orgs: Iterable[str] | None = None,
    limit: int | None = None,
) -> dict:
    standards_root_path = Path(standards_root)
    index_ready_root_path = Path(index_ready_root)
    state_path_obj = Path(state_path)
    previous_state = _load_state(state_path_obj)
    previous_records = {record["source_rel_path"]: record for record in previous_state.get("records", [])}

    source_paths = collect_standard_sources(standards_root_path, source_orgs=source_orgs)
    if limit is not None:
        source_paths = source_paths[:limit]

    active_rel_paths = {str(path.relative_to(standards_root_path)) for path in source_paths}
    current_records = []
    index_ready_files: set[Path] = set()
    stats = {
        "processed_sources": len(source_paths),
        "new_or_updated_sources": 0,
        "skipped_unchanged_sources": 0,
        "failed_sources": 0,
        "removed_sources": 0,
        "extracted_documents": 0,
        "index_ready_files": [],
        "state_path": str(state_path_obj),
    }

    for source_path in source_paths:
        rel_source = str(source_path.relative_to(standards_root_path))
        source_hash = compute_sha256(source_path)
        previous = previous_records.get(rel_source)

        if previous and previous.get("source_sha256") == source_hash:
            restored_outputs = _restore_existing_outputs(previous.get("index_ready_rel_paths", []), index_ready_root_path)
            if restored_outputs:
                index_ready_files.update(restored_outputs)
                current_records.append(previous)
                stats["skipped_unchanged_sources"] += 1
                continue

        if previous:
            _delete_outputs(previous.get("index_ready_rel_paths", []), index_ready_root_path)

        try:
            extracted_paths = extract_source_to_index_ready(
                source_path,
                standards_root=standards_root_path,
                index_ready_root=index_ready_root_path,
            )
            record = {
                "source_rel_path": rel_source,
                "source_sha256": source_hash,
                "source_size_bytes": source_path.stat().st_size,
                "source_org": _infer_source_org(source_path),
                "index_ready_rel_paths": [str(path.relative_to(index_ready_root_path)) for path in extracted_paths],
                "status": "indexed" if extracted_paths else "skipped_no_supported_content",
            }
            current_records.append(record)
            index_ready_files.update(extracted_paths)
            stats["new_or_updated_sources"] += 1
        except Exception as exc:
            logger.warning("standards_extract_failed source=%s error=%s", source_path, exc)
            current_records.append(
                {
                    "source_rel_path": rel_source,
                    "source_sha256": source_hash,
                    "source_size_bytes": source_path.stat().st_size,
                    "source_org": _infer_source_org(source_path),
                    "index_ready_rel_paths": [],
                    "status": "failed",
                    "error": str(exc),
                }
            )
            stats["failed_sources"] += 1

    removed_sources = 0
    for rel_source, record in previous_records.items():
        if rel_source in active_rel_paths:
            continue
        _delete_outputs(record.get("index_ready_rel_paths", []), index_ready_root_path)
        removed_sources += 1

    stats["removed_sources"] = removed_sources
    stats["extracted_documents"] = len(index_ready_files)
    stats["index_ready_files"] = [str(path) for path in sorted(index_ready_files)]

    state = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "standards_root": str(standards_root_path),
        "index_ready_root": str(index_ready_root_path),
        "records": current_records,
        "summary": {
            "processed_sources": stats["processed_sources"],
            "new_or_updated_sources": stats["new_or_updated_sources"],
            "skipped_unchanged_sources": stats["skipped_unchanged_sources"],
            "failed_sources": stats["failed_sources"],
            "removed_sources": stats["removed_sources"],
            "extracted_documents": stats["extracted_documents"],
        },
    }
    state_path_obj.parent.mkdir(parents=True, exist_ok=True)
    state_path_obj.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def extract_source_to_index_ready(
    source_path: str | Path,
    *,
    standards_root: str | Path,
    index_ready_root: str | Path,
) -> list[Path]:
    path = Path(source_path)
    standards_root_path = Path(standards_root)
    index_ready_root_path = Path(index_ready_root)
    suffix = path.suffix.lower()

    if suffix in DIRECT_INDEXABLE_SUFFIXES:
        destination = _index_ready_destination(path, standards_root_path, index_ready_root_path, suffix)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        return [destination]

    if suffix == ".docx":
        destination = _index_ready_destination(path, standards_root_path, index_ready_root_path, ".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(_extract_docx_text(path.read_bytes()), encoding="utf-8")
        return [destination]

    if suffix == ".doc":
        destination = _index_ready_destination(path, standards_root_path, index_ready_root_path, ".txt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(_extract_doc_text(path), encoding="utf-8")
        return [destination]

    if suffix == ".zip":
        return _extract_from_zip(path, standards_root_path, index_ready_root_path)

    return []


def compute_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_from_zip(source_path: Path, standards_root: Path, index_ready_root: Path) -> list[Path]:
    with zipfile.ZipFile(source_path) as archive:
        candidate_infos = [info for info in archive.infolist() if not info.is_dir()]
        selected = _select_primary_entry(candidate_infos, archive_name=source_path.stem)
        if selected is None:
            return []

        content = archive.read(selected)
        selected_suffix = Path(selected.filename).suffix.lower()
        destination_suffix = ".txt" if selected_suffix in {".doc", ".docx"} else selected_suffix
        destination = _index_ready_destination(source_path, standards_root, index_ready_root, destination_suffix)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if selected_suffix == ".docx":
            destination.write_text(_extract_docx_text(content), encoding="utf-8")
        elif selected_suffix == ".doc":
            destination.write_text(_extract_doc_bytes_text(content), encoding="utf-8")
        else:
            destination.write_bytes(content)

        return [destination]


def _select_primary_entry(infos: list[zipfile.ZipInfo], archive_name: str) -> zipfile.ZipInfo | None:
    best_info = None
    best_score = None
    archive_key = archive_name.lower().replace("_", "").replace("-", "")

    for info in infos:
        suffix = Path(info.filename).suffix.lower()
        if suffix not in ZIP_ENTRY_PRIORITY:
            continue

        filename = Path(info.filename).name.lower()
        stem_key = Path(filename).stem.replace("_", "").replace("-", "")
        score = (
            ZIP_ENTRY_PRIORITY[suffix],
            1 if archive_key and archive_key in stem_key else 0,
            0 if any(token in filename for token in ("cover", "history", "change", "cr")) else 1,
            info.file_size,
        )
        if best_score is None or score > best_score:
            best_info = info
            best_score = score

    return best_info


def _extract_docx_text(content: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        document_xml = archive.read("word/document.xml")
    root = ElementTree.fromstring(document_xml)
    text = " ".join(part.strip() for part in root.itertext() if part and part.strip())
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _extract_doc_bytes_text(content: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".doc") as handle:
        temp_path = Path(handle.name)
        handle.write(content)

    try:
        return _extract_doc_text(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


def _extract_doc_text(path: Path) -> str:
    command = _build_doc_extract_command(path)
    if not command:
        raise RuntimeError(
            "Legacy .doc extraction requires Microsoft Word automation on Windows, "
            "but no supported PowerShell executable was found."
        )

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = _decode_subprocess_bytes(exc.stderr).strip()
        raise RuntimeError(
            "Legacy .doc extraction failed. "
            "Make sure Microsoft Word is installed and this script is running in Windows Python. "
            f"{stderr}"
        ) from exc

    text = _decode_subprocess_bytes(completed.stdout).replace("\r\n", "\n").strip()
    if not text:
        raise RuntimeError("Legacy .doc extraction produced empty text.")
    return text


def _build_doc_extract_command(path: Path) -> list[str] | None:
    if os.name != "nt":
        return None

    powershell_exe = shutil.which("powershell") or shutil.which("powershell.exe") or shutil.which("pwsh")
    if not powershell_exe:
        return None

    script = """
$source = $args[0]
$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$doc = $word.Documents.Open($source)
try {
    $doc.Content.Text
} finally {
    $doc.Close()
    $word.Quit()
}
""".strip()
    return [powershell_exe, "-NoProfile", "-Command", script, str(path)]


def _decode_subprocess_bytes(value: bytes | None) -> str:
    if not value:
        return ""
    return value.decode("utf-8", errors="ignore")


def _index_ready_destination(source_path: Path, standards_root: Path, index_ready_root: Path, suffix: str) -> Path:
    relative = source_path.relative_to(standards_root)
    destination = index_ready_root / relative
    return destination.with_suffix(suffix)


def _restore_existing_outputs(rel_paths: list[str], index_ready_root: Path) -> list[Path]:
    restored = []
    for rel_path in rel_paths:
        path = index_ready_root / rel_path
        if path.exists():
            restored.append(path)
    return restored


def _delete_outputs(rel_paths: list[str], index_ready_root: Path) -> None:
    for rel_path in rel_paths:
        path = index_ready_root / rel_path
        if path.exists():
            path.unlink()


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"records": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_source_orgs(source_orgs: Iterable[str] | None) -> set[str]:
    return {item.lower() for item in source_orgs or []}


def _matches_source_org(path: Path, source_orgs: set[str]) -> bool:
    path_parts = {part.lower() for part in path.parts}
    for source_org in source_orgs:
        if source_org in path_parts:
            return True
        if source_org == "itu" and ("itu-r" in path_parts or "itu-t" in path_parts):
            return True
        if source_org == "itu" and "itu" in path_parts:
            return True
    return False


def _infer_source_org(path: Path) -> str:
    lowered_parts = {part.lower() for part in path.parts}
    if "3gpp" in lowered_parts:
        return "3GPP"
    if "etsi" in lowered_parts:
        return "ETSI"
    if "dvb" in lowered_parts:
        return "DVB"
    if "itu" in lowered_parts:
        return "ITU"
    if "itu-r" in lowered_parts:
        return "ITU-R"
    if "itu-t" in lowered_parts:
        return "ITU-T"
    return "unknown"
