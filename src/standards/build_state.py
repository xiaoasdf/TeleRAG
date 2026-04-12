from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_build_state(path: str | Path) -> dict[str, Any] | None:
    file_path = Path(path)
    if not file_path.exists():
        return None
    return json.loads(file_path.read_text(encoding="utf-8"))


def save_build_state(path: str | Path, state: dict[str, Any]) -> dict[str, Any]:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = _utc_now()
    file_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def init_build_state(
    *,
    total_files: int,
    batch_size: int,
    vector_store_dir: str,
    resume_enabled: bool,
) -> dict[str, Any]:
    now = _utc_now()
    return {
        "status": "pending",
        "current_stage": "pending",
        "started_at": now,
        "updated_at": now,
        "total_files": total_files,
        "processed_files": 0,
        "processed_batches": 0,
        "batch_size": batch_size,
        "accumulated_chunks": 0,
        "completed_files": [],
        "completed_file_hashes": {},
        "failed_files": [],
        "vector_store_dir": vector_store_dir,
        "index_backend": "uninitialized",
        "resume_enabled": resume_enabled,
        "requires_rebuild": False,
        "rebuild_reason": None,
        "last_completed_file": None,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
