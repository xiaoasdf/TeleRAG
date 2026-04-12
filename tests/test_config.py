import os
from pathlib import Path

from src.config import AppConfig, _resolve_local_path


def test_default_standards_source_dir_is_repo_local():
    config = AppConfig()
    assert config.standards_source_dir == "data/raw/standards"


def test_standards_source_env_override(monkeypatch):
    monkeypatch.setenv("TELERAG_STANDARDS_SOURCE_DIR", "/tmp/tele_rag_standards")
    config = AppConfig(standards_source_dir="data/raw/standards")

    assert str(config.standards_source_path).replace("\\", "/") == "/tmp/tele_rag_standards"


def test_drive_path_resolution_uses_native_or_wsl_style():
    resolved = _resolve_local_path("D:/project/TeleRAG/README.md")

    if os.name == "nt":
        assert str(resolved).replace("\\", "/") == "D:/project/TeleRAG/README.md"
    else:
        assert resolved == Path("/mnt/d/project/TeleRAG/README.md")
