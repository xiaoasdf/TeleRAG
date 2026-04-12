from pathlib import Path

from scripts.bootstrap import ensure_project_root_on_path, project_root


def test_project_root_points_to_repo_root():
    root = project_root()
    assert (root / "src").exists()
    assert (root / "scripts").exists()


def test_ensure_project_root_on_path_returns_repo_root():
    root = ensure_project_root_on_path()
    assert isinstance(root, Path)
    assert root == project_root()
