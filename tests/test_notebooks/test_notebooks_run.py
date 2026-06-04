"""Tests that all notebooks in the repo run without error."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def n_expected_notebooks() -> int:
    """A count of the expected number of notebooks in the root project directory."""
    return 1


def root_directory() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parents[2]


def notebook_paths() -> list[Path]:
    """Get paths to all notebooks in the root project directory."""

    # Hidden directories are excluded to avoid testing checkpoints
    return [
        path
        for path in root_directory().rglob("*.ipynb")
        if not any(part.startswith(".") for part in path.parts)
    ]


def test_notebooks_found(n_expected_notebooks) -> None:
    assert len(set(notebook_paths())) == n_expected_notebooks, (
        "The number of notebooks found does not match the expected count. "
        "If notebooks have been added or removed, update the "
        "n_expected_notebooks fixture."
    )


@pytest.mark.parametrize(
    "notebook",
    notebook_paths(),
    ids=lambda p: str(p.relative_to(root_directory())),
)
def test_notebooks_run(notebook: Path):
    """Tests that a notebook runs without error."""
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "--nbmake",
        str(notebook),
    ]
    res = subprocess.run(  # noqa: S603
        cmd,
        cwd=root_directory(),
        text=True,
        capture_output=True,
        check=False,
        shell=False,
    )

    assert res.returncode == 0, (
        f"nbmake failed for {notebook_paths()}\n\n"
        f"=== stdout ===\n{res.stdout}\n"
        f"=== stderr ===\n{res.stderr}"
    )
