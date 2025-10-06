"""Tests that all notebooks in the repo run without error."""

import subprocess
import sys
from pathlib import Path

import pytest

root_directory = Path(__file__).resolve().parents[2]

# Hidden directories are excluded to avoid testing checkpoints
notebook_paths = [
    str(path)
    for path in root_directory.rglob("*.ipynb")
    if not any(part.startswith(".") for part in path.parts)
]


def test_notebook_tests_are_setup_correctly():
    """
    In the `test_notebooks_run` test, we make assumptions about the repo in order to
    to find and test all notebooks. This test checks that those assumptions hold true.
    """

    # Check that root directory is named causalprog
    assert (
        root_directory.name == "causalprog"
    ), f"""Expected repo root named 'causalprog', got {root_directory.name!r}.
        Either the name of the package has changed or the test has been moved.
        If the test moved, update the root_directory definition in this test file.
        If the package name has changed, update this assertion accordingly."""

    # Check that the root directory contains a src/causalprog directory
    assert (
        root_directory / "src" / "causalprog"
    ).is_dir(), """Missing src/causalprog directory at repo root.
        Either the package name or package structure have changed, or the tests have
         been moved. If this test file has moved, update the root_directory definition
         in this test file. If the package name has changed, update the assertion
         accordingly."""

    # Checks that this test file is in the expected location
    expect_test_file = (
        root_directory / "tests" / "test_notebooks" / "test_notebooks_run.py"
    )
    assert (
        expect_test_file.is_file()
    ), """The structure of the tests have changed, this means that this test will fail
         to find the notebooks in this repo. If this test file has moved, update the
         root_directory definition and this assertion accordingly."""

    assert notebook_paths, f"No notebooks found under {root_directory}"


@pytest.mark.parametrize(
    "notebook",
    notebook_paths,
    ids=lambda p: str(Path(p).relative_to(root_directory)),
)
def test_notebooks_run(notebook: str):
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
        cwd=root_directory,
        text=True,
        capture_output=True,
        check=False,
        shell=False,
    )

    assert res.returncode == 0, (
        f"nbmake failed for {notebook_paths}\n\n"
        f"=== stdout ===\n{res.stdout}\n"
        f"=== stderr ===\n{res.stderr}"
    )
