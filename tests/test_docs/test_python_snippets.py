import os
from pathlib import Path

import pytest


def find_md_files(folder: Path) -> list[str]:
    files = []
    for item in folder.iterdir():
        if item.is_dir():
            files += find_md_files(item)
        elif item.name.endswith(".md") and not item.name.startswith("_"):
            files.append(str(item))
    return files


@pytest.mark.parametrize(
    "md_file",
    find_md_files(
        Path(os.path.realpath(__file__)).parent.parent.parent / "docs",
    ),
)
def test_python_shippets(md_file):
    with Path(md_file).open() as f:
        python = "\n\n".join(
            snippet.split("```")[0] for snippet in f.read().split("```python")[1:]
        )
    if python == "":
        pytest.skip()

    exec(python)  # noqa: S102
