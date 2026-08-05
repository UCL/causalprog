import os
from pathlib import Path

import pytest


def relative_to_docs(file: Path) -> str:
    name = []
    while file:
        if file.name == "docs":
            return "/".join(name[::-1])
        name.append(file.name)
        file = file.parent
    raise RuntimeError("File not in docs folder")


def find_md_files(folder: Path) -> list[str]:
    files = []
    for item in folder.iterdir():
        if item.is_dir():
            files += find_md_files(item)
        elif item.name.endswith(".md") and not item.name.startswith("_"):
            files.append(item)
    return files


@pytest.mark.parametrize(
    "md_file",
    [
        pytest.param(file, id=relative_to_docs(file))
        for file in find_md_files(
            Path(os.path.realpath(__file__)).parent.parent.parent / "docs",
        )
    ],
)
def test_python_shippets(md_file: Path):
    with md_file.open() as f:
        snippets = [
            snippet.split("```")[0] for snippet in f.read().split("```python")[1:]
        ]
    if len(snippets) == 0:
        pytest.skip("No Python snippets in documentation file")

    exec("\n\n".join(snippets))  # noqa: S102
