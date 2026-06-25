import pytest
from causalprog._abc.labelled import Labelled


@pytest.mark.parametrize("label", [
    "1",
    " ",
    "a b",
    "0a",
    "a.b",
    "a-b",
    "a+b",
    "a*b",
    "a/b",
])
def test_invalid_label(label, raises_context):
    with raises_context(ValueError("Label is not valid Python variable name")):
        Labelled(label=label)


@pytest.mark.parametrize("label", [
    "a",
    "A",
    "a0",
    "a_b",
    "_a",
])
def test_valid_label(label, raises_context):
    Labelled(label=label)
