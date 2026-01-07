import pytest

from causalprog.solvers.iteration_result import IterationResult
from causalprog.solvers.solver_callbacks import _normalise_callbacks


def _cb(ir: IterationResult) -> None:
    """Placeholder callback"""


@pytest.mark.parametrize(
    ("func_input", "expected"),
    [
        (_cb, [_cb]),
        ([_cb, _cb], [_cb, _cb]),
        (None, []),
        ([], []),
        (42, TypeError("'int' object is not iterable")),
    ],
    ids=[
        "single callable",
        "list of two callables",
        "callbacks=None",
        "callbacks=[]",
        "callbacks=42",
    ],
)
def test_normalise_callbacks(func_input, expected, raises_context) -> None:
    """Test that callbacks are normalised correctly."""

    if isinstance(expected, Exception):
        with raises_context(expected):
            _normalise_callbacks(func_input)
    else:
        assert _normalise_callbacks(func_input) == expected
