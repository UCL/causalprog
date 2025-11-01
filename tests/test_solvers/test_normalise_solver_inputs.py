from collections.abc import Callable

import pytest

from causalprog.solvers.iteration_result import IterationResult
from causalprog.solvers.solver_callbacks import _normalise_callbacks


@pytest.mark.parametrize(
    (
        "make_input_callbacks",
        "make_expected_output_callbacks",
    ),
    [
        (
            lambda cb: cb,
            lambda cb: [cb],
        ),
        (
            lambda cb: [cb, cb],
            lambda cb: [cb, cb],
        ),
        (
            lambda cb: None,  # noqa: ARG005
            lambda cb: [],  # noqa: ARG005
        ),
        (
            lambda cb: [],  # noqa: ARG005
            lambda cb: [],  # noqa: ARG005
        ),
    ],
    ids=[
        "single callable",
        "list of two callables",
        "callbacks=None",
        "callbacks=[]",
    ],
)
def test_normalise_callbacks(
    make_input_callbacks: Callable,
    make_expected_output_callbacks: Callable,
) -> None:
    """Test that callbacks are normalised correctly."""

    def callback(iter_result: IterationResult) -> None:
        pass

    input_callbacks = make_input_callbacks(callback)
    expected_output_callbacks = make_expected_output_callbacks(callback)

    assert _normalise_callbacks(input_callbacks) == expected_output_callbacks


def test_normalise_invalid_callbacks(raises_context) -> None:
    """Test that invalid callbacks raise TypeError."""
    with raises_context(TypeError("'int' object is not iterable")):
        _normalise_callbacks(42)  # type: ignore[arg-type]
