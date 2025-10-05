import pytest

from causalprog.solvers.iteration_result import IterationResult
from causalprog.solvers.solver_callbacks import _normalise_callbacks


def test_normalise_callbacks() -> None:
    """Test that callbacks are normalised correctly."""

    def callback(iter_result: IterationResult) -> None:
        pass

    # Test single callable
    assert _normalise_callbacks(callback) == [callback]

    # Test sequence of callables
    assert _normalise_callbacks([callback, callback]) == [callback, callback]

    # Test None
    assert _normalise_callbacks(None) == []

    # Test empty sequence
    assert _normalise_callbacks([]) == []

    # Test invalid input
    with pytest.raises(
        TypeError, match="Callbacks must be a callable or a sequence of callables"
    ):
        _normalise_callbacks(42)  # type: ignore[arg-type]
