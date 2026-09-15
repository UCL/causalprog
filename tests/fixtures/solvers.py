from collections.abc import Callable

import jax
import pytest

from causalprog._types import PyTree


@pytest.fixture
def sum_of_squares_obj() -> Callable[[PyTree], jax.Array]:
    """f(x) = ||x||_2^2 = sum_i x_i^2"""

    def _inner(x):
        return (x**2).sum()

    return _inner
