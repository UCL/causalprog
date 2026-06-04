from collections.abc import Callable

import numpy.typing as npt
import pytest

from causalprog.utils.norms import PyTree


@pytest.fixture
def sum_of_squares_obj() -> Callable[[PyTree], npt.ArrayLike]:
    """f(x) = ||x||_2^2 = sum_i x_i^2"""

    def _inner(x):
        return (x**2).sum()

    return _inner
