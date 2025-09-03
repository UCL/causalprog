from collections.abc import Callable

import numpy as np
import pytest

from causalprog.utils.norms import PyTree, l2_normsq


@pytest.mark.parametrize(
    ("pt", "norm", "expected_value"),
    [
        pytest.param(1.0, l2_normsq, 1.0, id="l2^2, scalar"),
        pytest.param(
            np.array([1.0, 2.0, 3.0]), l2_normsq, 14.0, id="l2^2, numpy array"
        ),
        pytest.param(
            {"a": 1.0, "b": (np.arange(3), [2.0, (-1.0, 0.0)])},
            l2_normsq,
            1.0 + (np.arange(3) ** 2).sum() + 4.0 + 1.0,
            id="l2^2, PyTree",
        ),
    ],
)
def test_norm_value(pt: PyTree, norm: Callable[[PyTree], float], expected_value: float):
    assert np.allclose(norm(pt), expected_value)
