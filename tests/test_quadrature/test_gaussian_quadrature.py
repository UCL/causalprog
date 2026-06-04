import numpy as np
import pytest

import causalprog


@pytest.mark.parametrize("degree", range(1, 7))
def test_gaussian_quadrature(degree):
    q = causalprog.quadrature.GaussianQuadrature(degree)

    pts, wts = q.points_and_weights()

    for i in range(degree + 1):
        integral = sum(2 * wts * pts**i)
        exact_integral = 2 / (i + 1) if i % 2 == 0 else 0
        assert np.isclose(integral, exact_integral)
