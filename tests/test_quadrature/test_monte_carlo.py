import numpy as np
import pytest

import causalprog


@pytest.mark.parametrize("degree", [500, 1000])
def test_monte_carlo_quadrature(degree, rng_key):
    q = causalprog.quadrature.MonteCarloGaussianQuadrature(degree, rng_key=rng_key)

    pts, wts = q.points_and_weights()

    assert np.isclose(sum(wts), 1.0)
    assert np.isclose(sum(wts * pts), 0.0, atol=1e-2)
