import numpy as np
import pytest

import causalprog
from causalprog.quadrature import MonteCarloGaussianQuadrature


@pytest.mark.parametrize("degree", [500, 1000])
def test_monte_carlo_quadrature(degree, rng_key):
    q = causalprog.quadrature.MonteCarloGaussianQuadrature(degree, rng_key=rng_key)

    pts, wts = q.points_and_weights()

    assert np.isclose(sum(wts), 1.0)
    assert np.isclose(sum(wts * pts), 0.0, atol=1e-2)


def test_monte_carlo_integration(
    rng_key,
    npoints=20_000,
) -> None:
    q = MonteCarloGaussianQuadrature(npoints, rng_key=rng_key)

    def _integrand(x):
        # return np.exp(-(x**2))
        return x

    a = -float("inf")
    b = float("inf")

    a = 0.0
    b = 1.0

    computed_integral = q.integrate(_integrand, a=a, b=b)
    sqrt_pi = np.sqrt(np.pi)

    print(computed_integral - sqrt_pi)
