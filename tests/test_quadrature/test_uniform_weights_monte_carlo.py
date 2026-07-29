from collections.abc import Callable

import jax.numpy as jnp
import pytest
import pytest_mock
from jax.scipy.stats.norm import cdf as norm_cdf
from jax.scipy.stats.truncnorm import pdf as truncnorm_pdf

from causalprog.quadrature import MonteCarloGaussianQuadrature
from causalprog.quadrature import (
    UniformWeightMonteCarloGaussianQuadrature as UWMonteCarloGQ,
)


@pytest.mark.parametrize("n_points", [10, 100])
@pytest.mark.parametrize(
    "interval",
    [(-1.0, 1.0), (0.0, 10.0), (-float("inf"), float("inf")), (0.0, float("inf"))],
    ids=["(-1,1)", "(0,10)", "Real line", "Half-line"],
)
def test_monte_carlo_integration_constant(
    n_points: int,
    interval: tuple[float, float],
    rng_key,
    constant_value: float = 2.0,
) -> None:
    """Under this scheme, integrating a constant function should just return the
    value of the constant multiplied by the probability that a normally-distributed
    RV X lies in the interval $[a, b]$.
    """
    q = UWMonteCarloGQ(n_points, rng_key=rng_key)
    computed_integral = q.integrate(
        lambda _: constant_value, a=interval[0], b=interval[1]
    )

    prob_factor = norm_cdf(interval[1]) - norm_cdf(interval[0])
    assert computed_integral == (constant_value * prob_factor)


def test_uwgsmc_integration_formula(
    mocker: pytest_mock.MockerFixture,
    rng_key,
    n_points: int = 100,
    a: float = -1.0,
    b: float = 1.0,
) -> None:
    """
    Assert that, given sample point values and weights for these points, the Monte
    Carlo integral is correctly computed.

    Note that we deliberately mock the generation of the sample points and weights
    in this function, to determine if the correct formula is being applied. The actual
    result that will be computed by the integration process in this test is nonsensical,
    because we are deliberately using a fixed set of custom values for the points &
    weights that have not come from a distribution.
    """

    def _integrand(x):
        return x**2 - 2.0 * x + 1.0

    def _fixed_pts_and_weights(_a=-1.0, _b=1.0, *args, **kwargs):
        return jnp.linspace(_a, _b, num=n_points, endpoint=True), None

    def _fixed_prefactor_weighting(*args):
        return 2.0

    q = UWMonteCarloGQ(n_points, rng_key=rng_key)
    mocker.patch.object(
        q,
        "points_and_weights",
        new=_fixed_pts_and_weights,
    )
    mocker.patch.object(
        q,
        "_scalar_weight",
        new=_fixed_prefactor_weighting,
    )
    computed_integral = q.integrate(_integrand, a=a, b=b)

    # Uniform weight MC does not actually use the constant weight in the computation,
    # just applies the factor at the end of the pointwise evaluation.
    expected_pts, _ = _fixed_pts_and_weights(a=a, b=b)
    expected_wt = _fixed_prefactor_weighting()
    expected_integral = 0.0
    for p in expected_pts:
        expected_integral += _integrand(p)
    expected_integral *= expected_wt

    assert jnp.isclose(computed_integral, expected_integral)


@pytest.mark.parametrize(
    ("interval", "integrand"),
    [
        pytest.param(
            (-1.0, 1.0),
            lambda _: 1.0,
            id="Constant function on (-1.0, 1.0)",
        ),
        pytest.param(
            (-1.0, 5.0),
            lambda x: x**2 - 2 * x + 1,
            id="Polynomial on interval either side of 0",
        ),
        pytest.param(
            (0.0, float("inf")),
            lambda x: jnp.exp(-(x**2)),
            id="Gaussian shape on +ve real line",
        ),
    ],
)
def test_uwgsmc_matches_normal_mc(
    interval: tuple[float, float],
    integrand: Callable[[float], float],
    rng_key,
    n_points: int = 100,
) -> None:
    """The uniform-weighted gaussian sampling quadrature scheme is related to the
    standard Monte Carlo with gaussian sampling scheme, as described in the
    `.integrate` method's docstring on the former class.

    This test validates that relationship holds.
    """

    def _uwgs_integrand(x):
        return integrand(x) / truncnorm_pdf(x, a=interval[0], b=interval[1])

    normal_mc = MonteCarloGaussianQuadrature(n_points, rng_key=rng_key)
    uwgs_mc = UWMonteCarloGQ(n_points, rng_key=rng_key)

    # Fixing the RNG key should also cause the points generated to be identical,
    # but we should confirm this in testing here.
    assert jnp.allclose(
        normal_mc.points_and_weights(*interval)[0],
        uwgs_mc.points_and_weights(*interval)[0],
    )

    mc_integral = normal_mc.integrate(integrand, a=interval[0], b=interval[1])
    uwgs_integral = uwgs_mc.integrate(_uwgs_integrand, a=interval[0], b=interval[1])

    assert jnp.isclose(mc_integral, uwgs_integral)
