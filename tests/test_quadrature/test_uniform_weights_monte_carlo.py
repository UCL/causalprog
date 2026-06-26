import jax.numpy as jnp
import jax.scipy as jsp
import pytest
import pytest_mock

from causalprog.quadrature import MonteCarloGaussianQuadrature
from causalprog.quadrature import (
    UniformWeightMonteCarloGaussianQuadrature as UWMonteCarloGQ,
)


@pytest.mark.parametrize("n_points", [3, 10, 100])
@pytest.mark.parametrize(
    "interval",
    [(-1.0, 1.0), (0.0, 10.0), (-float("inf"), float("inf"))],
    ids=["(-1,1)", "(0,10)", "Infinite interval"],
)
def test_monte_carlo_integration_constant(
    n_points: int,
    interval: tuple[float, float],
    rng_key,
    constant_value: float = 2.0,
) -> None:
    """Under this scheme, integrating a constant function should just return the
    value of the constant as the result, regardless of the interval length & number of
    points used.

    This is because we are effectively integrating `f(x) = constant * P(x; a, b)` over
    $[a, b]$, where `P` is the PDF of a truncated normal distribution on $[a, b]$.
    """
    q = UWMonteCarloGQ(n_points, rng_key=rng_key)
    computed_integral = q.integrate(
        lambda _: constant_value, a=interval[0], b=interval[1]
    )

    assert computed_integral == constant_value


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

    q = UWMonteCarloGQ(n_points, rng_key=rng_key)
    mocker.patch.object(
        q,
        "points_and_weights",
        new=_fixed_pts_and_weights,
    )
    computed_integral = q.integrate(_integrand, a=a, b=b)

    # Uniform weight MC does not actually use the constant weight in the computation,
    # just applies the factor at the end of the pointwise evaluation.
    expected_pts_to_use, _ = q.points_and_weights(a=a, b=b)
    expected_integral = 0.0
    for p in expected_pts_to_use:
        expected_integral += _integrand(p)
    expected_integral /= n_points

    assert jnp.isclose(computed_integral, expected_integral)


@pytest.mark.parametrize(
    "interval",
    [(-1.0, 1.0), (0.0, 10.0), (0.0, float("inf")), (-float("inf"), float("inf"))],
    ids=["(-1,1)", "(0,10)", "Half-line", "Real-line"],
)
def test_uwgsmc_matches_normal_mc(
    interval: tuple[float, float],
    rng_key,
    n_points: int = 100,
) -> None:
    """The uniform-weighted gaussian sampling quadrature scheme is related to the
    standard Monte Carlo with gaussian sampling scheme, as described in the
    `.integrate` method's docstring on the former class.

    This test validates that relationship holds.
    """

    def _integrand(x):
        return jnp.exp(-(x**2))

    def _uwgs_integrand(x):
        return _integrand(x) / jsp.stats.truncnorm.pdf(x, a=interval[0], b=interval[1])

    normal_mc = MonteCarloGaussianQuadrature(n_points, rng_key=rng_key)
    uwgs_mc = UWMonteCarloGQ(n_points, rng_key=rng_key)

    # Fixing the RNG key should also cause the points generated to be identical,
    # but we should confirm this in testing here.
    assert jnp.allclose(
        normal_mc.points_and_weights(*interval)[0],
        uwgs_mc.points_and_weights(*interval)[0],
    )

    mc_integral = normal_mc.integrate(_integrand, a=interval[0], b=interval[1])
    uwgs_integral = uwgs_mc.integrate(_uwgs_integrand, a=interval[0], b=interval[1])

    assert jnp.isclose(mc_integral, uwgs_integral)
