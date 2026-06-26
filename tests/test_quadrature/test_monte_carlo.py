import jax.numpy as jnp
import pytest
import pytest_mock

from causalprog.quadrature import MonteCarloGaussianQuadrature


def _gaussian_shape(x: float) -> float:
    return jnp.exp(-(x**2))


@pytest.mark.parametrize("n_points", [100, 1000, 10_000])
def test_monte_carlo_integration_constant(
    n_points: int,
    assert_within_mc_error,
    rng_key,
    constant_value: float = 2.0,
    interval: tuple[float, float] = (0.0, 1.0),
) -> None:
    """Check that the constant function is (suitably correctly) integrated."""
    expected_integral = (interval[1] - interval[0]) * constant_value

    q = MonteCarloGaussianQuadrature(n_points, rng_key=rng_key)
    computed_integral = q.integrate(
        lambda _: constant_value, a=interval[0], b=interval[1]
    )

    assert_within_mc_error(computed_integral, expected_integral, n_points)


@pytest.mark.parametrize("n_points", [100, 1000, 10_000, 100_000])
@pytest.mark.parametrize("half_interval", [True, False])
def test_monte_carlo_integration_gaussians(
    n_points: int,
    rng_key,
    assert_within_mc_error,
    *,
    half_interval: bool,
    expected_integral: float = jnp.sqrt(jnp.pi),
) -> None:
    """Test the performance of Monte-Carlo integration on a Gaussian-shaped integrand,
    along the entire real line and positive half-line.
    """
    interval = [-float("inf"), float("inf")]
    expected_integral = jnp.sqrt(jnp.pi)
    if half_interval:
        interval[0] = 0.0
        expected_integral /= 2.0

    q = MonteCarloGaussianQuadrature(n_points, rng_key=rng_key)
    computed_integral = q.integrate(_gaussian_shape, a=interval[0], b=interval[1])

    assert_within_mc_error(computed_integral, expected_integral, n_points)


def test_monte_carlo_integration_formula(
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
        return jnp.linspace(_a, _b, num=n_points, endpoint=True), 0.5 * jnp.ones(
            (n_points,)
        )

    q = MonteCarloGaussianQuadrature(n_points, rng_key=rng_key)
    mocker.patch.object(
        q,
        "points_and_weights",
        new=_fixed_pts_and_weights,
    )
    expected_pts_to_use, expected_wts_to_use = q.points_and_weights(a=a, b=b)

    computed_integral = q.integrate(_integrand, a=a, b=b)

    expected_integral = 0.0
    for p, w in zip(expected_pts_to_use, expected_wts_to_use, strict=True):
        expected_integral += _integrand(p) / w
    expected_integral /= n_points

    assert jnp.isclose(computed_integral, expected_integral)
