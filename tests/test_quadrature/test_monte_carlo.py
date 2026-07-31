import jax
import jax.numpy as jnp
import pytest
import pytest_mock
from jax.scipy.stats.norm import cdf as norm_cdf

from causalprog.quadrature import MonteCarloGaussianQuadrature


def _guassian_shape(x: jax.Array) -> jax.Array:
    """Note that this function returns a value that is equal to

    `sqrt(2 * pi) * jax.scipy.stats.norm.pdf`.
    """
    return jnp.exp(-(x**2) / 2.0)


@pytest.mark.parametrize(
    ("interval"),
    [
        pytest.param((-float("inf"), float("inf")), id="Real line, 100 pts"),
        pytest.param((0.0, float("inf")), id="Half-line, 100 pts"),
        pytest.param((-1.0, 1.0), id="(-1, 1), 100 pts"),
    ],
)
def test_monte_carlo_integration_gaussians(
    interval: tuple[float, float],
    rng_key,
    n_points: int = 100,
) -> None:
    """Test the performance of Monte-Carlo integration on a Gaussian-shaped integrand,
    along the entire real line and positive half-line.
    """
    expected_integral = jnp.sqrt(2.0 * jnp.pi) * (
        norm_cdf(interval[1]) - norm_cdf(interval[0])
    )

    q = MonteCarloGaussianQuadrature(n_points, rng_key=rng_key)
    computed_integral = q.integrate(_guassian_shape, a=interval[0], b=interval[1])

    assert jnp.allclose(expected_integral, computed_integral)


@pytest.mark.parametrize(
    "n_points",
    [100, 10_000, 1_000_000],
    ids=["100 samples", "10k samples", "1M samples"],
)
@pytest.mark.parametrize(
    ("interval", "integrand", "expected_integral"),
    [
        pytest.param(
            (-1.0, 1.0),
            lambda _: 2.0,
            4.0,
            id="Constant function",
        ),
        pytest.param(
            (0, 1),
            lambda x: x,
            0.5,
            id="x on (0,1)",
        ),
        pytest.param(
            (-1.0, 1.0),
            lambda x: 1.0 / (1.0 + x**2),
            jnp.pi / 2.0,
            id="Cauchy PDF on (-1, 1)",
        ),
    ],
)
def test_monte_carlo_integration(
    n_points: int,
    interval: tuple[float, float],
    integrand,
    expected_integral,
    assert_within_mc_error,
    rng_key,
) -> None:
    """Check the approximation to a few integrals."""
    q = MonteCarloGaussianQuadrature(n_points, rng_key=rng_key)
    computed_integral = q.integrate(integrand, a=interval[0], b=interval[1])

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
