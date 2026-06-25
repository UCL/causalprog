import jax
import jax.numpy as jnp
import pytest
import pytest_mock

from causalprog.quadrature import GaussianQuadrature


@pytest.mark.parametrize(
    ("coeffs", "interval", "exact_integral"),
    [
        pytest.param(
            jnp.array([1.0]),
            (-1, 1),
            2.0,
            id="[-1, 1] f(x) = 1",
        ),
        pytest.param(
            jnp.array([0.0, 1.0]),
            (-1, 1),
            0.0,
            id="[-1, 1] f(x) = x",
        ),
        pytest.param(
            jnp.array([1.0, -2.0, 1.0]),
            (-1, 1),
            8.0 / 3.0,
            id="[-1, 1] f(x) = x^2 - 2x + 1",
        ),
        pytest.param(
            jnp.array([0.0, 1.0]),
            (0, 5),
            5.0 * 5.0 / 2.0,
            id="[0, 5] f(x) = x",
        ),
        pytest.param(
            jnp.array([0.0, 1.0]),
            (-5, 0),
            -5.0 * 5.0 / 2.0,
            id="[-5, 0] f(x) = x",
        ),
    ],
)
def test_gaussian_quadrature_polynomials(
    coeffs: jax.Array, interval: tuple[float, float], exact_integral: float
) -> None:
    """Gaussian quadrature with n points is exact for polynomials with degree < n."""

    def _integrand(x: float, c: jax.Array):
        return (c * x ** jnp.arange(c.size)).sum()

    computed_integral = GaussianQuadrature(coeffs.size).integrate(
        _integrand, a=interval[0], b=interval[1], c=coeffs
    )

    assert jnp.isclose(exact_integral, computed_integral)


def test_gaussian_integration_formula(
    mocker: pytest_mock.MockerFixture,
    n_points: int = 10,
    a: float = -1.0,
    b: float = 1.0,
) -> None:
    """
    Assert that, given sample point values and weights for these points, the
    approximation to the integral is correctly computed.

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

    q = GaussianQuadrature(n_points)
    mocker.patch.object(
        q,
        "points_and_weights",
        new=_fixed_pts_and_weights,
    )
    expected_pts_to_use, expected_wts_to_use = q.points_and_weights(a=a, b=b)

    computed_integral = q.integrate(_integrand, a=a, b=b)

    expected_integral = 0.0
    for p, w in zip(expected_pts_to_use, expected_wts_to_use, strict=True):
        expected_integral += _integrand(p) * w
    expected_integral *= (b - a) / 2

    assert jnp.isclose(computed_integral, expected_integral)
