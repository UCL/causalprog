import jax
import jax.numpy as jnp
import pytest

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
    def _integrand(x: float, c: jax.Array):
        return (c * x ** jnp.arange(c.size)).sum()

    computed_integral = GaussianQuadrature(coeffs.size).integrate(
        _integrand, a=interval[0], b=interval[1], c=coeffs
    )

    assert jnp.isclose(exact_integral, computed_integral)
