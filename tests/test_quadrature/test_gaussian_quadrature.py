import jax.numpy as jnp

from causalprog.quadrature import GaussianQuadrature


def test_gaussian_quadrature_polynomials(
    coeffs: jnp.Array, interval: tuple[float, float], exact_integral: float
) -> None:
    degree = coeffs.size - 1

    def _integrand(x: float, c: jnp.Array):
        return (c * x ** jnp.arange(c.size)).sum()

    computed_integral = GaussianQuadrature(degree).integrate(
        _integrand, a=interval[0], b=interval[1], c=coeffs
    )

    assert jnp.isclose(exact_integral, computed_integral)
