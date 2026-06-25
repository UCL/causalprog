"""Monte Carlo quadrature."""

import jax
import numpy as np
import numpy.typing as npt

from .base import Integrand, IntegrandArgs, QuadratureMethod


class MonteCarloGaussianQuadrature(QuadratureMethod):
    r"""
    Monte Carlo quadrature, sampled from a standard Gaussian.

    Let $N$ be the number of sample points to be used by the scheme.
    The quadrature method then draws $N$ sample points $p_i$ from the standard
    Gaussian, and approximates integrals as

    $$
    \int_a^b f(x) dx =
    \int_{-\inf}^{\inf} \mathbb{1}_{[a,b]} f(x) dx \approx
    \frac{1}{N}\sum_{p_i} f(p_i).
    $$
    """

    def __init__(self, npoints: int, *, rng_key: jax.Array) -> None:
        """
        Initialise.

        Args:
            npoints: The number of quadrature points

        """
        super().__init__(npoints)
        self.rng_key = rng_key

    def integrate(
        self,
        integrand: Integrand,
        a: float = -1.0,
        b: float = 1.0,
        *integrand_args: IntegrandArgs.args,
        **integrand_kwargs: IntegrandArgs.kwargs,
    ) -> float:
        """
        Perform Monte-Carlo integration, sampling from a standard Gaussian.

        By default, the domain of integration is the real line. Integrals over
        smaller domains are approximated by multiplying the integrand by the
        indicator function.

        FIXME: This means that we are not guaranteed to use `self.npoints` samples
        on definite integrals! Since we are sampling from a standard Gaussian, we'll
        have to filter some samples we draw. Hence the valid_samples stuff. Would be
        better if we could reliably generate appropriate samples in the interval.
        """
        result = 0.0

        pts = jax.random.truncated_normal(
            self.rng_key, lower=a, upper=b, shape=(self.npoints,)
        )
        wts = jax.scipy.stats.truncnorm.pdf(pts, a, b)

        for p_i, w_i in zip(pts, wts, strict=True):
            result += integrand(p_i, *integrand_args, **integrand_kwargs) / w_i

        return result / self.npoints

    def points_and_weights(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Get the quadrature points and weights."""
        return jax.random.normal(self.rng_key, (self.npoints,)), np.ones(
            self.npoints
        ) / self.npoints
