"""Monte Carlo quadrature."""

import jax
import numpy.typing as npt
from typing_extensions import override

from .base import Integrand, IntegrandArgs, RNGQuadratureMethod


class MonteCarloGaussianQuadrature(RNGQuadratureMethod):
    r"""
    Monte Carlo quadrature, sampled from a standard Gaussian.

    Let $N$ be the number of sample points to be used by the scheme.
    The quadrature method approximates the integral

    $$
    \int_a^b f(x) dx
    \approx \frac{1}{N}\sum_{p_i} \frac{f(p_i)}{\mathcal{P}(p_i)},
    $$

    where $p_i\in[a,b]$ are $N$ samples drawn from a standard Gaussian.
    """

    def integrate(
        self,
        integrand: Integrand,
        a: float = -1.0,
        b: float = 1.0,
        *integrand_args: IntegrandArgs.args,
        **integrand_kwargs: IntegrandArgs.kwargs,
    ) -> float:
        """Perform Monte-Carlo integration of the `integrand` over $[a,b]$."""
        result = 0.0

        for p_i, w_i in self.pts_wts_tuples(a=a, b=b):
            result += integrand(p_i, *integrand_args, **integrand_kwargs) / w_i

        return result / self.n_points

    @override
    def points_and_weights(
        self, a: float = -1.0, b: float = 1.0
    ) -> tuple[npt.NDArray, npt.NDArray]:
        pts = jax.random.truncated_normal(
            self.rng_key, lower=a, upper=b, shape=(self.n_points,)
        )
        wts = jax.scipy.stats.truncnorm.pdf(pts, a, b)
        return pts, wts


class UniformWeightGaussianSamplesMonteCarloQuadrature(RNGQuadratureMethod):
    r"""
    Monte Carlo quadrature, sampled from a standard Gaussian, but using uniform weights.

    Let $N$ be the number of sample points to be used by the scheme.
    The quadrature method approximates the integral

    $$
    \int_a^b f(x) dx
    \approx \frac{1}{N}\sum_{p_i}\frac{f(p_i)},
    $$

    where $p_i\in[a,b]$ are $N$ samples drawn from a standard Gaussian.
    """

    def integrate(
        self,
        integrand: Integrand,
        a: float = -1.0,
        b: float = 1.0,
        *integrand_args: IntegrandArgs.args,
        **integrand_kwargs: IntegrandArgs.kwargs,
    ) -> float:
        """Perform Monte-Carlo integration of the `integrand` over $[a,b]$."""
        pts, _ = self.points_and_weights(a=a, b=b)
        ptwise_evaluation = jax.numpy.apply_along_axis(
            integrand, 0, pts, *integrand_args, **integrand_kwargs
        )

        return ptwise_evaluation.sum() / self.n_points

    @override
    def points_and_weights(
        self, a: float = -1.0, b: float = 1.0
    ) -> tuple[npt.NDArray, npt.NDArray]:
        pts = jax.random.truncated_normal(
            self.rng_key, lower=a, upper=b, shape=(self.n_points,)
        )
        wts = jax.numpy.full((self.n_points,), 1.0 / self.n_points)
        return pts, wts
