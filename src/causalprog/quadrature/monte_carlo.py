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


class UniformWeightMonteCarloGaussianQuadrature(RNGQuadratureMethod):
    r"""
    Monte Carlo quadrature, sampled from a standard Gaussian, but using uniform weights.

    Let $N$ be the number of sample points to be used by the scheme.
    The quadrature method approximates the integral

    $$
    \int_a^b f(x) dx
    \approx \frac{1}{N}\sum_{p_i}\frac{f(p_i)},
    $$

    where $p_i\in[a,b]$ are $N$ samples drawn from a standard Gaussian.

    Note that the above rule for integrating $f$ is identical to conducting standard
    Monte-Carlo integration (with Gaussian importance sampling), but on the integrand
    $F(x) = \frac{f(x)}{\mathcal{P(x)}}$, where $\mathcal{P}$ is the PDF of a
    (truncated to $[a, b]$) normal distribution.
    """

    def integrate(
        self,
        integrand: Integrand,
        a: float = -1.0,
        b: float = 1.0,
        *integrand_args: IntegrandArgs.args,
        **integrand_kwargs: IntegrandArgs.kwargs,
    ) -> float:
        r"""
        Perform Monte-Carlo integration of the `integrand` over $[a,b]$.

        In terms of the concrete classes in the codebase; if `P`
        again represents the PDF of a truncated normal distribution, the following are
        identical:
        - `UniformWeightGaussianSamplesMonteCarloQuadrature.integrate(f, ...)`
        - `MonteCarloGaussianQuadrature.integrate(f/P, ...)`.
        """
        pts, _ = self.points_and_weights(a=a, b=b)
        ptwise_evaluation: jax.Array = jax.vmap(
            lambda x: integrand(x, *integrand_args, **integrand_kwargs)
        )(pts)

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
