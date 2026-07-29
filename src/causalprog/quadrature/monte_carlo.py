"""Monte Carlo quadrature."""

import jax
import numpy.typing as npt
from jax.scipy.stats.norm import cdf as norm_cdf
from typing_extensions import override

from .base import Integrand, IntegrandArgs, RNGQuadratureMethod


# FIXME this one also may need to change!!!
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
    \int_a^b f(x) p_{N}(x) dx
    \approx \frac{P}{N}\sum_{i} f(x_i),
    $$

    where

    - $p_{N}$ is the PDF of a standard normal distribution,
    - $x_i\in[a,b]$ are $N$ samples from a truncated normal distribution on $[a,b]$,
    - $P = \mathbb{P}[a < X < b \vert X \sim \mathcal{N}(0,1)]$.

    When $a=-\infty$ and $b=\infy$, this effectively computes
    $\mathbb{E}[f(X) \vert X \sim \mathcal{N}(0,1)]$.
    """

    def _scalar_weight(self, a: float, b: float) -> float:
        r"""
        Compute the scalar weight applied to all samples.

        This weight is $\mathcal{P}[a < X < b \vert X\sim \mathcal{N}(0,1)]$
        divided by `self.n_points`.
        """
        probability_in_interval = norm_cdf(b) - norm_cdf(a)
        return probability_in_interval / self.n_points

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

        Specifically, given a function $f$, return an approximation to

        $$ \int_a^b f(x) p_{N}(x) dx. $$
        """
        pts, _ = self.points_and_weights(a=a, b=b)
        ptwise_evaluation: jax.Array = jax.vmap(
            lambda x: integrand(x, *integrand_args, **integrand_kwargs)
        )(pts)

        return ptwise_evaluation.sum() * self._scalar_weight(a, b)

    @override
    def points_and_weights(
        self, a: float = -1.0, b: float = 1.0
    ) -> tuple[npt.NDArray, npt.NDArray]:
        pts = jax.random.truncated_normal(
            self.rng_key, lower=a, upper=b, shape=(self.n_points,)
        )
        wts = jax.numpy.full((self.n_points,), self._scalar_weight(a, b))
        return pts, wts
