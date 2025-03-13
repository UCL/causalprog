"""(Multivariate) normal distribution, implemented via ``jax.random`` backend."""

from typing import TypeAlias, TypeVar

import jax.numpy as jnp
import jax.random as jrn
from jax import Array as JaxArray
from numpy.typing import ArrayLike

from .base import Distribution

ArrayCompatible = TypeVar("ArrayCompatible", JaxArray, ArrayLike)
RNGKey: TypeAlias = JaxArray


class _Normal:
    mean: JaxArray
    cov: JaxArray

    def __init__(self, mean: ArrayCompatible, cov: ArrayCompatible) -> None:
        self.mean = jnp.array(mean)
        self.cov = jnp.array(cov)

    def sample(self, rng_key: RNGKey, sample_shape: ArrayLike) -> JaxArray:
        return jrn.multivariate_normal(rng_key, self.mean, self.cov, shape=sample_shape)


class Normal(Distribution):
    r"""
    A (possibly multivaraiate) normal distribution, $\mathcal{N}(\mu, \Sigma)$.

    The normal distribution is parametrised by its (vector of) mean value(s) $\mu$ and
    (matrix of) covariate(s) $\Sigma$. These values must be supplied to an instance at
    upon construction, and can be accessed via the ``mean`` ($\mu$) and ``cov``
    ($\Sigma$) attributes, respectively.

    """

    _dist: _Normal

    @property
    def mean(self) -> JaxArray:
        r"""Mean of the distribution, $\mu$."""
        return self._dist.mean

    @property
    def cov(self) -> JaxArray:
        r"""Covariate matrix of the distribution, $\Sigma$."""
        return self._dist.cov

    def __init__(self, mean: ArrayCompatible, cov: ArrayCompatible) -> None:
        r"""
        Create a new normal distribution.

        Args:
            mean (ArrayCompatible): Vector of mean values, $\mu$.
            cov (ArrayCompatible): Matrix of covariates, $\Sigma$.

        """
        super().__init__(_Normal(mean, cov))
