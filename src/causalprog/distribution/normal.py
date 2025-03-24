"""(Multivariate) normal distribution, implemented via ``jax.random`` backend."""

from typing import TypeAlias, TypeVar

import jax.numpy as jnp
import jax.random as jrn
from jax import Array as JaxArray
from numpy.typing import ArrayLike

from .base import Distribution
from .family import DistributionFamily

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
        mean = jnp.atleast_1d(mean)
        cov = jnp.atleast_2d(cov)
        super().__init__(_Normal(mean, cov), label=f"({mean.ndim}-dim) Normal")


class NormalFamily(DistributionFamily):
    r"""
    Constructor class for (possibly multivariate) normal distributions.

    The multivariate normal distribution is parametrised by a (vector of) mean values
    $\mu$, and (matrix of) covariates $\Sigma$. A ``NormalFamily`` represents this
    family of distributions, $\mathcal{N}(\mu, \Sigma)$. The ``.construct`` method can
    be used to construct a ``Normal`` distribution with a fixed mean and covariate
    matrix.
    """

    def __init__(self) -> None:
        """Create a family of normal distributions."""
        super().__init__(Normal, family_name="Normal")

    def construct(self, mean: ArrayCompatible, cov: ArrayCompatible) -> Normal:  # type: ignore # noqa: PGH003
        r"""
        Construct a normal distribution with the given mean and covariates.

        Args:
            mean (ArrayCompatible): Vector of mean values, $\mu$.
            cov (ArrayCompatible): Matrix of covariates, $\Sigma$.

        """
        return super().construct(mean=mean, cov=cov)
