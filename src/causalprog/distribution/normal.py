"""(Multivariate) normal distribution, implemented via ``jax.random`` backend."""

from typing import TypeAlias, TypeVar

import jax.numpy as jnp
import jax.random as jrn
from jax import Array as JaxArray
from numpy.typing import ArrayLike

from .base import AbstractDistribution, SupportsRNG

ArrayCompatible = TypeVar("ArrayCompatible", JaxArray, ArrayLike)
RNGKey: TypeAlias = JaxArray


class Normal(AbstractDistribution):
    r"""
    A (possibly multivaraiate) normal distribution, $\mathcal{N}(\mu, \Sigma)$.

    The normal distribution is parametrised by its (vector of) mean value(s) $\mu$ and
    (matrix of) covariate(s) $\Sigma$. These values must be supplied to an instance at
    upon construction, and can be accessed via the ``mean`` ($\mu$) and ``cov``
    ($\Sigma$) attributes, respectively.

    """

    def __init__(self) -> None:
        """Create a new normal distribution."""
        super().__init__(label="Normal")

    def sample(self, rng_key: SupportsRNG, sample_shape: ArrayLike = (), **kwargs: ArrayLike) -> ArrayLike:
        """
        Draw samples from the distribution.

        Args:
            rng_key (SupportsRNG): Key or seed object to generate random samples.
            sample_shape (ArrayLike): Shape of samples to draw.

        Returns:
            Randomly-drawn samples from the distribution.

        """
        standard = jrn.normal(rng_key, shape=sample_shape)
        return kwargs["mean"] + standard * jnp.sqrt(kwargs["cov"])
