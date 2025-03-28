"""Integration tests checking that the ``Distribution`` class is backend-agnostic."""

import distrax
import jax.numpy as jnp
from numpyro.distributions.continuous import MultivariateNormal

from causalprog.distribution.base import Distribution, SampleTranslator


def test_different_backends(rng_key) -> None:
    """
    Test that ``Distribution`` can use different (but equivalent) backends.

    In this integration test, we setup the same multivariate normal distribution
    using both ``NumPyro`` and ``distrax`` as backends. We then use the
    ``Distribution`` wrapper class to draw samples from each distribution using the
    frontend ``sample`` method, and check the results are identical.
    """
    n_dims = 2
    mean = jnp.array([0.0] * n_dims)
    cov = jnp.diag(jnp.array([1.0] * n_dims))
    sample_size = (10, 5)

    distrax_normal = distrax.MultivariateNormalFullCovariance(mean, cov)
    distrax_dist = Distribution(distrax_normal, SampleTranslator(rng_key="seed"))
    distrax_samples = distrax_dist.sample(rng_key, sample_size)

    npyo_normal = MultivariateNormal(mean, cov)
    npyo_dist = Distribution(npyo_normal, SampleTranslator(rng_key="key"))
    npyo_samples = npyo_dist.sample(rng_key, sample_size)

    assert jnp.allclose(distrax_samples, npyo_samples)
