"""Integration tests checking that the ``Distribution`` class is backend-agnostic."""

import distrax
import jax.numpy as jnp
import jax.random as jrn
from numpyro.distributions.continuous import MultivariateNormal

from causalprog.distributions.base import Distribution


def test_different_backends() -> None:
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
    rng = jrn.key(0)
    sample_size = (10, 5)

    distrax_normal = distrax.MultivariateNormalFullCovariance(mean, cov)
    distrax_dist = Distribution(distrax_normal, backend_sample_key_arg="seed")
    distrax_samples = distrax_dist.sample(rng, sample_size)

    npyo_normal = MultivariateNormal(mean, cov)
    npyo_dist = Distribution(npyo_normal)
    npyo_samples = npyo_dist.sample(rng, sample_size)

    assert jnp.allclose(distrax_samples, npyo_samples)
