"""Integration tests checking that the ``Distribution`` class is backend-agnostic."""

import jax.numpy as jnp
from numpyro.distributions.continuous import MultivariateNormal

from causalprog.distribution.base import Distribution, SampleTranslator


def test_backend_matches_explicit(rng_key) -> None:
    """
    Test that a ``Distribution`` operates identically to the backend it is supposed
    to support..

    In this integration test, we setup the same multivariate normal distribution
    using both ``NumPyro``. We then use the ``Distribution`` wrapper class to draw
    samples using the frontend ``sample`` method, and check the results are identical
    to what we get from just directly sampling from the ``NumPyro`` object.
    """
    n_dims = 2
    mean = jnp.array([0.0] * n_dims)
    cov = jnp.diag(jnp.array([1.0] * n_dims))
    sample_size = (10, 5)

    npyo_normal = MultivariateNormal(mean, cov)
    npyo_dist = Distribution(npyo_normal, SampleTranslator(rng_key="key"))
    npyo_samples = npyo_dist.sample(rng_key, sample_size)

    npyo_samples_explicitly_drawn = MultivariateNormal(mean, cov).sample(
        rng_key, sample_size
    )
    assert jnp.allclose(npyo_samples_explicitly_drawn, npyo_samples)
