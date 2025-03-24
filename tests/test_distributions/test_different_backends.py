"""Integration tests checking that the ``Distribution`` class is backend-agnostic."""

import distrax
import jax.numpy as jnp
from numpyro.distributions.continuous import MultivariateNormal

from causalprog.backend.translation import Translation
from causalprog.distribution.base import Distribution
from causalprog.distribution.normal import Normal


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
    distrax_dist = Distribution(
        Translation(
            backend_name="sample",
            frontend_name="sample",
            param_map={"seed": "rng_key"},
        ),
        backend=distrax_normal,
        label="Distrax normal",
    )
    distrax_samples = distrax_dist.sample(rng_key, sample_size)

    npyo_normal = MultivariateNormal(mean, cov)
    npyo_dist = Distribution(
        Translation(
            backend_name="sample", frontend_name="sample", param_map={"key": "rng_key"}
        ),
        backend=npyo_normal,
        label="NumPyro normal",
    )
    npyo_samples = npyo_dist.sample(rng_key, sample_size)

    native_normal = Normal(mean, cov)
    native_samples = native_normal.sample(rng_key, sample_size)

    assert jnp.allclose(distrax_samples, npyo_samples)
    assert jnp.allclose(distrax_samples, native_samples)
