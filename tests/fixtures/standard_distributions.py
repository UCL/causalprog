"""Fixture file for quick-creation of standard distributions."""

import jax.numpy as jnp
import pytest
from jax import Array


@pytest.fixture
def n_dim_std_normal(request: pytest.FixtureRequest) -> dict[str, Array]:
    """
    Mean and covariance matrix of the n-dimensional standard normal distribution.

    ``request.param`` should be an integer corresponding to the number of dimensions.
    """
    n_dims = request.param
    mean = jnp.array([0.0] * n_dims)
    cov = jnp.diag(jnp.array([1.0] * n_dims))
    return {"mean": mean, "cov": cov}
