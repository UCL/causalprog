import jax.numpy as jnp
import jax.random as jrn
import pytest
from jax._src.basearray import Array


@pytest.fixture
def seed() -> int:
    return 0


@pytest.fixture
def rng_key(seed: int):
    return jrn.key(seed)


@pytest.fixture
def n_dim_std_normal(request) -> tuple[Array, Array]:
    """
    Mean and covariance matrix of the n-dimensional standard normal distribution.

    ``request.param`` should be an integer corresponding to the number of dimensions.
    """
    n_dims = request.param
    mean = jnp.array([0.0] * n_dims)
    cov = jnp.diag(jnp.array([1.0] * n_dims))
    return mean, cov
