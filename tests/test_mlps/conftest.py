import jax
import jax.numpy as jnp
import pytest


@pytest.fixture
def x_3() -> jax.Array:
    return jnp.array([1.0, -2.0, 0.5])
