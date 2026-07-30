import pytest

import jax
import jax.numpy as jnp

from causalprog.solvers import augmented_lagrangian, penalty


def f(_xzl: dict[str, jax.Array], theta: dict[str, jax.Array]) -> jax.Array:
    return (theta["x1"] ** 2 + theta["x2"] ** 2) / 2


def bounds(_xzl: dict[str, jax.Array], theta: dict[str, jax.Array]) -> jax.Array:
    return jnp.array([
        theta["x1"] - 1.0,
        1.0 - theta["x1"],
    ])

@pytest.mark.parametrize("epsilon", [None, 1e-10, 1e-1, 1.0])
def test_penalty(epsilon):
    solution = penalty.minimise(f, bounds, bounds_epsilon=epsilon, variables=["x1", "x2"])

    if epsilon is None:
        assert jnp.isclose(solution["x1"], 1.0)
        assert jnp.isclose(solution["x2"], 0.0)
    else:
        assert jnp.isclose(solution["x1"], 1.0 - epsilon, epsilon / 10)
        assert jnp.isclose(solution["x2"], 0.0)


def test_augmented_lagrangian():
    solution = augmented_lagrangian.minimise(f, bounds, variables=["x1", "x2"])

    assert jnp.isclose(solution["x1"], 1.0)
    assert jnp.isclose(solution["x2"], 0.0)
