import jax
import jax.numpy as jnp

from causalprog.solvers import augmented_lagrangian, penalty


def f(_xzl: dict[str, jax.Array], theta: dict[str, jax.Array]) -> jax.Array:
    return (theta["x1"] ** 2 + theta["x2"] ** 2) / 2


def bounds(_xzl: dict[str, jax.Array], theta: dict[str, jax.Array]) -> jax.Array:
    return theta["x1"] - 1.0


def test_penalty():
    solution = penalty.minimise(f, bounds, variables=["x1", "x2"])

    assert jnp.isclose(solution["x1"], 1.0)
    assert jnp.isclose(solution["x2"], 0.0)


def test_augmented_lagrangian():
    solution = augmented_lagrangian.minimise(f, bounds, variables=["x1", "x2"])

    assert jnp.isclose(solution["x1"], 1.0)
    assert jnp.isclose(solution["x2"], 0.0)
