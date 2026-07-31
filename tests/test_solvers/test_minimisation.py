import jax
import jax.numpy as jnp
import pytest

from causalprog.solvers import augmented_lagrangian, penalty


def f(x: dict[str, jax.Array]) -> jax.Array:
    return (x["x1"] ** 2 + x["x2"] ** 2) / 2


def bounds(x: dict[str, jax.Array]) -> jax.Array:
    return jnp.array([1.0 - x["x1"]])


@pytest.mark.parametrize("epsilon", [None, 1e-10, 1e-4, 1e-1, 1.0])
def test_penalty(epsilon):
    solution = penalty.minimise(
        f, {"x1": 0.0, "x2": 0.0}, bounds, bounds_epsilon=epsilon
    ).fn_args

    if epsilon is None:
        assert jnp.isclose(solution["x1"], 1.0)
        assert jnp.isclose(solution["x2"], 0.0)
    else:
        assert jnp.isclose(solution["x1"], 1.0 - epsilon)
        assert jnp.isclose(solution["x2"], 0.0)


def test_penalty_maximise():
    min_solution = penalty.minimise(f, {"x1": 0.0, "x2": 0.0}, bounds)
    max_solution = penalty.maximise(
        lambda x: 5.0 - f(x), {"x1": 0.0, "x2": 0.0}, bounds
    )

    assert jnp.isclose(max_solution.fn_args["x1"], min_solution.fn_args["x1"])
    assert jnp.isclose(max_solution.fn_args["x2"], min_solution.fn_args["x2"])
    assert jnp.isclose(max_solution.obj_val, 5.0 - min_solution.obj_val)


@pytest.mark.parametrize("epsilon", [None, 1e-10, 1e-4, 1e-1, 1.0])
def test_augmented_lagrangian(epsilon):
    solution = augmented_lagrangian.minimise(
        f, {"x1": 0.0, "x2": 0.0}, bounds, bounds_epsilon=epsilon
    ).fn_args

    if epsilon is None:
        assert jnp.isclose(solution["x1"], 1.0)
        assert jnp.isclose(solution["x2"], 0.0)
    else:
        assert jnp.isclose(solution["x1"], 1.0 - epsilon)
        assert jnp.isclose(solution["x2"], 0.0)
