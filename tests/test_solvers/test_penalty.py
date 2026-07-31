import jax
import jax.numpy as jnp
import pytest

from causalprog.solvers import penalty


def f(x: dict[str, jax.Array]) -> jax.Array:
    return jnp.minimum((x["x1"] ** 2 + x["x2"] ** 2) / 2, 50.0)


def bounds(x: dict[str, jax.Array]) -> jax.Array:
    return jnp.array([1.0 - x["x1"]])


@pytest.mark.parametrize("epsilon", [0.0, 1e-10, 1e-4, 1e-1, 1.0])
def test_minimise_epsilon(epsilon):
    solution = penalty.minimise(
        f, {"x1": 0.0, "x2": 0.0}, bounds, bounds_epsilon=epsilon
    ).fn_args

    assert jnp.isclose(solution["x1"], 1.0 - epsilon)
    assert jnp.isclose(solution["x2"], 0.0)


def test_maximise():
    min_solution = penalty.minimise(f, {"x1": 0.0, "x2": 0.0}, bounds)
    max_solution = penalty.minimise(
        lambda x: 5.0 - f(x),
        {"x1": 0.0, "x2": 0.0},
        bounds,
        max_or_min="max",
    )

    assert jnp.isclose(max_solution.fn_args["x1"], min_solution.fn_args["x1"])
    assert jnp.isclose(max_solution.fn_args["x2"], min_solution.fn_args["x2"])
    assert jnp.isclose(max_solution.obj_val, 5.0 - min_solution.obj_val)


@pytest.mark.parametrize(
    ("bounds_function", "expected_minimum", "expected_maximum"),
    [
        pytest.param(
            lambda _x: jnp.array([]),
            {"args": {"x1": 0.0, "x2": 0.0}, "value": 0.0},
            None,
            id="No bounds.",
        ),
        pytest.param(
            lambda x: jnp.array([1.0 - x["x1"]]),
            {"args": {"x1": 1.0, "x2": 0.0}, "value": 0.5},
            None,
            id="Lower bound on x1.",
        ),
        pytest.param(
            lambda x: jnp.array([1.0 - x["x2"]]),
            {"args": {"x1": 0.0, "x2": 1.0}, "value": 0.5},
            None,
            id="Lower bound on x2.",
        ),
        pytest.param(
            lambda x: jnp.array([1.0 - x["x1"], 0.5 - x["x1"]]),
            {"args": {"x1": 1.0, "x2": 0.0}, "value": 0.5},
            None,
            id="Two bounds on x1, one irrelevant.",
        ),
        pytest.param(
            lambda x: jnp.array([1.0 - x["x1"], x["x1"] - 1.0]),
            {"args": {"x1": 1.0, "x2": 0.0}, "value": 0.5},
            None,
            id="Two-sided bound on x1.",
        ),
        pytest.param(
            lambda x: jnp.array(
                [1.0 - x["x1"], x["x1"] - 1.0, 1.5 - x["x2"], x["x2"] - 1.5]
            ),
            {"args": {"x1": 1.0, "x2": 1.5}, "value": 1.625},
            {"args": {"x1": 1.0, "x2": 1.5}, "value": 1.625},
            id="Two-sided bounds on x1 and x2",
        ),
        pytest.param(
            lambda x: jnp.array([1.0 - x["x1"], x["x1"] - 3.0]),
            {"args": {"x1": 1.0, "x2": 0.0}, "value": 0.5},
            None,
            id="Upper and lower bound on x1.",
        ),
        pytest.param(
            lambda x: jnp.array([1.0 - x["x1"], x["x2"] - 3.0]),
            {"args": {"x1": 1.0, "x2": 0.0}, "value": 0.5},
            None,
            id="Lower bound on x1, upper bound on x2",
        ),
        pytest.param(
            lambda x: jnp.array([1.0 - x["x1"], 1.5 - x["x2"]]),
            {"args": {"x1": 1.0, "x2": 1.5}, "value": 1.625},
            None,
            id="Lower bounds on x1 and x2",
        ),
        pytest.param(
            lambda x: jnp.array(
                [1.0 - x["x1"], x["x1"] - 2.5, 1.5 - x["x2"], x["x2"] - 2.5]
            ),
            {"args": {"x1": 1.0, "x2": 1.5}, "value": 1.625},
            {"args": {"x1": 2.5, "x2": 2.5}, "value": 6.25},
            id="Upper and lower bounds on x1 and x2.",
        ),
        pytest.param(
            lambda x: jnp.array(
                [2.0 * x["x1"] + x["x2"] - 2.0, -x["x1"], 1.0 - x["x1"] - x["x2"]]
            ),
            {"args": {"x1": 0.5, "x2": 0.5}, "value": 0.25},
            {"args": {"x1": 0.0, "x2": 2.0}, "value": 2.0},
            id="A triangular fesible region.",
        ),
    ],
)
def test_bounds(bounds_function, expected_minimum, expected_maximum):
    min_solution = penalty.minimise(f, {"x1": 0.0, "x2": 0.0}, bounds_function)
    assert jnp.isclose(min_solution.fn_args["x1"], expected_minimum["args"]["x1"])
    assert jnp.isclose(min_solution.fn_args["x2"], expected_minimum["args"]["x2"])
    assert jnp.isclose(min_solution.obj_val, expected_minimum["value"])

    if expected_maximum is not None:
        max_solution = penalty.minimise(
            f, {"x1": 0.0, "x2": 0.0}, bounds_function, max_or_min="max"
        )
        assert jnp.isclose(
            max_solution.fn_args["x1"], expected_maximum["args"]["x1"], atol=1e-5
        )
        assert jnp.isclose(
            max_solution.fn_args["x2"], expected_maximum["args"]["x2"], atol=1e-5
        )
        assert jnp.isclose(max_solution.obj_val, expected_maximum["value"], atol=1e-5)
