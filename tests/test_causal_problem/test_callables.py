import re
from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
import pytest

from causalprog.algorithms import expectation, standard_deviation
from causalprog.causal_problem import CausalProblem
from causalprog.graph import Graph, Node

n_samples_for_estimands = 1000


@pytest.fixture
def expectation_fixture(
    rng_key: jax.Array,
) -> Callable[[Graph, Node, dict[str, float] | None], float]:
    def _inner(g: Graph, x: Node, parameter_values=None) -> float:
        return expectation(
            g,
            x.label,
            samples=n_samples_for_estimands,
            rng_key=rng_key,
            parameter_values=parameter_values,
        )

    return _inner


@pytest.fixture
def std_fixture(
    rng_key: jax.Array,
) -> Callable[[Graph, Node, dict[str, float] | None], float]:
    def _inner(g: Graph, x: Node, parameter_values=None) -> float:
        return (
            standard_deviation(
                g,
                x.label,
                samples=n_samples_for_estimands,
                rng_key=rng_key,
                parameter_values=parameter_values,
            )
            ** 2
        )

    return _inner


@pytest.fixture
def vector_fixture(
    rng_key: jax.Array,
) -> Callable[[Graph, Node, Node, dict[str, float] | None], jax.Array]:
    """vector_fixture(g, x1, x2) = [mean of x1, std of x2]."""

    def _inner(g: Graph, x1: Node, x2: Node, parameter_values=None) -> jax.Array:
        return jnp.array(
            [
                expectation(
                    g,
                    x1.label,
                    samples=n_samples_for_estimands,
                    rng_key=rng_key,
                    parameter_values=parameter_values,
                ),
                standard_deviation(
                    g,
                    x2.label,
                    samples=n_samples_for_estimands,
                    rng_key=rng_key,
                    parameter_values=parameter_values,
                )
                ** 2,
            ]
        )

    return _inner


@pytest.fixture(params=["causal_estimand", "constraints"])
def which(request: pytest.FixtureRequest) -> Literal["causal_estimand", "constraints"]:
    """For tests applicable to both the causal_estimand and constraints methods."""
    return request.param


@pytest.mark.parametrize(
    ("initial_param_values", "args_to_setter", "expected", "atol"),
    [
        pytest.param(
            {"mean": 1.0, "cov2": 1.0},
            {
                "fn": "expectation_fixture",
                "rvs_to_nodes": {"x": "mean"},
                "graph_argument": "g",
            },
            1.0,
            1.0e-12,
            id="mean",
        ),
        pytest.param(
            {"mean": 1.0, "cov2": 1.0},
            {
                "fn": "expectation_fixture",
                "rvs_to_nodes": {"x": "cov2"},
                "graph_argument": "g",
            },
            1.0,
            1.0e-12,
            id="cov2",
        ),
        pytest.param(
            {"mean": 0.0, "cov2": 1.0},
            {
                "fn": "expectation_fixture",
                "rvs_to_nodes": {"x": "X"},
                "graph_argument": "g",
            },
            0.0,
            3.0e-2,
            id="E[x], infer association",
        ),
        pytest.param(
            {"mean": 0.0, "cov2": 1.0},
            {
                "fn": "std_fixture",
                "rvs_to_nodes": {"x": "X"},
                "graph_argument": "g",
            },
            # UX has fixed std 1, and cov2 will be set to 1.
            1.0**2 + 1.0**2,
            3.0e-1,
            id="Var[y]",
        ),
        pytest.param(
            {"mean": 0.0, "cov2": 1.0},
            {
                "fn": "vector_fixture",
                "rvs_to_nodes": {"x1": "UX", "x2": "X"},
                "graph_argument": "g",
            },
            # As per the previous test cases
            jnp.array([0.0, 1.0**2 + 1.0**2]),
            jnp.array([3.0e-2, 2.0e-1]),
            id="E[x], Var[y]",
        ),
    ],
)
def test_callables(
    two_normal_graph: Callable[..., Graph],
    which: Literal["causal_estimand", "constraints"],
    initial_param_values: dict[str, float],
    args_to_setter: dict[str, Callable[..., float] | dict[str, str] | str],
    expected: float | jax.Array,
    atol: float,
    request: pytest.FixtureRequest,
) -> None:
    """
    Test the set_{causal_estimand, constraints} and .{casual_estimand, constraints}
    evaluation method.

    Test works by:
    - Set the parameter values using the initial_param_values.
    - Call the setter method using the given arguments.
    - Evaluate the method that should have been set at the current parameter_vector,
        which should evaluate the corresponding function at the current values of the
        parameter vector, which will be the initial values just set.
    - Check the result (lies within a given tolerance).

    In theory, there is no difference between the causal estimand and constraints when
    it comes to this test - the constraints may be vector-valued but there is nothing
    preventing the ``causal_estimand`` (programmatically) from being vector-valued
    either.
    """
    # Parametrised fixtures edit-in-place objects
    args_to_setter = dict(args_to_setter)
    if isinstance(args_to_setter["fn"], str):
        args_to_setter["fn"] = request.getfixturevalue(args_to_setter["fn"])

    if which == "constraints":
        args_to_setter["constraints"] = args_to_setter.pop("fn")
    else:
        args_to_setter["sigma"] = args_to_setter.pop("fn")

    expected = jnp.array(expected, ndmin=1)

    # Test properly begins.
    graph = two_normal_graph(cov=1.0)
    cp = CausalProblem(graph)

    method = getattr(cp, which)
    setter_method = getattr(cp, f"set_{which}")

    # Before setting the causal estimand, it should throw an error if called.
    with pytest.raises(
        NotImplementedError,
        match=re.escape(
            f"{which.replace('_', ' ').capitalize()} not set for CausalProblem."
        ),
    ):
        method(cp.parameter_vector)

    cp.set_parameter_values(**initial_param_values)
    setter_method(**args_to_setter)
    result = jnp.array(method(cp.parameter_vector), ndmin=1)

    assert jnp.allclose(result, expected, atol=atol)
