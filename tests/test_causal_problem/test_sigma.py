import re
from collections.abc import Callable

import jax
import pytest

from causalprog.algorithms import expectation, standard_deviation
from causalprog.causal_problem import CausalProblem
from causalprog.graph import Graph, Node


@pytest.fixture
def n_samples_for_estimands() -> int:
    return 1000


@pytest.fixture
def expectation_fixture(
    n_samples_for_estimands: int, rng_key: jax.Array
) -> Callable[[Graph, Node], float]:
    return lambda g, x: expectation(
        g, x.label, samples=n_samples_for_estimands, rng_key=rng_key
    )


@pytest.fixture
def std_fixture(
    n_samples_for_estimands: int, rng_key: jax.Array
) -> Callable[[Graph, Node], float]:
    return (
        lambda g, x: standard_deviation(
            g, x.label, samples=n_samples_for_estimands, rng_key=rng_key
        )
        ** 2
    )


@pytest.mark.parametrize(
    ("initial_param_values", "args_to_setter", "expected", "atol"),
    [
        pytest.param(
            {"mu_x": 1.0, "nu_y": 1.0},
            {
                "sigma": "expectation_fixture",
                "rvs_to_nodes": {"x": "mu_x"},
                "graph_argument": "g",
            },
            1.0,
            1.0e-12,
            id="Return mu_x",
        ),
        pytest.param(
            {"mu_x": 1.0, "nu_y": 1.0},
            {
                "sigma": "expectation_fixture",
                "rvs_to_nodes": {"x": "nu_y"},
                "graph_argument": "g",
            },
            1.0,
            1.0e-12,
            id="Return nu_y",
        ),
        pytest.param(
            {"mu_x": 0.0, "nu_y": 1.0},
            {
                "sigma": "expectation_fixture",
                "rvs_to_nodes": {},
                "graph_argument": "g",
            },
            0.0,
            # Empirical calculation with 1000 samples with fixture RNG key
            # should give 1.8808 as the empirical expectation.
            2.0e-2,
            id="Return E[x], infer association",
        ),
        pytest.param(
            {"mu_x": 0.0, "nu_y": 1.0},
            {
                "sigma": "std_fixture",
                "rvs_to_nodes": {"x": "y"},
                "graph_argument": "g",
            },
            # x has fixed std 1, and nu_y will be set to 1.
            1.0**2 + 1.0**2,
            # Empirical calculation with 1000 samples with fixture RNG key
            # should give 1.8506 as the empirical std of y.
            2.0e-1,
            id="Return Var[y]",
        ),
    ],
)
def test_sigma_interactions(
    graph: Graph,
    initial_param_values: dict[str, float],
    args_to_setter: dict[str, Callable[..., float] | dict[str, str] | str],
    expected: dict[str, float],
    atol: float,
    request: pytest.FixtureRequest,
) -> None:
    """
    Test the set_causal_estimand and casual_estimand evaluation method.

    Test works by:
    - Set the parameter values using the initial_param_values.
    - Set the causal_estimand using the setter and given arguments.
    - Call .causal_estimand(parameter_vector), which should evaluate the causal estimand
      at the current values of the parameter vector, which will be the initial values
      just set.
    - Check the result (lies within a given tolerance).
    """
    if isinstance(args_to_setter["sigma"], str):
        args_to_setter["sigma"] = request.getfixturevalue(args_to_setter["sigma"])

    cp = CausalProblem(graph)

    # Before setting the causal estimand, it should throw an error if called.
    with pytest.raises(
        NotImplementedError,
        match=re.escape("Causal estimand not set for CausalProblem."),
    ):
        cp.causal_estimand(cp.parameter_vector)

    cp.set_parameter_values(**initial_param_values)
    cp.set_causal_estimand(**args_to_setter)

    result = cp.causal_estimand(cp.parameter_vector)

    assert result == pytest.approx(
        expected,
        abs=atol,
    )
