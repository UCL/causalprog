"""Tests for evaluate algorithms."""

import jax.numpy as jnp
import pytest
from jax import Array

from causalprog.algorithms import evaluate, evaluate_down_to
from causalprog.graph import Graph
from causalprog.graph.continuous_treatment import continuous_treatment_model


@pytest.fixture
def evaluate_test_graph() -> Graph:
    return continuous_treatment_model(
        z_len=2,
        compute_u_x=lambda data, _params: data["c"] + data["l"][0],
        compute_u_y=lambda data, _params: data["c"] + 1,
        compute_x=lambda data, _params: data["z"][0] + data["u_x"] - data["l"][0],
        compute_y=lambda data, params: data["x"] * data["u_y"] + params["k"],
    )


@pytest.mark.parametrize(
    ("outcome_node_label", "initial_values", "parameters", "expected_result"),
    [
        pytest.param(
            "l",
            {"l": jnp.array([5.5]), "x": 2.0, "c": 4.0},
            {},
            {"l": jnp.array([5.5])},
            id="DataNode evaluation w/ excess information provided",
        ),
        pytest.param(
            "z",
            {"z": jnp.array([2.0, 0.0])},
            {},
            {"z": jnp.array([2.0, 0.0])},
            id="DataNode evaluation",
        ),
        pytest.param(
            "c",
            {"c": 4.0},
            {},
            {"c": 4.0},
            id="DiscreteRVNode evaluation",
        ),
        pytest.param(
            "u_x",
            {"l": jnp.atleast_1d(0.0), "c": 4.0},
            {},
            {"u_x": 4.0},
            id="CtsRVNode evaluation",
        ),
        pytest.param(
            "u_x",
            {"l": jnp.atleast_1d(0.0), "c": 4.0, "u_x": 1.0},
            {},
            {"u_x": 1.0},
            id="CtsRVNode evaluation, 'given that' overrides computed value",
        ),
        pytest.param(
            "u_y",
            {"l": jnp.atleast_1d(0.0), "c": 4.0},
            {},
            {"u_y": 5.0},
            id="CtsRVNode evaluation, with parents that need evaluating",
        ),
        pytest.param(
            "x",
            {"l": jnp.array([5.5]), "z": jnp.array([2.0, 0.0]), "c": 4.0},
            {},
            {"u_x": 9.5, "x": 6.0},
            id="Multiple paths from different root nodes",
        ),
        pytest.param(
            "x",
            {"l": jnp.array([5.5]), "z": jnp.array([2.0, 0.0]), "c": 4.0, "u_x": 10.0},
            {},
            {"x": 6.5},
            id="Multiple paths from different root nodes, with some given values",
        ),
        pytest.param(
            "y",
            {"l": jnp.array([5.5]), "z": jnp.array([2.0, 0.0]), "c": 4.0},
            {"k": 0.0},
            {"u_x": 9.5, "u_y": 5.0, "x": 6.0, "y": 30.0},
            id="Evaluating the 'outcome' node.",
        ),
        pytest.param(
            "y",
            {"l": jnp.array([5.5]), "z": jnp.array([2.0, 0.0]), "c": 4.0},
            {"k": 1.0},
            {"u_x": 9.5, "u_y": 5.0, "x": 6.0, "y": 31.0},
            id="Evaluating the 'outcome' node with a parameters.",
        ),
    ],
)
def test_evaluate(
    evaluate_test_graph: Graph,
    outcome_node_label: str,
    initial_values: dict[str, Array],
    parameters: dict[str, Array],
    expected_result: dict[str, Array],
) -> None:
    computed_result = evaluate_down_to(
        evaluate_test_graph,
        outcome_node_label,
        initial_values,
        parameters,
    )

    # Same number of entries
    assert len(expected_result) == len(computed_result)
    # Keys are correct
    assert set(expected_result.keys()) == set(computed_result.keys())

    # All entries match to acceptable precision for floats
    for node_label, computed_value in computed_result.items():
        assert jnp.allclose(computed_value, expected_result[node_label])

    # Just asking for one value did indeed extract the correct node value
    computed_result_single = evaluate(
        evaluate_test_graph,
        outcome_node_label,
        initial_values,
        parameters,
    )
    assert jnp.allclose(computed_result_single, computed_result[outcome_node_label])


@pytest.mark.parametrize(
    ("outcome_node_label", "initial_values", "parameters", "expected_error"),
    [
        pytest.param(
            "c",
            {"l": jnp.array([5.5]), "z": jnp.array([2.0, 0.0]), "c": 4.5},
            {},
            ValueError("Invalid value for "),
            id="Invalid value for discrete RV node",
        ),
        pytest.param(
            "x",
            {"z": jnp.array([2.0, 0.0])},
            {},
            ValueError("Missing input for node"),
            id="Missing value for a parent",
        ),
        pytest.param(
            "y",
            {"z": jnp.array([2.0, 0.0]), "c": 4.0, "l": jnp.array([1.0])},
            {},
            ValueError("Missing parameter"),
            id="Missing parameter",
        ),
    ],
)
def test_evaluate_error(
    evaluate_test_graph: Graph,
    outcome_node_label: str,
    initial_values: dict[str, Array],
    parameters: dict[str, Array],
    expected_error: BaseException,
    raises_context,
) -> None:
    with raises_context(expected_error):
        evaluate(evaluate_test_graph, outcome_node_label, initial_values, parameters)
