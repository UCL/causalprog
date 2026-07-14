"""Tests for evaluate algorithms."""

import jax.numpy as jnp
import pytest
from jax import Array

from causalprog.algorithms import evaluate, evaluate_down_to
from causalprog.graph import Graph
from causalprog.graph.ricardo import example_model


@pytest.fixture
def evaluate_test_graph() -> Graph:
    return example_model(
        z_len=2,
        compute_u_x=lambda data: data["c"],
        compute_u_y=lambda data: data["c"] + 1,
        compute_phi_x=lambda data: data["l"][0],
        compute_x=lambda data: data["z"][0] + data["u_x"] - data["phi_x"],
        compute_y=lambda data: data["x"] * data["u_y"],
    )


@pytest.mark.parametrize(
    ("outcome_node_label", "initial_values", "expected_result"),
    [
        pytest.param(
            "l",
            {"l": jnp.array([5.5]), "x": 2.0, "c": 4.0},
            {"l": jnp.array([5.5])},
            id="DataNode evaluation w/ excess information provided",
        ),
        pytest.param(
            "z",
            {"z": jnp.array([2.0, 0.0])},
            {"z": jnp.array([2.0, 0.0])},
            id="DataNode evaluation",
        ),
        pytest.param(
            "c",
            {"c": 4.0},
            {"c": 4.0},
            id="DiscreteRVNode evaluation",
        ),
        pytest.param(
            "u_x",
            {"c": 4.0},
            {"u_x": 4.0},
            id="CtsRVNode evaluation",
        ),
        pytest.param(
            "u_x",
            {"c": 4.0, "u_x": 1.0},
            {"u_x": 1.0},
            id="CtsRVNode evaluation, 'given that' overrides computed value",
        ),
        pytest.param(
            "u_y",
            {"c": 4.0},
            {"u_y": 5.0},
            id="CtsRVNode evaluation, with parents that need evaluating",
        ),
        pytest.param(
            "x",
            {"l": jnp.array([5.5]), "z": jnp.array([2.0, 0.0]), "c": 4.0},
            {"u_x": 4.0, "phi_x": 5.5, "x": 0.5},
            id="Multiple paths from different root nodes",
        ),
        pytest.param(
            "x",
            {"l": jnp.array([5.5]), "z": jnp.array([2.0, 0.0]), "c": 4.0, "phi_x": 0.0},
            {"u_x": 4.0, "x": 6.0},
            id="Multiple paths from different root nodes, with some given values",
        ),
        pytest.param(
            "y",
            {"l": jnp.array([5.5]), "z": jnp.array([2.0, 0.0]), "c": 4.0},
            {"u_x": 4.0, "u_y": 5.0, "phi_x": 5.5, "x": 0.5, "y": 2.5},
            id="Evaluating the 'outcome' node.",
        ),
    ],
)
def test_evaluate(
    evaluate_test_graph: Graph,
    outcome_node_label: str,
    initial_values: dict[str, Array],
    expected_result: dict[str, Array],
) -> None:
    computed_result = evaluate_down_to(
        evaluate_test_graph, outcome_node_label, initial_values
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
        evaluate_test_graph, outcome_node_label, initial_values
    )
    assert jnp.allclose(computed_result_single, computed_result[outcome_node_label])


@pytest.mark.parametrize(
    ("outcome_node_label", "initial_values", "expected_error"),
    [
        pytest.param(
            "c",
            {"l": jnp.array([5.5]), "z": jnp.array([2.0, 0.0]), "c": 4.5},
            ValueError("Invalid value for "),
            id="Invalid value for discrete RV node",
        ),
        pytest.param(
            "phi_x",
            {"z": jnp.array([2.0, 0.0]), "x": 4.0},
            ValueError("Missing input for node"),
            id="Missing value for a parent",
        ),
    ],
)
def test_evaluate_error(
    evaluate_test_graph: Graph,
    outcome_node_label: str,
    initial_values: dict[str, Array],
    expected_error: BaseException,
    raises_context,
) -> None:
    with raises_context(expected_error):
        evaluate(evaluate_test_graph, outcome_node_label, initial_values)
