"""Tests for evaluate algorithms."""

import jax.numpy as jnp
import pytest
from jax import Array

from causalprog.algorithms import evaluate, evaluate_down_to
from causalprog.graph import Graph
from causalprog.graph.special import example_model


@pytest.fixture
def evaluate_test_graph() -> Graph:
    return example_model(
        z_len=2,
        compute_u_x=lambda C: C,
        compute_u_y=lambda C: C + 1,
        compute_phi_x=lambda L: L[0],
        compute_x=lambda Z, PhiX, UX: Z[0] + UX - PhiX,
        compute_y=lambda X, UY: X * UY,
    )


@pytest.mark.parametrize(
    ("outcome_node_label", "initial_values", "expected_result"),
    [
        pytest.param(
            "L",
            {"L": jnp.array([5.5]), "Z": jnp.array([2.0, 0.0]), "C": 4.0},
            {},
            id="DataNode evaluation w/ excess information provided",
        ),
        pytest.param(
            "Z",
            {"Z": jnp.array([2.0, 0.0])},
            {},
            id="DataNode evaluation",
        ),
        pytest.param(
            "C",
            {"C": 4.0},
            {},
            id="DiscreteRVNode evaluation",
        ),
        pytest.param(
            "UX",
            {"C": 4.0},
            {"UX": 4.0},
            id="CtsRVNode evaluation",
        ),
        pytest.param(
            "UX",
            {"C": 4.0, "UX": 1.0},
            {},
            id="CtsRVNode evaluation, 'given that' overrides computed value",
        ),
        pytest.param(
            "UY",
            {"C": 4.0},
            {"UY": 5.0},
            id="CtsRVNode evaluation, with parents that need evaluating",
        ),
        pytest.param(
            "X",
            {"L": jnp.array([5.5]), "Z": jnp.array([2.0, 0.0]), "C": 4.0},
            {"UX": 4.0, "PhiX": 5.5, "X": 0.5},
            id="Multiple paths from different root nodes",
        ),
        pytest.param(
            "X",
            {"L": jnp.array([5.5]), "Z": jnp.array([2.0, 0.0]), "C": 4.0, "PhiX": 0.0},
            {"UX": 4.0, "X": 6.0},
            id="Multiple paths from different root nodes, with some given values",
        ),
        pytest.param(
            "Y",
            {"L": jnp.array([5.5]), "Z": jnp.array([2.0, 0.0]), "C": 4.0},
            {"UX": 4.0, "UY": 5.0, "PhiX": 5.5, "X": 0.5, "Y": 2.5},
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
        evaluate_test_graph, outcome_node_label, **initial_values
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
        evaluate_test_graph, outcome_node_label, **initial_values
    )
    if outcome_node_label in initial_values:
        assert jnp.allclose(computed_result_single, initial_values[outcome_node_label])
    else:
        assert jnp.allclose(computed_result_single, computed_result[outcome_node_label])


@pytest.mark.parametrize(
    ("outcome_node_label", "initial_values", "expected_error"),
    [
        pytest.param(
            "C",
            {"L": jnp.array([5.5]), "Z": jnp.array([2.0, 0.0]), "C": 4.5},
            ValueError("Invalid value for "),
            id="Invalid value for C",
        )
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
        evaluate(evaluate_test_graph, outcome_node_label, **initial_values)
