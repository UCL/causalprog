"""Tests for evaluate algorithms."""

import jax.numpy as jnp
import pytest
from jax import Array

from causalprog.algorithms import evaluate
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
            jnp.array([5.5]),
            id="DataNode evaluation w/ excess information provided",
        ),
        pytest.param(
            "Z",
            {"Z": jnp.array([2.0, 0.0])},
            jnp.array([2.0, 0.0]),
            id="DataNode evaluation",
        ),
        pytest.param(
            "C",
            {"C": 4.0},
            4.0,
            id="DiscreteRVNode evaluation",
        ),
        pytest.param(
            "UX",
            {"C": 4.0},
            4.0,
            id="CtsRVNode evaluation",
        ),
        pytest.param(
            "UX",
            {"C": 4.0, "UX": 1.0},
            1.0,
            id="CtsRVNode evaluation, 'given that' overrides computed value",
        ),
        pytest.param(
            "UY",
            {"C": 4.0},
            5.0,
            id="CtsRVNode evaluation, with parents that need evaluating",
        ),
        pytest.param(
            "X",
            {"L": jnp.array([5.5]), "Z": jnp.array([2.0, 0.0]), "C": 4.0},
            0.5,
            id="Multiple paths from different root nodes",
        ),
        pytest.param(
            "X",
            {"L": jnp.array([5.5]), "Z": jnp.array([2.0, 0.0]), "C": 4.0, "PhiX": 0.0},
            6.0,
            id="Multiple paths from different root nodes, with some given values",
        ),
        pytest.param(
            "Y",
            {"L": jnp.array([5.5]), "Z": jnp.array([2.0, 0.0]), "C": 4.0},
            2.5,
            id="Evaluating the 'outcome' node.",
        ),
    ],
)
def test_evaluate(
    evaluate_test_graph: Graph,
    outcome_node_label: str,
    initial_values: dict[str, Array],
    expected_result: Array,
) -> None:
    computed_result = evaluate(
        evaluate_test_graph, outcome_node_label, **initial_values
    )
    assert jnp.allclose(expected_result, computed_result)
