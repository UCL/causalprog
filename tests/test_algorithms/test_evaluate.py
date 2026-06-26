"""Tests for evaluate algorithms."""

import numpy as np
import pytest

from causalprog.algorithms import evaluate
from causalprog.graph import (
    ComponentNode,
    ContinuousRandomVariableNode,
    DataNode,
    DistributionNode,
    Graph,
)


@pytest.mark.parametrize(
    ("node", "kwargs_to_evaluate", "expected_result"),
    [
        pytest.param(
            DataNode(label="A"), {"A": 2.0}, 2.0, id="Evaluate DataNode itself"
        ),
        pytest.param(
            ComponentNode("Parent", 1, label="Child"),
            {"Parent": np.arange(4)},
            1.0,
            id="Evaluate ComponentNode, given parent",
        ),
    ],
)
def test_evaluate_node(node, kwargs_to_evaluate, expected_result):
    assert np.allclose(node.evaluate(**kwargs_to_evaluate), expected_result)


@pytest.mark.parametrize(
    ("node", "kwargs_for_evaluate", "expected_error"),
    [
        pytest.param(
            DataNode(label="A"),
            {},
            ValueError("Missing input for node: A"),
            id="DataNode missing input value",
        ),
        pytest.param(
            DistributionNode(distribution=None, label="A"),
            {},
            RuntimeError("Cannot evaluate a DistributionNode"),
            id="Attempt to evaluate DistributionNode",
        ),
    ],
)
def test_evaluate_node_fail_on_missing_data(
    node, kwargs_for_evaluate, expected_error, raises_context
):
    with raises_context(expected_error):
        node.evaluate(**kwargs_for_evaluate)


def test_evaluate_algorithm_three_node():
    graph = Graph(label="g")
    graph.add_node(DataNode(label="a"))
    graph.add_node(DataNode(label="b"))
    graph.add_node(
        ContinuousRandomVariableNode(label="x", compute=lambda a, b: a + 2.0 * b)
    )

    assert np.isclose(evaluate(graph, "x", a=2.0, b=1.5), 5.0)


def test_evaluate_algorithm_four_node():
    graph = Graph(label="g")
    graph.add_node(DataNode(label="a"))
    graph.add_node(DataNode(label="b"))
    graph.add_node(ContinuousRandomVariableNode(label="c", compute=lambda a: a - 0.5))
    graph.add_node(
        ContinuousRandomVariableNode(label="x", compute=lambda b, c: c + 2.0 * b)
    )

    assert np.isclose(evaluate(graph, "c", a=2.0, b=1.5), 1.5)
    assert np.isclose(evaluate(graph, "x", a=2.0, b=1.5), 4.5)
