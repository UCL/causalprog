"""Tests for evaluate algorithms."""

import numpy as np
import pytest

from causalprog.graph import ComponentNode, DataNode, DistributionNode


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
