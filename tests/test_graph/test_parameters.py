"""Tests for graph module."""

import re
from typing import Literal, TypeAlias

import numpy as np
import pytest

from causalprog.graph import DistributionNode, Graph, ParameterNode

NormalGraphNodeNames: TypeAlias = Literal["mean", "cov", "outcome"]
NormalGraphNodes: TypeAlias = dict[
    NormalGraphNodeNames, DistributionNode | ParameterNode
]


@pytest.mark.parametrize(
    ("param_values_before", "params_to_set", "expected"),
    [
        pytest.param(
            {},
            {"outcome": 4.0},
            TypeError("Node outcome is not a parameter node."),
            id="Give non-parameter node",
        ),
        pytest.param(
            {},
            {"mean": 4.0},
            {"mean": 4.0, "cov": None},
            id="Set only one parameter",
        ),
        pytest.param(
            {},
            {},
            {"mean": None, "cov": None},
            id="Doing nothing is fine",
        ),
        pytest.param(
            {"mean": 0.0, "cov": 0.0},
            {"cov": 1.0},
            {"mean": 0.0, "cov": 1.0},
            id="Omission preserves current value",
        ),
    ],
)
def test_set_parameters(
    normal_graph_nodes: NormalGraphNodes,
    normal_graph: Graph,
    param_values_before: dict[NormalGraphNodeNames, float],
    params_to_set: dict[str, float],
    expected: Exception | dict[NormalGraphNodeNames, float],
) -> None:
    """Test that we can identify parameter nodes, and set their values."""
    parameter_nodes = normal_graph.parameter_nodes
    assert normal_graph_nodes["mean"] in parameter_nodes
    assert normal_graph_nodes["cov"] in parameter_nodes
    assert normal_graph_nodes["outcome"] not in parameter_nodes

    # Set any pre-existing values we might want the parameter nodes to have in
    # this test.
    for node_label, value in param_values_before.items():
        n = normal_graph.get_node(node_label)
        assert isinstance(n, ParameterNode), (
            "Cannot set .value on non-parameter node (test input error)."
        )
        n.value = value

    # Check behaviour of set_parameters method.
    if isinstance(expected, Exception):
        with pytest.raises(type(expected), match=re.escape(str(expected))):
            normal_graph.set_parameters(**params_to_set)
    else:
        normal_graph.set_parameters(**params_to_set)

        for node_name, expected_value in expected.items():
            assert normal_graph.get_node(node_name).value == expected_value


def test_parameter_node(rng_key):
    node = ParameterNode(label="mu")

    with pytest.raises(ValueError, match="Cannot sample"):
        node.sample({}, 1, rng_key)

    node.value = 0.3

    assert np.allclose(node.sample({}, 10, rng_key)[0], [0.3] * 10)
