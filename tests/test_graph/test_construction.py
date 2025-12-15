"""Tests for graph module."""

import pytest
from numpyro.distributions import Normal

import causalprog
from causalprog.graph import DistributionNode, Graph


def test_label():
    node = DistributionNode(Normal, label="X")
    node2 = DistributionNode(Normal, label="Y")
    node_copy = node

    assert node.label == node_copy.label == "X"
    assert node.label != node2.label
    assert node2.label == "Y"

    assert isinstance(node, causalprog.graph.node.Node)
    assert isinstance(node2, causalprog.graph.node.Node)


def test_duplicate_label(raises_context):
    graph = Graph(label="G0")
    graph.add_node(DistributionNode(Normal, label="X"))
    with raises_context(ValueError("Duplicate node label: X")):
        graph.add_node(DistributionNode(Normal, label="X"))


@pytest.mark.parametrize(
    "use_labels",
    [pytest.param(True, id="Via labels"), pytest.param(False, id="Via variables")],
)
def test_build_graph(*, use_labels: bool) -> None:
    root_label = "root"
    outcome_label = "outcome_label"

    root_node = DistributionNode(Normal, label=root_label)
    outcome_node = DistributionNode(Normal, label=outcome_label)

    graph = Graph(label="G0")
    graph.add_node(root_node)
    graph.add_node(outcome_node)

    if use_labels:
        graph.add_edge(root_label, outcome_label)
    else:
        graph.add_edge(root_node, outcome_node)

    assert graph.roots_down_to_outcome(outcome_label) == (root_node, outcome_node)
