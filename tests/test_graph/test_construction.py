"""Tests for graph module."""

import re
from typing import Literal, TypeAlias

import pytest

import causalprog
from causalprog.distribution.normal import NormalFamily
from causalprog.graph import DistributionNode, Graph, ParameterNode

NormalGraphNodeNames: TypeAlias = Literal["mean", "cov", "outcome"]
NormalGraphNodes: TypeAlias = dict[
    NormalGraphNodeNames, DistributionNode | ParameterNode
]


def test_label():
    d = NormalFamily()
    node = DistributionNode(d, label="X")
    node2 = DistributionNode(d, label="Y")
    node_copy = node

    assert node.label == node_copy.label == "X"
    assert node.label != node2.label
    assert node2.label == "Y"

    assert isinstance(node, causalprog.graph.node.Node)
    assert isinstance(node2, causalprog.graph.node.Node)


def test_duplicate_label():
    d = NormalFamily()

    graph = Graph(label="G0")
    graph.add_node(DistributionNode(d, label="X"))
    with pytest.raises(ValueError, match=re.escape("Duplicate node label: X")):
        graph.add_node(DistributionNode(d, label="X"))


@pytest.mark.parametrize(
    "use_labels",
    [pytest.param(True, id="Via labels"), pytest.param(False, id="Via variables")],
)
def test_build_graph(*, use_labels: bool) -> None:
    root_label = "root"
    outcome_label = "outcome_label"
    d = NormalFamily()

    root_node = DistributionNode(d, label=root_label)
    outcome_node = DistributionNode(d, label=outcome_label)

    graph = Graph(label="G0")
    graph.add_node(root_node)
    graph.add_node(outcome_node)

    if use_labels:
        graph.add_edge(root_label, outcome_label)
    else:
        graph.add_edge(root_node, outcome_node)

    assert graph.roots_down_to_outcome(outcome_label) == [root_node, outcome_node]


def test_cycle() -> None:
    d = NormalFamily()

    node0 = DistributionNode(d, label="X")
    node1 = DistributionNode(d, label="Y")
    node2 = DistributionNode(d, label="Z")

    graph = Graph(label="G0")
    graph.add_edge(node0, node1)
    graph.add_edge(node1, node2)
    graph.add_edge(node2, node0)

    with pytest.raises(RuntimeError, match="Graph is not acyclic."):
        graph.roots_down_to_outcome("X")
