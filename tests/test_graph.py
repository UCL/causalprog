"""Tests for graph module."""

import re

import numpy as np
import pytest

import causalprog


def test_label():
    family = causalprog.graph.node.DistributionFamily()
    node = causalprog.graph.RootDistributionNode(family)
    node2 = causalprog.graph.RootDistributionNode(family, "node1")
    node3 = causalprog.graph.RootDistributionNode(family, "Y")
    node4 = causalprog.graph.DistributionNode(family)
    node_copy = node

    assert node.label == node_copy.label
    assert node.label == node4.label

    graph = causalprog.graph.Graph("G0")
    graph.add_node(node)
    graph.add_node(node2)
    graph.add_node(node3)
    graph.add_node(node4)

    assert node.label is not None
    assert node2.label is not None
    assert node3.label is not None
    assert node4.label is not None
    assert node.label == node_copy.label
    assert node.label != node2.label
    assert node.label != node3.label
    assert node.label != node4.label

    assert isinstance(node, causalprog.graph.node.Node)
    assert isinstance(node2, causalprog.graph.node.Node)
    assert isinstance(node3, causalprog.graph.node.Node)
    assert isinstance(node4, causalprog.graph.node.Node)


def test_duplicate_label():
    family = causalprog.graph.node.DistributionFamily()

    graph = causalprog.graph.Graph("G0")
    graph.add_node(causalprog.graph.RootDistributionNode(family, "X"))
    with pytest.raises(ValueError, match=re.escape("Duplicate node label: X")):
        graph.add_node(causalprog.graph.RootDistributionNode(family, "X"))


def test_simple_graph():
    family = causalprog.graph.node.DistributionFamily()
    n_x = causalprog.graph.RootDistributionNode(family, "N_X")
    n_m = causalprog.graph.RootDistributionNode(family, "N_M")
    u_y = causalprog.graph.RootDistributionNode(family, "U_Y")
    x = causalprog.graph.DistributionNode(family, "X")
    m = causalprog.graph.DistributionNode(family, "M")
    y = causalprog.graph.DistributionNode(family, "Y", is_outcome=True)

    graph = causalprog.graph.Graph("G0")
    graph.add_edge(n_x, x)
    graph.add_edge(n_m, m)
    graph.add_edge(u_y, y)
    graph.add_edge(x, m)
    graph.add_edge(m, y)

    assert graph.label == "G0"


def test_simple_graph_build_using_labels():
    family = causalprog.graph.node.DistributionFamily()

    graph = causalprog.graph.Graph("G0")
    graph.add_node(causalprog.graph.RootDistributionNode(family, "N_X"))
    graph.add_node(causalprog.graph.RootDistributionNode(family, "N_M"))
    graph.add_node(causalprog.graph.RootDistributionNode(family, "U_Y"))
    graph.add_node(causalprog.graph.DistributionNode(family, "X"))
    graph.add_node(causalprog.graph.DistributionNode(family, "M"))
    graph.add_node(causalprog.graph.DistributionNode(family, "Y", is_outcome=True))

    graph.add_edge("N_X", "X")
    graph.add_edge("N_M", "M")
    graph.add_edge("U_Y", "Y")
    graph.add_edge("X", "M")
    graph.add_edge("M", "Y")

    assert graph.label == "G0"


def test_single_normal_node():
    normal = causalprog.graph.node.Distribution()
    node = causalprog.graph.RootDistributionNode(normal, "X", is_outcome=True)

    graph = causalprog.graph.Graph("G0")
    graph.add_node(node)

    assert np.isclose(
        causalprog.algorithms.expectation(graph, samples=1000), 1.0, rtol=1e-1
    )
    assert np.isclose(
        causalprog.algorithms.expectation(graph, samples=100000), 1.0, rtol=1e-2
    )
