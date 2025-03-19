"""Tests for graph module."""

import re

import numpy as np
import pytest

import causalprog


def test_label():
    d = causalprog.graph.node.NormalDistribution()
    node = causalprog.graph.DistributionNode(d)
    node2 = causalprog.graph.DistributionNode(d, "node1")
    node3 = causalprog.graph.DistributionNode(d, "Y")
    node4 = causalprog.graph.DistributionNode(d)
    node_copy = node

    assert node._label == node_copy._label  # noqa: SLF001
    assert node._label == node4._label  # noqa: SLF001

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
    d = causalprog.graph.node.NormalDistribution()

    graph = causalprog.graph.Graph("G0")
    graph.add_node(causalprog.graph.DistributionNode(d, "X"))
    with pytest.raises(ValueError, match=re.escape("Duplicate node label: X")):
        graph.add_node(causalprog.graph.DistributionNode(d, "X"))


def test_simple_graph():
    d = causalprog.graph.node.NormalDistribution()
    n_x = causalprog.graph.DistributionNode(d, "N_X")
    n_m = causalprog.graph.DistributionNode(d, "N_M")
    u_y = causalprog.graph.DistributionNode(d, "U_Y")
    x = causalprog.graph.DistributionNode(d, "X")
    m = causalprog.graph.DistributionNode(d, "M")
    y = causalprog.graph.DistributionNode(d, "Y", is_outcome=True)

    graph = causalprog.graph.Graph("G0")
    graph.add_edge(n_x, x)
    graph.add_edge(n_m, m)
    graph.add_edge(u_y, y)
    graph.add_edge(x, m)
    graph.add_edge(m, y)

    assert graph.label == "G0"


def test_simple_graph_build_using_labels():
    d = causalprog.graph.node.NormalDistribution()

    graph = causalprog.graph.Graph("G0")
    graph.add_node(causalprog.graph.DistributionNode(d, "N_X"))
    graph.add_node(causalprog.graph.DistributionNode(d, "N_M"))
    graph.add_node(causalprog.graph.DistributionNode(d, "U_Y"))
    graph.add_node(causalprog.graph.DistributionNode(d, "X"))
    graph.add_node(causalprog.graph.DistributionNode(d, "M"))
    graph.add_node(causalprog.graph.DistributionNode(d, "Y", is_outcome=True))

    graph.add_edge("N_X", "X")
    graph.add_edge("N_M", "M")
    graph.add_edge("U_Y", "Y")
    graph.add_edge("X", "M")
    graph.add_edge("M", "Y")

    assert graph.label == "G0"


@pytest.mark.parametrize("mean", [1.0, 2.0])
@pytest.mark.parametrize("stdev", [0.8, 1.0])
@pytest.mark.parametrize(
    ("samples", "rtol"),
    [
        (10, 1),
        (1000, 1e-1),
        (100000, 1e-2),
        (10000000, 1e-3),
    ],
)
def test_single_normal_node(samples, rtol, mean, stdev):
    normal = causalprog.graph.node.NormalDistribution(mean, stdev)
    node = causalprog.graph.DistributionNode(normal, "X", is_outcome=True)

    graph = causalprog.graph.Graph("G0")
    graph.add_node(node)

    assert np.isclose(
        causalprog.algorithms.expectation(graph, samples=samples), mean, rtol=rtol
    )
    assert np.isclose(
        causalprog.algorithms.standard_deviation(graph, samples=samples),
        stdev,
        rtol=rtol,
    )


@pytest.mark.parametrize("mean", [1.0, 2.0])
@pytest.mark.parametrize("stdev", [0.8, 1.0])
@pytest.mark.parametrize("stdev2", [0.8, 1.0])
@pytest.mark.parametrize(
    ("samples", "rtol"),
    [
        (100, 1),
        (10000, 1e-1),
        (1000000, 1e-2),
    ],
)
def test_two_node_graph(samples, rtol, mean, stdev, stdev2):
    normal = causalprog.graph.node.NormalDistribution(mean, stdev)
    normal2 = causalprog.graph.node.NormalDistribution("UX", stdev2)

    graph = causalprog.graph.Graph("G0")
    graph.add_node(causalprog.graph.DistributionNode(normal, "UX"))
    graph.add_node(causalprog.graph.DistributionNode(normal2, "X", is_outcome=True))
    graph.add_edge("UX", "X")

    assert np.isclose(
        causalprog.algorithms.expectation(graph, samples=samples), mean, rtol=rtol
    )
    assert np.isclose(
        causalprog.algorithms.standard_deviation(graph, samples=samples),
        np.sqrt(stdev**2 + stdev2**2),
        rtol=rtol,
    )
