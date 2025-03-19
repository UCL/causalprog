"""Tests for graph module."""

import re

import numpy as np
import pytest

import causalprog


def test_label():
    d = causalprog.graph.node.NormalDistribution()
    node = causalprog.graph.DistributionNode(d, "X")
    node2 = causalprog.graph.DistributionNode(d, "Y")
    node_copy = node

    assert node.label == node_copy.label == "X"
    assert node.label != node2.label
    assert node2.label == "Y"

    assert isinstance(node, causalprog.graph.node.Node)
    assert isinstance(node2, causalprog.graph.node.Node)


def test_duplicate_label():
    d = causalprog.graph.node.NormalDistribution()

    graph = causalprog.graph.Graph("G0")
    graph.add_node(causalprog.graph.DistributionNode(d, "X"))
    with pytest.raises(ValueError, match=re.escape("Duplicate node label: X")):
        graph.add_node(causalprog.graph.DistributionNode(d, "X"))


@pytest.mark.parametrize(
    ("use_labels",),
    [pytest.param(True, id="Via labels"), pytest.param(False, id="Via variables")]
)
def test_build_graph(use_labels: bool) -> None:
    root_label = "root"
    outcome_label = "outcome_label"
    d = causalprog.graph.node.NormalDistribution()

    root_node = causalprog.graph.DistributionNode(d, root_label)
    outcome_node = causalprog.graph.DistributionNode(d, outcome_label, is_outcome=True)

    graph = causalprog.graph.Graph("G0")
    graph.add_node(root_node)
    graph.add_node(outcome_node)

    if use_labels:
        graph.add_edge(root_label, outcome_label)
    else:
        graph.add_edge(root_node, outcome_node)

    nodes = graph.roots_down_to_outcome(outcome_label)
    assert nodes == [root_node, outcome_node]


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
