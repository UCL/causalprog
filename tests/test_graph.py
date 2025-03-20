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
    "use_labels",
    [pytest.param(True, id="Via labels"), pytest.param(False, id="Via variables")],
)
def test_build_graph(*, use_labels: bool) -> None:
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

    assert graph.roots_down_to_outcome(outcome_label) == [root_node, outcome_node]


def test_roots_down_to_outcome() -> None:
    d = causalprog.graph.node.NormalDistribution()

    graph = causalprog.graph.Graph("G0")

    u = causalprog.graph.DistributionNode(d, "U")
    v = causalprog.graph.DistributionNode(d, "V")
    w = causalprog.graph.DistributionNode(d, "W")
    x = causalprog.graph.DistributionNode(d, "X")
    y = causalprog.graph.DistributionNode(d, "Y")
    z = causalprog.graph.DistributionNode(d, "Z")

    graph.add_node(u)
    graph.add_node(v)
    graph.add_node(w)
    graph.add_node(x)
    graph.add_node(y)
    graph.add_node(z)

    graph.add_edge("V", "W")
    graph.add_edge("V", "X")
    graph.add_edge("V", "Y")
    graph.add_edge("X", "Z")
    graph.add_edge("Y", "Z")
    graph.add_edge("U", "Z")

    assert graph.roots_down_to_outcome("V") == [v]
    assert graph.roots_down_to_outcome("W") == [v, w]
    nodes = graph.roots_down_to_outcome("Z")
    assert len(nodes) == 5  # noqa: PLR2004
    assert (
        nodes.index(v)
        < min(nodes.index(x), nodes.index(y))
        < max(nodes.index(x), nodes.index(y))
        < nodes.index(z)
    )
    assert nodes.index(u) < nodes.index(z)


def test_cycle() -> None:
    d = causalprog.graph.node.NormalDistribution()

    node0 = causalprog.graph.DistributionNode(d, "X")
    node1 = causalprog.graph.DistributionNode(d, "Y")
    node2 = causalprog.graph.DistributionNode(d, "Z")

    graph = causalprog.graph.Graph("G0")
    graph.add_edge(node0, node1)
    graph.add_edge(node1, node2)
    graph.add_edge(node2, node0)

    with pytest.raises(RuntimeError, match="Graph is not acyclic."):
        graph.roots_down_to_outcome("X")


@pytest.mark.parametrize(
    ("mean", "stdev", "samples", "rtol"),
    [
        pytest.param(1.0, 1.0, 10, 1, id="std normal, 10 samples"),
        pytest.param(2.0, 0.8, 1000, 1e-1, id="non-standard normal, 100 samples"),
        pytest.param(1.0, 1.0, 100000, 1e-2, id="std normal, 10^5 samples"),
        pytest.param(1.0, 1.0, 10000000, 1e-3, id="std normal, 10^7 samples"),
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


@pytest.mark.parametrize(
    ("mean", "stdev", "stdev2", "samples", "rtol"),
    [
        pytest.param(
            1.0,
            1.0,
            0.8,
            100,
            1,
            id="N(mean=N(mean=0, stdev=1), stdev=0.8), 100 samples",
        ),
        pytest.param(
            3.0,
            0.5,
            1.0,
            10000,
            1e-1,
            id="N(mean=N(mean=3, stdev=0.5), stdev=1), 10^4 samples",
        ),
        pytest.param(
            2.0,
            0.7,
            0.8,
            1000000,
            1e-2,
            id="N(mean=N(mean=2, stdev=0.7), stdev=0.8), 10^6 samples",
        ),
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
