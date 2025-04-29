"""Tests for graph module."""

import re

import jax
import numpy as np
import pytest

import causalprog
from causalprog.distribution.normal import NormalFamily
from causalprog.graph import DistributionNode, Graph, ParameterNode


def test_label():
    d = NormalFamily()
    node = DistributionNode(d, "X")
    node2 = DistributionNode(d, "Y")
    node_copy = node

    assert node.label == node_copy.label == "X"
    assert node.label != node2.label
    assert node2.label == "Y"

    assert isinstance(node, causalprog.graph.node.Node)
    assert isinstance(node2, causalprog.graph.node.Node)


def test_duplicate_label():
    d = NormalFamily()

    graph = Graph("G0")
    graph.add_node(DistributionNode(d, "X"))
    with pytest.raises(ValueError, match=re.escape("Duplicate node label: X")):
        graph.add_node(DistributionNode(d, "X"))


@pytest.mark.parametrize(
    "use_labels",
    [pytest.param(True, id="Via labels"), pytest.param(False, id="Via variables")],
)
def test_build_graph(*, use_labels: bool) -> None:
    root_label = "root"
    outcome_label = "outcome_label"
    d = NormalFamily()

    root_node = DistributionNode(d, root_label)
    outcome_node = DistributionNode(d, outcome_label, is_outcome=True)

    graph = Graph("G0")
    graph.add_node(root_node)
    graph.add_node(outcome_node)

    if use_labels:
        graph.add_edge(root_label, outcome_label)
    else:
        graph.add_edge(root_node, outcome_node)

    assert graph.roots_down_to_outcome(outcome_label) == [root_node, outcome_node]


def test_roots_down_to_outcome() -> None:
    d = NormalFamily()

    graph = Graph("G0")

    u = DistributionNode(d, "U")
    v = DistributionNode(d, "V")
    w = DistributionNode(d, "W")
    x = DistributionNode(d, "X")
    y = DistributionNode(d, "Y")
    z = DistributionNode(d, "Z")

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
    d = NormalFamily()

    node0 = DistributionNode(d, "X")
    node1 = DistributionNode(d, "Y")
    node2 = DistributionNode(d, "Z")

    graph = Graph("G0")
    graph.add_edge(node0, node1)
    graph.add_edge(node1, node2)
    graph.add_edge(node2, node0)

    with pytest.raises(RuntimeError, match="Graph is not acyclic."):
        graph.roots_down_to_outcome("X")


@pytest.mark.parametrize(
    ("mean", "stdev", "samples", "rtol"),
    [
        pytest.param(1.0, 1.0, 10, 1, id="N(mean=1, stdev=1), 10 samples"),
        pytest.param(2.0, 0.8, 1000, 1e-1, id="N(mean=2, stdev=0.8), 1000 samples"),
        pytest.param(1.0, 0.8, 100000, 1e-2, id="N(mean=1, stdev=0.8), 10^5 samples"),
        pytest.param(1.0, 1.2, 10000000, 1e-3, id="N(mean=1, stdev=1.2), 10^7 samples"),
    ],
)
def test_single_normal_node(samples, rtol, mean, stdev, rng_key):
    node = DistributionNode(
        NormalFamily(),
        "X",
        constant_parameters={"mean": mean, "cov": stdev**2},
        is_outcome=True,
    )

    graph = Graph("G0")
    graph.add_node(node)

    # To compensate for rng-key splitting in sample methods, note the "split" key
    # that is actually used to draw the samples from the distribution, so we can
    # attempt to replicate its behaviour explicitly.
    key = jax.random.split(rng_key, 1)[0]
    what_we_should_get = jax.random.multivariate_normal(
        key, jax.numpy.atleast_1d(mean), jax.numpy.atleast_2d(stdev**2), shape=samples
    )
    expected_mean = what_we_should_get.mean()
    expected_std_dev = what_we_should_get.std()

    # Check within hand-computation
    assert np.isclose(
        causalprog.algorithms.expectation(
            graph, outcome_node_label="X", samples=samples, rng_key=rng_key
        ),
        mean,
        rtol=rtol,
    )
    assert np.isclose(
        causalprog.algorithms.standard_deviation(
            graph, outcome_node_label="X", samples=samples, rng_key=rng_key
        ),
        stdev,
        rtol=rtol,
    )
    # Check within computational distance
    assert np.isclose(
        causalprog.algorithms.expectation(
            graph, outcome_node_label="X", samples=samples, rng_key=rng_key
        ),
        expected_mean,
    )
    assert np.isclose(
        causalprog.algorithms.standard_deviation(
            graph, outcome_node_label="X", samples=samples, rng_key=rng_key
        ),
        expected_std_dev,
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
def test_two_node_graph(samples, rtol, mean, stdev, stdev2, rng_key):
    if samples > 100:  # noqa: PLR2004
        pytest.xfail("Test currently too slow")
    graph = causalprog.graph.Graph("G0")
    graph.add_node(
        DistributionNode(
            NormalFamily(), "UX", constant_parameters={"mean": mean, "cov": stdev**2}
        )
    )
    graph.add_node(
        DistributionNode(
            NormalFamily(),
            "X",
            parameters={"mean": "UX"},
            constant_parameters={"cov": stdev2**2},
            is_outcome=True,
        )
    )
    graph.add_edge("UX", "X")

    assert np.isclose(
        causalprog.algorithms.expectation(
            graph, outcome_node_label="X", samples=samples, rng_key=rng_key
        ),
        mean,
        rtol=rtol,
    )
    assert np.isclose(
        causalprog.algorithms.standard_deviation(
            graph, outcome_node_label="X", samples=samples, rng_key=rng_key
        ),
        np.sqrt(stdev**2 + stdev2**2),
        rtol=rtol,
    )


def test_paramater_node(rng_key):
    node = ParameterNode("mu")

    with pytest.raises(ValueError, match="Cannot sample"):
        node.sample({}, 1, rng_key)

    node.value = 0.3

    assert np.allclose(node.sample({}, 10, rng_key)[0], [0.3] * 10)
