"""Tests for graph algorithms."""

from typing import Literal, TypeAlias

import jax
import numpy as np
import pytest

import causalprog
from causalprog.distribution.normal import NormalFamily
from causalprog.graph import DistributionNode, Graph, ParameterNode

NormalGraphNodeNames: TypeAlias = Literal["mean", "cov", "outcome"]
NormalGraphNodes: TypeAlias = dict[
    NormalGraphNodeNames, DistributionNode | ParameterNode
]


def test_roots_down_to_outcome() -> None:
    d = NormalFamily()

    graph = Graph(label="G0")

    graph.add_node(DistributionNode(d, label="U"))
    graph.add_node(DistributionNode(d, label="V"))
    graph.add_node(DistributionNode(d, label="W"))
    graph.add_node(DistributionNode(d, label="X"))
    graph.add_node(DistributionNode(d, label="Y"))
    graph.add_node(DistributionNode(d, label="Z"))

    edges = [
        ["V", "W"],
        ["V", "X"],
        ["V", "Y"],
        ["X", "Z"],
        ["Y", "Z"],
        ["U", "Z"],
    ]
    for e in edges:
        graph.add_edge(*e)

    assert graph.roots_down_to_outcome("V") == [graph.get_node("V")]
    assert graph.roots_down_to_outcome("W") == [
        graph.get_node("V"),
        graph.get_node("W"),
    ]
    nodes = graph.roots_down_to_outcome("Z")
    assert len(nodes) == 5  # noqa: PLR2004
    for e in edges:
        if "W" not in e:
            assert nodes.index(graph.get_node(e[0])) < nodes.index(graph.get_node(e[1]))


def test_do(rng_key):
    graph = causalprog.graph.Graph(label="G0")
    graph.add_node(
        DistributionNode(
            NormalFamily(), label="UX", constant_parameters={"mean": 5.0, "cov": 1.0}
        )
    )
    graph.add_node(
        DistributionNode(
            NormalFamily(),
            label="X",
            parameters={"mean": "UX"},
            constant_parameters={"cov": 1.0},
        )
    )
    graph.add_edge("UX", "X")

    graph2 = causalprog.algorithms.do(graph, "UX", 4.0)

    assert "mean" in graph.get_node("X").parameters
    assert "mean" not in graph.get_node("X").constant_parameters
    assert "mean" not in graph2.get_node("X").parameters
    assert "mean" in graph2.get_node("X").constant_parameters

    assert np.isclose(
        causalprog.algorithms.expectation(
            graph, outcome_node_label="X", samples=1000, rng_key=rng_key
        ),
        5.0,
        rtol=1e-1,
    )

    assert np.isclose(
        causalprog.algorithms.expectation(
            graph2, outcome_node_label="X", samples=1000, rng_key=rng_key
        ),
        4.0,
        rtol=1e-1,
    )


@pytest.mark.parametrize(
    ("mean", "stdev", "samples", "rtol"),
    [
        pytest.param(1.0, 1.0, 10, 1, id="N(mean=1, stdev=1), 10 samples"),
        pytest.param(2.0, 0.8, 1000, 1e-1, id="N(mean=2, stdev=0.8), 1000 samples"),
        pytest.param(1.0, 0.8, 100000, 1e-2, id="N(mean=1, stdev=0.8), 10^5 samples"),
        pytest.param(1.0, 1.2, 10000000, 1e-3, id="N(mean=1, stdev=1.2), 10^7 samples"),
    ],
)
def test_mean_stdev_single_normal_node(samples, rtol, mean, stdev, rng_key):
    node = DistributionNode(
        NormalFamily(),
        label="X",
        constant_parameters={"mean": mean, "cov": stdev**2},
    )

    graph = Graph(label="G0")
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
def test_mean_stdev_two_node_graph(samples, rtol, mean, stdev, stdev2, rng_key):
    if samples > 100:  # noqa: PLR2004
        pytest.xfail("Test currently too slow")
    graph = causalprog.graph.Graph(label="G0")
    graph.add_node(
        DistributionNode(
            NormalFamily(),
            label="UX",
            constant_parameters={"mean": mean, "cov": stdev**2},
        )
    )
    graph.add_node(
        DistributionNode(
            NormalFamily(),
            label="X",
            parameters={"mean": "UX"},
            constant_parameters={"cov": stdev2**2},
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
