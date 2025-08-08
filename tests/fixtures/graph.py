"""Tests for graph module."""

from collections.abc import Callable
from typing import Literal, TypeAlias

import pytest

from causalprog.distribution.normal import NormalFamily
from causalprog.graph import DistributionNode, Graph, ParameterNode

NormalGraphNodeNames: TypeAlias = Literal["mean", "cov", "outcome"]
NormalGraphNodes: TypeAlias = dict[
    NormalGraphNodeNames, DistributionNode | ParameterNode
]


@pytest.fixture
def normal_graph() -> Callable[[NormalGraphNodes | None], Graph]:
    """Creates a 3-node graph:

    mean (P)          cov (P)
      |---> outcome <----|

    where outcome is a normal distribution.

    Parameter nodes are initialised with no `value` set.
    """

    def _normal_graph(normal_graph_nodes: NormalGraphNodes | None = None) -> Graph:
        if normal_graph_nodes is None:
            normal_graph_nodes = {
                "mean": ParameterNode(label="mean"),
                "cov": ParameterNode(label="cov"),
                "outcome": DistributionNode(
                    NormalFamily(),
                    label="outcome",
                    parameters={"mean": "mean", "cov": "std"},
                ),
            }
        graph = Graph(label="normal dist")
        graph.add_edge(normal_graph_nodes["mean"], normal_graph_nodes["outcome"])
        graph.add_edge(normal_graph_nodes["cov"], normal_graph_nodes["outcome"])
        return graph

    return _normal_graph


@pytest.fixture
def ux_x_graph() -> Callable[[float, float, float], Graph]:
    """Creates a 2 node graph:

    UX --> X

    where EX is a normal distribution with mean `mean` and covariance `cov`, and X is
    a normal distrubution with mean UX and covariance `cov2`.

    """

    def _ux_x_graph(mean: float = 5.0, cov: float = 1.0, cov2: float = 1.0) -> Graph:
        graph = Graph(label="G0")
        graph.add_node(
            DistributionNode(
                NormalFamily(),
                label="UX",
                constant_parameters={"mean": mean, "cov": cov},
            )
        )
        graph.add_node(
            DistributionNode(
                NormalFamily(),
                label="X",
                parameters={"mean": "UX"},
                constant_parameters={"cov": cov2},
            )
        )
        graph.add_edge("UX", "X")

        return graph

    return _ux_x_graph
