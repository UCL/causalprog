"""Tests for graph module."""

from typing import Literal, TypeAlias

import pytest

from causalprog.distribution.normal import NormalFamily
from causalprog.graph import DistributionNode, Graph, ParameterNode

NormalGraphNodeNames: TypeAlias = Literal["mean", "cov", "outcome"]
NormalGraphNodes: TypeAlias = dict[
    NormalGraphNodeNames, DistributionNode | ParameterNode
]


@pytest.fixture
def normal_graph_nodes() -> NormalGraphNodes:
    """Collection of Nodes used to construct `normal_graph`.

    See `normal_graph` docstring for more details.
    """
    return {
        "mean": ParameterNode(label="mean"),
        "cov": ParameterNode(label="cov"),
        "outcome": DistributionNode(
            NormalFamily(), label="outcome", parameters={"mean": "mean", "cov": "std"}
        ),
    }


@pytest.fixture
def normal_graph(normal_graph_nodes: NormalGraphNodes) -> Graph:
    """Creates a 3-node graph:

    mean (P)          cov (P)
      |---> outcome <----|

    where outcome is a normal distribution.

    Parameter nodes are initialised with no `value` set.
    """
    graph = Graph(label="normal dist")
    graph.add_edge(normal_graph_nodes["mean"], normal_graph_nodes["outcome"])
    graph.add_edge(normal_graph_nodes["cov"], normal_graph_nodes["outcome"])
    return graph
