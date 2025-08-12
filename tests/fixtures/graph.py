"""Tests for graph module."""

from collections.abc import Callable
from typing import Literal, TypeAlias

import numpy.typing as npt
import numpyro
import pytest
from numpyro.distributions import Normal

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

    def _inner(normal_graph_nodes: NormalGraphNodes | None = None) -> Graph:
        if normal_graph_nodes is None:
            normal_graph_nodes = {
                "mean": ParameterNode(label="mean"),
                "cov": ParameterNode(label="cov"),
                "outcome": DistributionNode(
                    Normal,
                    label="outcome",
                    parameters={"loc": "mean", "scale": "std"},
                ),
            }
        graph = Graph(label="normal dist")
        graph.add_edge(normal_graph_nodes["mean"], normal_graph_nodes["outcome"])
        graph.add_edge(normal_graph_nodes["cov"], normal_graph_nodes["outcome"])
        return graph

    return _inner


@pytest.fixture
def two_normal_graph() -> Callable[[float, float, float], Graph]:
    """Creates a 2 node graph:

    UX --> X

    where UX is a normal distribution with mean `mean` and covariance `cov`, and X is
    a normal distrubution with mean UX and covariance `cov2`.

    """

    def _inner(mean: float = 5.0, cov: float = 1.0, cov2: float = 1.0) -> Graph:
        graph = Graph(label="G0")
        graph.add_node(
            DistributionNode(
                Normal,
                label="UX",
                constant_parameters={"loc": mean, "scale": cov},
            )
        )
        graph.add_node(
            DistributionNode(
                Normal,
                label="X",
                parameters={"loc": "UX"},
                constant_parameters={"scale": cov2},
            )
        )
        graph.add_edge("UX", "X")

        return graph

    return _inner


@pytest.fixture
def two_normal_graph_parametrized_mean() -> Callable[[float], Graph]:
    """Creates a graph:
           SDUX   SDX
             |     |
             V     v
    mu_x --> UX --> X

    where UX is a normal distribution with mean mu_x and covariance `co`, and X is
    a normal distrubution with mean UX and covariance nu_y.

    """

    def _inner(cov: float = 1.0) -> Graph:
        graph = Graph(label="G0")
        graph.add_node(ParameterNode(label="nu_y"))
        graph.add_node(ParameterNode(label="mu_x"))
        graph.add_node(
            DistributionNode(
                Normal,
                label="UX",
                parameters={"loc": "mu_x"},
                constant_parameters={"scale": cov},
            )
        )
        graph.add_node(
            DistributionNode(
                Normal,
                label="X",
                parameters={"loc": "UX", "scale": "nu_y"},
            )
        )
        graph.add_edge("UX", "X")

        return graph

    return _inner


@pytest.fixture
def two_normal_graph_expected_model() -> Callable[..., dict[str, npt.ArrayLike]]:
    """Creates the model that the two_normal_graph should produce."""

    def _inner(mu_x: float, nu_y: float) -> dict[str, npt.ArrayLike]:
        ux = numpyro.sample("UX", Normal(loc=mu_x, scale=1.0))
        x = numpyro.sample("X", Normal(loc=ux, scale=nu_y))

        return {"X": x, "UX": ux}

    return _inner
