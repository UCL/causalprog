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
def normal_graph() -> Callable[[float, float], Graph]:
    """Creates a graph with one normal distribution X.

    Parameter nodes are included if no values are given for the mean and covariance.
    """

    def _inner(mean: float | None = None, cov: float | None = None):
        graph = Graph(label="normal dist")
        parameters = {}
        constant_parameters = {}
        if mean is None:
            graph.add_node(ParameterNode(label="mean"))
            parameters["loc"] = "mean"
        else:
            constant_parameters["loc"] = mean
        if cov is None:
            graph.add_node(ParameterNode(label="cov"))
            parameters["scale"] = "cov"
        else:
            constant_parameters["scale"] = cov
        graph.add_node(
            DistributionNode(
                Normal,
                label="X",
                parameters=parameters,
                constant_parameters=constant_parameters,
            )
        )
        for node in parameters.values():
            graph.add_edge(node, "X")
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
                constant_parameters={"loc": mean, "scale": cov**2},
            )
        )
        graph.add_node(
            DistributionNode(
                Normal,
                label="X",
                parameters={"loc": "UX"},
                constant_parameters={"scale": cov2**2},
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
