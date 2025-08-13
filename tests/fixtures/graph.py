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
        graph = Graph(label="normal_graph")
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

    Parameter nodes are included if no values are given for the mean and covariances.

    """

    def _inner(
        mean: float | None = None, cov: float | None = None, cov2: float | None = None
    ) -> Graph:
        graph = Graph(label="two_normal_graph")

        x_parameters = {"loc": "UX"}
        x_constant_parameters = {}
        ux_parameters = {}
        ux_constant_parameters = {}
        if mean is None:
            graph.add_node(ParameterNode(label="mean"))
            ux_parameters["loc"] = "mean"
        else:
            ux_constant_parameters["loc"] = mean
        if cov is None:
            graph.add_node(ParameterNode(label="cov"))
            ux_parameters["scale"] = "cov"
        else:
            ux_constant_parameters["scale"] = cov
        if cov2 is None:
            graph.add_node(ParameterNode(label="cov2"))
            x_parameters["scale"] = "cov2"
        else:
            x_constant_parameters["scale"] = cov2

        graph.add_node(
            DistributionNode(
                Normal,
                label="UX",
                parameters=ux_parameters,
                constant_parameters=ux_constant_parameters,
            )
        )
        graph.add_node(
            DistributionNode(
                Normal,
                label="X",
                parameters=x_parameters,
                constant_parameters=x_constant_parameters,
            )
        )
        graph.add_edge("UX", "X")
        for node in ux_parameters.values():
            graph.add_edge(node, "UX")
        for node in x_parameters.values():
            graph.add_edge(node, "X")

        return graph

    return _inner


@pytest.fixture
def two_normal_graph_expected_model() -> Callable[..., dict[str, npt.ArrayLike]]:
    """Creates the model that the two_normal_graph should produce."""

    def _inner(mean: float, cov2: float) -> dict[str, npt.ArrayLike]:
        ux = numpyro.sample("UX", Normal(loc=mean, scale=1.0))
        x = numpyro.sample("X", Normal(loc=ux, scale=cov2))

        return {"X": x, "UX": ux}

    return _inner
