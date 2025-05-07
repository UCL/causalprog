import pytest

from causalprog.distribution.normal import NormalFamily
from causalprog.graph import DistributionNode, Graph, ParameterNode


@pytest.fixture
def graph() -> Graph:
    """A graph fixture that can be re-used during testing.

    Nodes are:
    - mu_X (Parameter)
    - X ~ N(mu_X, 1.0)
    - nu_Y (Parameter)
    - Y ~ N(X, nu_Y)
    """
    graph = Graph(label="Graph")

    mu_x = ParameterNode(label="mu_x")
    x = DistributionNode(
        NormalFamily(),
        label="x",
        parameters={"mean": "mu_x"},
        constant_parameters={"cov": 1.0},
    )
    nu_y = ParameterNode(label="nu_y")
    y = DistributionNode(
        NormalFamily(),
        label="y",
        parameters={"mean": "x", "cov": "nu_y"},
        is_outcome=True,
    )

    graph.add_edge(mu_x, x)
    graph.add_edge(nu_y, y)
    graph.add_edge(x, y)
    return graph
