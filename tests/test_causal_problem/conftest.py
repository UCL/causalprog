import pytest
from numpyro.distributions import Normal

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
        Normal,
        label="x",
        parameters={"loc": "mu_x"},
        constant_parameters={"scale": 1.0},
    )
    nu_y = ParameterNode(label="nu_y")
    y = DistributionNode(
        Normal,
        label="y",
        parameters={"loc": "x", "scale": "nu_y"},
    )

    graph.add_edge(mu_x, x)
    graph.add_edge(nu_y, y)
    graph.add_edge(x, y)
    return graph
