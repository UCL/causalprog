import re

import jax.numpy as jnp
import pytest

from causalprog.causal_problem import CausalProblem
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


def test_graph_and_parameter_interactions(graph: Graph) -> None:
    cp = CausalProblem(label="TestCP")

    # Without a graph, we can't do anything
    with pytest.raises(ValueError, match=re.escape("No graph set for TestCP")):
        cp.graph  # noqa: B018
    with pytest.raises(ValueError, match=re.escape("No graph set for TestCP")):
        cp.parameter_values  # noqa: B018

    # Cannot set graph to non-graph value
    with pytest.raises(
        TypeError, match=re.escape("TestCP.graph must be a Graph instance")
    ):
        cp.graph = 1.0

    # Provide an actual graph value
    cp.graph = graph

    # We should now be able to fetch parameter values, but they are all unset.
    assert jnp.all(jnp.isnan(cp.parameter_vector))
    assert cp.parameter_vector.shape == (len(cp.graph.parameter_nodes),)
    assert all(jnp.isnan(value) for value in cp.parameter_values.values())
    assert set(cp.parameter_values.keys()) == {"mu_x", "nu_y"}

    # Users should only ever need to set parameter values via their names.
    cp.set_parameter_values(mu_x=1.0, nu_y=2.0)
    assert cp.parameter_values == {"mu_x": 1.0, "nu_y": 2.0}
    # We don't know which way round the internal parameter vector is being stored,
    # but that doesn't matter. We do know that it should contain the values 1 & 2
    # in some order though.
    assert jnp.allclose(cp.parameter_vector, jnp.array([1.0, 2.0])) or jnp.allclose(
        cp.parameter_vector, jnp.array([2.0, 1.0])
    )
