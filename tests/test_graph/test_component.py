import pytest
from numpyro.distributions import Normal

from causalprog.graph import DistributionNode, Graph


@pytest.mark.parametrize(
    "param_values",
    [
        pytest.param({"mean": 0.0, "cov2": 1.0}, id="mean(X) = 0, cov(UX) = 1"),
        pytest.param({"mean": 1.0, "cov2": 2.0}, id="mean(X) = 1, cov(UX) = 2"),
    ],
)
def test_component_node(param_values):
    graph = Graph(label="test_graph")
    graph.add_node(
        DistributionNode(
            Normal,
            label="X",
            shape=(3, 1, 4),
            constant_parameters=param_values,
        )
    )
    graph.add_node(graph.get_node("X")[0])
    graph.add_node(graph.get_node("X")[1, 0, 2])
    graph.add_node(graph.get_node("X")[1, :, 2])

    assert graph.get_node("X").shape == (3, 1, 4)
    assert graph.get_node("X[0]").shape == (1, 4)
    assert graph.get_node("X[1, 0, 2]").shape == ()
    assert graph.get_node("X[1, :, 2]").shape == (1,)
