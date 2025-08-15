from collections.abc import Callable

import jax.numpy as jnp

from causalprog.causal_problem import CausalProblem
from causalprog.graph import Graph


def test_graph_and_parameter_interactions(
    two_normal_graph: Callable[..., Graph],
    raises_context,
) -> None:
    cp = CausalProblem(label="TestCP")

    # Without a graph, we can't do anything
    with raises_context(ValueError("No graph set for TestCP")):
        cp.graph  # noqa: B018
    with raises_context(ValueError("No graph set for TestCP")):
        cp.parameter_values  # noqa: B018

    # Cannot set graph to non-graph value
    with raises_context(TypeError("TestCP.graph must be a Graph instance")):
        cp.graph = 1.0

    # Provide an actual graph value
    cp.graph = two_normal_graph(cov=1.0)

    # We should now be able to fetch parameter values, but they are all unset.
    assert jnp.all(jnp.isnan(cp.parameter_vector))
    assert cp.parameter_vector.shape == (len(cp.graph.parameter_nodes),)
    assert all(jnp.isnan(value) for value in cp.parameter_values.values())
    assert set(cp.parameter_values.keys()) == {"mean", "cov2"}

    # Users should only ever need to set parameter values via their names.
    cp.set_parameter_values(mean=1.0, cov2=2.0)
    assert cp.parameter_values == {"mean": 1.0, "cov2": 2.0}
    # We don't know which way round the internal parameter vector is being stored,
    # but that doesn't matter. We do know that it should contain the values 1 & 2
    # in some order though.
    assert jnp.allclose(cp.parameter_vector, jnp.array([1.0, 2.0])) or jnp.allclose(
        cp.parameter_vector, jnp.array([2.0, 1.0])
    )
