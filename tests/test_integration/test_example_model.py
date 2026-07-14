"""Test the example model special graph."""

from causalprog.algorithms import replace_node
from causalprog.graph import ContinuousRandomVariableNode
from causalprog.graph.ricardo import example_model


def test_example_model_update():
    """Test ust of replace_node to reverse an edge."""
    graph = example_model(
        compute_u_x=lambda _data: 1.0,
        compute_u_y=lambda _data: 1.0,
        compute_phi_x=lambda _data: 1.0,
        compute_x=lambda _data: 1.0,
        compute_y=lambda _data: 1.0,
    )
    assert len(graph.nodes) == 8
    assert len(graph.edges) == 8

    g = replace_node(
        graph,
        "x",
        ContinuousRandomVariableNode(
            label="x_updated",
            parents=["z", "phi_x"],
        ),
    )
    updated_graph = replace_node(
        g,
        "u_x",
        ContinuousRandomVariableNode(
            label="u_x_updated",
            parents=["x_updated", "c"],
        ),
    )

    original_edges = {(e[0].label, e[1].label) for e in graph.edges}
    assert original_edges == {
        ("l", "phi_x"),
        ("c", "u_y"),
        ("c", "u_x"),
        ("u_x", "x"),
        ("phi_x", "x"),
        ("z", "x"),
        ("u_y", "y"),
        ("x", "y"),
    }

    assert len(updated_graph.nodes) == 8
    assert len(updated_graph.edges) == 8
    edges = {(e[0].label, e[1].label) for e in updated_graph.edges}
    assert edges == {
        ("l", "phi_x"),
        ("c", "u_y"),
        ("c", "u_x_updated"),
        ("x_updated", "u_x_updated"),
        ("phi_x", "x_updated"),
        ("z", "x_updated"),
        ("u_y", "y"),
        ("x_updated", "y"),
    }
