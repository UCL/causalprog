"""Test the example model special graph."""

from causalprog.algorithms import replace_node
from causalprog.graph import ContinuousRandomVariableNode
from causalprog.graph.continuous_treatment import continuous_treatment_model


def test_treatment_model():
    graph = continuous_treatment_model(
        compute_u_x=lambda _data: 1.0,
        compute_u_y=lambda _data: 1.0,
        compute_x=lambda _data: 1.0,
        compute_y=lambda _data: 1.0,
    )
    assert len(graph.nodes) == 7
    assert len(graph.edges) == 9
    edges = {(e[0].label, e[1].label) for e in graph.edges}
    assert edges == {
        ("l", "u_x"),
        ("c", "u_y"),
        ("c", "u_x"),
        ("u_x", "u_y"),
        ("u_x", "x"),
        ("l", "x"),
        ("z", "x"),
        ("u_y", "y"),
        ("x", "y"),
    }


def test_treatment_model_update():
    """Test ust of replace_node to reverse an edge."""
    graph = continuous_treatment_model(
        compute_u_x=lambda _data: 1.0,
        compute_u_y=lambda _data: 1.0,
        compute_x=lambda _data: 1.0,
        compute_y=lambda _data: 1.0,
    )
    assert len(graph.nodes) == 7
    assert len(graph.edges) == 9

    g = replace_node(
        graph,
        "x",
        ContinuousRandomVariableNode(
            label="x_updated",
            parents=["z", "l"],
        ),
    )
    updated_graph = replace_node(
        g,
        "u_x",
        ContinuousRandomVariableNode(
            label="u_x_updated",
            parents=["x_updated", "c", "l"],
        ),
    )

    original_edges = {(e[0].label, e[1].label) for e in graph.edges}
    assert original_edges == {
        ("l", "u_x"),
        ("c", "u_y"),
        ("c", "u_x"),
        ("u_x", "u_y"),
        ("u_x", "x"),
        ("l", "x"),
        ("z", "x"),
        ("u_y", "y"),
        ("x", "y"),
    }

    assert len(updated_graph.nodes) == 7
    assert len(updated_graph.edges) == 9
    edges = {(e[0].label, e[1].label) for e in updated_graph.edges}
    assert edges == {
        ("l", "u_x_updated"),
        ("c", "u_y"),
        ("c", "u_x_updated"),
        ("u_x_updated", "u_y"),
        ("x_updated", "u_x_updated"),
        ("l", "x_updated"),
        ("z", "x_updated"),
        ("u_y", "y"),
        ("x_updated", "y"),
    }
