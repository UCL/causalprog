from causalprog.graph.ricardo import example_model


def test_example_model():
    graph = example_model(
        compute_u_x=lambda _data: 1.0,
        compute_u_y=lambda _data: 1.0,
        compute_phi_x=lambda _data: 1.0,
        compute_x=lambda _data: 1.0,
        compute_y=lambda _data: 1.0,
    )
    assert len(graph.nodes) == 8
    assert len(graph.edges) == 8
    edges = {(e[0].label, e[1].label) for e in graph.edges}
    assert edges == {
        ("l", "phi_x"),
        ("c", "u_y"),
        ("c", "u_x"),
        ("u_x", "x"),
        ("phi_x", "x"),
        ("z", "x"),
        ("u_y", "y"),
        ("x", "y"),
    }
