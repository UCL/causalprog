"""Tests for ordering of nodes in a graph."""

from causalprog.graph import Graph, ParameterNode


def test_roots_down_to_outcome() -> None:
    graph = Graph(label="G0")

    graph.add_node(ParameterNode(label="U"))
    graph.add_node(ParameterNode(label="V"))
    graph.add_node(ParameterNode(label="W"))
    graph.add_node(ParameterNode(label="X"))
    graph.add_node(ParameterNode(label="Y"))
    graph.add_node(ParameterNode(label="Z"))

    edges = [
        ["V", "W"],
        ["V", "X"],
        ["V", "Y"],
        ["X", "Z"],
        ["Y", "Z"],
        ["U", "Z"],
    ]
    for e in edges:
        graph.add_edge(*e)

    assert graph.roots_down_to_outcome("V") == (graph.get_node("V"),)
    assert graph.roots_down_to_outcome("W") == (
        graph.get_node("V"),
        graph.get_node("W"),
    )
    nodes = graph.roots_down_to_outcome("Z")
    assert len(nodes) == 5  # noqa: PLR2004
    for e in edges:
        if "W" not in e:
            assert nodes.index(graph.get_node(e[0])) < nodes.index(graph.get_node(e[1]))
