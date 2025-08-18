"""Tests for the do algorithm."""

from causalprog import algorithms
from causalprog.graph import Graph, ParameterNode

max_samples = 10**5


def test_do(two_normal_graph, raises_context):
    graph = two_normal_graph(5.0, 1.2, 0.8)
    graph2 = algorithms.do(graph, "UX", 4.0)

    assert "loc" in graph.get_node("X").parameters
    assert "loc" not in graph.get_node("X").constant_parameters
    assert "loc" not in graph2.get_node("X").parameters
    assert "loc" in graph2.get_node("X").constant_parameters

    graph.get_node("UX")
    with raises_context(KeyError('Node not found with label "UX"')):
        graph2.get_node("UX")


def test_do_removes_dependencies(two_normal_graph, raises_context):
    graph = two_normal_graph()
    graph2 = algorithms.do(graph, "UX", 4.0)

    for node in ["UX", "mean", "cov"]:
        graph.get_node(node)
        with raises_context(KeyError(f'Node not found with label "{node}"')):
            graph2.get_node(node)


def test_do_edges(two_normal_graph):
    graph = two_normal_graph()
    graph2 = algorithms.do(graph, "UX", 4.0)

    edges = [(e[0].label, e[1].label) for e in graph.edges]
    edges2 = [(e[0].label, e[1].label) for e in graph2.edges]

    # Check that correct edges are removed
    for e in [
        ("UX", "X"),
        ("mean", "UX"),
        ("cov", "UX"),
    ]:
        assert e in edges
        assert e not in edges2

    # Check that correct edges remain
    for e in [
        ("cov2", "X"),
    ]:
        assert e in edges
        assert e in edges2


def test_do_error(raises_context):
    graph = Graph(label="ABC")
    graph.add_node(ParameterNode(label="A"))
    graph.add_node(ParameterNode(label="B1"))
    graph.add_node(ParameterNode(label="B2"))
    graph.add_node(ParameterNode(label="C"))
    graph.add_edge("A", "B1")
    graph.add_edge("A", "B2")
    graph.add_edge("B1", "C")
    graph.add_edge("B2", "C")

    # Currently, applying to do to a node that had predecessors that cannot be removed
    # raises an error, see https://github.com/UCL/causalprog/issues/80
    with raises_context(
        ValueError(
            "Node that is predecessor of node set by do and nodes that are not removed "
            "found"
        )
    ):
        algorithms.do(graph, "B1", 1.0)
