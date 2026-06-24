"""Tests for the do algorithm."""

from causalprog import algorithms
from causalprog.graph import DataNode, Graph, ConstantNode

max_samples = 10**5


def test_do(two_normal_graph):
    graph = two_normal_graph(5.0, 1.2, 0.8)
    graph2 = algorithms.do(graph, "UX", 4.0)

    assert "loc" in graph.get_node("X").parents
    assert "loc" in graph2.get_node("X").parents

    graph.get_node("UX")
    assert isinstance(graph2.get_node("UX"), ConstantNode)


def test_do_removes_dependencies(two_normal_graph, raises_context):
    graph = two_normal_graph()
    graph2 = algorithms.do(graph, "UX", 4.0)

    graph.get_node("UX")
    assert isinstance(graph2.get_node("UX"), ConstantNode)
    for node in ["mean", "cov"]:
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
        ("mean", "UX"),
        ("cov", "UX"),
    ]:
        assert e in edges
        assert e not in edges2

    # Check that correct edges remain
    for e in [
        ("UX", "X"),
        ("cov2", "X"),
    ]:
        assert e in edges
        assert e in edges2


def test_do_error(raises_context):
    graph = Graph(label="ABC")
    graph.add_node(DataNode(label="A"))
    graph.add_node(DataNode(label="B1"))
    graph.add_node(DataNode(label="B2"))
    graph.add_node(DataNode(label="C"))
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
