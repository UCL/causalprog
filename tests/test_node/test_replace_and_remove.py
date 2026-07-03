from typing import Callable
import pytest

from causalprog.graph import (
    ContinuousRandomVariableNode,
    Graph,
)

@pytest.fixture
def small_graph() -> Callable[[], Graph]:
    def _inner() -> Graph:
        graph = Graph(label="G")
        graph.add_node(ContinuousRandomVariableNode(label="A"))
        graph.add_node(ContinuousRandomVariableNode(label="B", parents=["A"]))
        graph.add_node(ContinuousRandomVariableNode(label="C"))
        graph.add_node(ContinuousRandomVariableNode(label="D"))
        graph.add_node(ContinuousRandomVariableNode(label="E", parents=["A", "B", "C"]))
        graph.add_node(ContinuousRandomVariableNode(label="F", parents=["B"]))
        graph.add_node(ContinuousRandomVariableNode(label="G"))
        return graph

    return _inner


@pytest.mark.parametrize(("node_label", "old_node_label", "new_node_label", "expected_parents"), [
    pytest.param("B", "A", "C", {"C"}, id="Replace parent"),
    pytest.param("E", "A", "D", {"D", "B", "C"}, id="Replace one of many parents"),
])
def test_replace_parent(small_graph, node_label, old_node_label, new_node_label, expected_parents):
    graph = small_graph()
    node = graph.get_node(node_label)
    node.replace_parent(old_node_label, new_node_label)
    assert set(node.parents) == expected_parents


@pytest.mark.parametrize(("node_label", "old_node_label", "new_node_label", "error"), [
    pytest.param("A", "B", "C", ValueError("B is not a parent of A"), id="Attempt to replace non-parent"),
    pytest.param("B", "A", "B", ValueError("Node cannot be its own parent"), id="Node cannot be its own parent"),
])
def test_replace_parent_error(
    raises_context, small_graph, error, node_label, old_node_label, new_node_label,
):
    graph = small_graph()
    with raises_context(error):
        graph.get_node(node_label).replace_parent(old_node_label, new_node_label)


@pytest.mark.parametrize(("label_to_remove", "expected_nodes"), [
    pytest.param("D", {"A", "B", "C", "E", "F", "G"}, id="Remove D"),
    pytest.param("G", {"A", "B", "C", "D", "E", "F"}, id="Remove G"),
])
def test_remove_node(
    small_graph, label_to_remove, expected_nodes
):
    graph = small_graph()
    graph.remove_node(label_to_remove)
    assert {node.label for node in graph.nodes} == expected_nodes


@pytest.mark.parametrize(("label_to_remove", "error"), [
    pytest.param("B", ValueError("Cannot remove node"), id="Cannot remove node with parent"),
    pytest.param("E", ValueError("Cannot remove node"), id="Cannot remove node with multiple parents"),
    pytest.param("A", ValueError("Cannot remove node"), id="Cannot remove node with children"),
])
def test_remove_node_error(
    raises_context, small_graph, label_to_remove, error
):
    graph = small_graph()
    with raises_context(error):
        graph.remove_node(label_to_remove)


@pytest.mark.parametrize(("start_node", "end_node", "expected_edges"), [
    pytest.param("A", "B", {("A", "E"), ("B", "E"), ("C", "E"), ("B", "F")}, id="Remove A->B"),
    pytest.param("A", "E", {("A", "B"), ("B", "E"), ("C", "E"), ("B", "F")}, id="Remove A->E"),
    pytest.param("B", "E", {("A", "B"), ("A", "E"), ("C", "E"), ("B", "F")}, id="Remove B->E"),
    pytest.param("B", "F", {("A", "B"), ("A", "E"), ("B", "E"), ("C", "E")}, id="Remove B->F"),
])
def test_remove_edge(
    small_graph, start_node, end_node, expected_edges
):
    graph = small_graph()
    graph.remove_edge(start_node, end_node)
    assert {(e[0].label, e[1].label) for e in graph.edges} == expected_edges


@pytest.mark.parametrize(("start_node", "end_node", "error"), [
    pytest.param("G", "C", ValueError("Cannot remove edge"), id="Cannot remove non-existent edge"),
    pytest.param("B", "A", ValueError("Cannot remove edge"), id="Cannot remove reversed edge"),
])
def test_remove_edge_error(
    raises_context, small_graph, start_node, end_node, error
):
    graph = small_graph()
    with raises_context(error):
        graph.remove_edge(start_node, end_node)


"""
    pytest.param("A", "B", ContinuousRandomVariableNode(label="C"), ValueError("B is not a parent of A"), id="Attempt to replace non-parent"),
    pytest.param("B", "A", ContinuousRandomVariableNode(label="C"), ValueError("nopw"), id="Attempt to replace with node with duplicate label"),
    pytest.param("B", "A", ContinuousRandomVariableNode(label="D", parents=["A"]), ValueError("nopw"), id="Attempt to replace with node with that has removed node as parent"),
"""
