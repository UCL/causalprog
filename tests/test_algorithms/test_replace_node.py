"""Tests for evaluate algorithms."""

from collections.abc import Callable

import pytest

from causalprog.algorithms import replace_node
from causalprog.graph import ContinuousRandomVariableNode, Graph


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


@pytest.mark.parametrize(
    ("old_node_label", "new_node", "expected_nodes", "expected_edges"),
    [
        pytest.param(
            "G",
            ContinuousRandomVariableNode(label="H"),
            {"A", "B", "C", "D", "E", "F", "H"},
            {("A", "B"), ("A", "E"), ("B", "E"), ("C", "E"), ("B", "F")},
            id="Replace node",
        ),
        pytest.param(
            "A",
            ContinuousRandomVariableNode(label="H"),
            {"B", "C", "D", "E", "F", "G", "H"},
            {("H", "B"), ("H", "E"), ("B", "E"), ("C", "E"), ("B", "F")},
            id="Replace node with children",
        ),
        pytest.param(
            "E",
            ContinuousRandomVariableNode(label="H", parents=["A", "B", "C"]),
            {"A", "B", "C", "D", "F", "G", "H"},
            {("A", "B"), ("A", "H"), ("B", "H"), ("C", "H"), ("B", "F")},
            id="Replace node with parents",
        ),
        pytest.param(
            "E",
            ContinuousRandomVariableNode(label="H"),
            {"A", "B", "C", "D", "F", "G", "H"},
            {("A", "B"), ("B", "F")},
            id="Removing parents",
        ),
        pytest.param(
            "G",
            ContinuousRandomVariableNode(label="H", parents=["A", "B"]),
            {"A", "B", "C", "D", "E", "F", "H"},
            {
                ("A", "B"),
                ("A", "E"),
                ("B", "E"),
                ("C", "E"),
                ("B", "F"),
                ("A", "H"),
                ("B", "H"),
            },
            id="Adding parents",
        ),
        pytest.param(
            "G",
            ContinuousRandomVariableNode(label="G"),
            {"A", "B", "C", "D", "E", "F", "G"},
            {("A", "B"), ("A", "E"), ("B", "E"), ("C", "E"), ("B", "F")},
            id="Replace node with same label",
        ),
    ],
)
def test_replace_node(
    small_graph, old_node_label, new_node, expected_nodes, expected_edges
):
    graph = small_graph()
    updated_graph = replace_node(graph, old_node_label, new_node)

    # Check that graph is unchanged
    assert {n.label for n in graph.nodes} == {"A", "B", "C", "D", "E", "F", "G"}
    assert {(e[0].label, e[1].label) for e in graph.edges} == {
        ("A", "B"),
        ("A", "E"),
        ("B", "E"),
        ("C", "E"),
        ("B", "F"),
    }

    # Check that algorithm has expected result
    assert {n.label for n in updated_graph.nodes} == expected_nodes
    assert {(e[0].label, e[1].label) for e in updated_graph.edges} == expected_edges


@pytest.mark.parametrize(
    ("old_node_label", "new_node", "error"),
    [
        pytest.param(
            "G",
            ContinuousRandomVariableNode(label="A"),
            ValueError("Duplicate node label"),
            id="Duplicated label",
        ),
        pytest.param(
            "G",
            ContinuousRandomVariableNode(label="H", parents=["G"]),
            ValueError("Node being replace cannot be parent of replacement node"),
            id="Cannot use node being replaced as parent",
        ),
        pytest.param(
            "A",
            ContinuousRandomVariableNode(label="A", parents=["B"]),
            ValueError("Replacement would create a cycle"),
            id="Adding a cycle",
        ),
    ],
)
def test_replace_node_error(
    raises_context, small_graph, old_node_label, new_node, error
):
    graph = small_graph()
    with raises_context(error):
        replace_node(graph, old_node_label, new_node)
