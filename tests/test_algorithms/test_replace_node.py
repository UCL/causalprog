"""Tests for evaluate algorithms."""

import pytest

from causalprog.algorithms import replace_node
from causalprog.graph import ContinuousRandomVariableNode


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
    seven_node_graph, old_node_label, new_node, expected_nodes, expected_edges
):
    graph = seven_node_graph()
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
            ValueError("Node being replaced cannot be parent of replacement node"),
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
    raises_context, seven_node_graph, old_node_label, new_node, error
):
    graph = seven_node_graph()
    with raises_context(error):
        replace_node(graph, old_node_label, new_node)
