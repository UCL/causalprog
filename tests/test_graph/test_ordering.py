"""Tests for ordering of nodes in a graph."""

from causalprog.graph import DataNode, DistributionNode, Graph


def test_roots_down_to_outcome() -> None:
    graph = Graph(label="G0")

    graph.add_node(DataNode(label="U"))
    graph.add_node(DataNode(label="V"))
    graph.add_node(DataNode(label="W"))
    graph.add_node(DataNode(label="X"))
    graph.add_node(DataNode(label="Y"))
    graph.add_node(DataNode(label="Z"))

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
    assert len(nodes) == 5
    for e in edges:
        if "W" not in e:
            assert nodes.index(graph.get_node(e[0])) < nodes.index(graph.get_node(e[1]))


def test_roots_and_leaves() -> None:
    """Test roots and leaves for the following graph.

         B         I
         |         ^
         v         |
    A -> C -> D    H
              ^    |
              |    v
         E -> F -> G
    """
    graph = Graph(label="G0")
    graph.add_node(DataNode(label="A"))
    graph.add_node(DataNode(label="B"))
    graph.add_node(DistributionNode(None, label="C", parameters={"A": "A", "B": "B"}))
    graph.add_node(DataNode(label="E"))
    graph.add_node(DistributionNode(None, label="F", parameters={"E": "E"}))
    graph.add_node(DistributionNode(None, label="D", parameters={"C": "C", "F": "F"}))
    graph.add_node(DataNode(label="H"))
    graph.add_node(DistributionNode(None, label="G", parameters={"F": "F", "H": "H"}))
    graph.add_node(DistributionNode(None, label="I", parameters={"H": "H"}))

    root_nodes = {node.label for node in graph.root_nodes}
    assert root_nodes == {"A", "B", "E", "H"}

    leaf_nodes = {node.label for node in graph.leaf_nodes}
    assert leaf_nodes == {"D", "G", "I"}
