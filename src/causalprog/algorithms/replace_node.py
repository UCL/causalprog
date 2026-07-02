"""Algorithm for replacing a graph node."""

from causalprog.graph import Graph, Node


def replace_node(
    graph: Graph, node_label: str, node: Node, label: str | None = None
) -> Graph:
    """
    Replace a node in a graph.

    Args:
        graph: The graph to replace a node in.
        node_label: The label of the node to be replaced.
        node: The new node to be inserted.
        label: The label of the new graph.

    Returns:
        A copy of the graph with the replacement made.

    """
    if label is None:
        label = f"{graph.label}_updated"
    g = graph.copy()

    g.add_node(node)
    for start, end in g.edges:
        if start.label == node_label:
            end.replace_parent(node_label, node.label)
            g.remove_edge(start, end)
            g.add_edge(node.label, end)
        elif end.label == node_label:
            g.remove_edge(start, end)
    g.remove_node(node_label)

    return g
