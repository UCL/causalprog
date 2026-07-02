"""Algorithm for replacing a graph node."""

from causalprog.graph import Graph, Node


def replace_node(
    graph: Graph,
    node_label_to_replace: str,
    replacement_node: Node,
    new_graph_label: str | None = None,
) -> Graph:
    """
    Replace a node in a graph.

    Args:
        graph: The graph to replace a node in.
        node_label_to_replace: The label of the node to be replaced.
        replacement_node: The new node to be inserted.
        new_graph_label: The label of the new graph.

    Returns:
        A copy of the graph with the replacement made.

    """
    g = graph.copy(label=new_graph_label)

    g.add_node(replacement_node)
    for start, end in g.edges:
        if start.label == node_label_to_replace:
            end.replace_parent(node_label_to_replace, replacement_node.label)
            g.remove_edge(start, end)
            g.add_edge(replacement_node.label, end)
        elif end.label == node_label_to_replace:
            g.remove_edge(start, end)
    g.remove_node(node_label_to_replace)

    return g
