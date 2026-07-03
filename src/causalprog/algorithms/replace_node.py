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
    if node_label_to_replace in replacement_node.parents:
        msg = "Node being replace cannot be parent of replacement node"
        raise ValueError(msg)

    g = graph.copy(label=new_graph_label)

    new_edges = []
    for start, end in g.edges:
        if start.label == node_label_to_replace:
            end.replace_parent(node_label_to_replace, replacement_node.label)
            g.remove_edge(start, end)
            new_edges.append((replacement_node.label, end))
        elif end.label == node_label_to_replace:
            g.remove_edge(start, end)

    g.remove_node(node_label_to_replace)
    g.add_node(replacement_node)

    for start, end in new_edges:
        g.add_edge(start, end)

    if not g.is_dag():
        msg = "Replacement would create a cycle"
        raise ValueError(msg)
    return g
