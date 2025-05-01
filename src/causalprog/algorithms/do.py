"""Algorithms for applying do to a graph."""

from causalprog.graph import Graph, ParameterNode


def do(graph: Graph, node: str, value: float, label: str | None = None) -> Graph:
    """
    Apply do to a graph.

    Args:
        graph: The graph to apply do to. This will be copied.
        node: The label of the node to apply do to.
        value: The value to set the node to.
        label: The label of the new graph

    Returns:
        A copy of the graph with do applied

    """
    if label is None:
        label = f"{graph.label}|do({node}={value})"

    old_g = graph._graph  # noqa: SLF001
    g = old_g.copy()

    new_node = ParameterNode(node, value=value)
    g.add_node(new_node)

    for e in old_g.edges:
        if e[0].label == node:
            g.add_edge(new_node, e[1])
            g.remove_edge(*e)
        elif e[1].label == node:
            g.remove_edge(*e)

    g.remove_node(graph.get_node(node))

    return Graph(label, g)
