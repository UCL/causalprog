"""Algorithms for applying do to a graph."""

from causalprog.graph import Graph, Node
from copy import deepcopy


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
    g = deepcopy(old_g)

    g.remove_node(graph.get_node(node))

    new_nodes: dict[str, Node] = {}
    for n in old_g.nodes:
        new_n = None
        for i, j in n.parameters.items():
            if j == node:
                if new_n is None:
                    new_n = deepcopy(n)
                new_n.constant_parameters[i] = value
                del new_n.parameters[i]
        if new_n is not None:
            g.add_node(new_n)

    for e in old_g.edges:
        if e[0].label in new_nodes or e[1].label in new_nodes:
            g.add_edge(new_nodes.get(e[0].label, e[0]), new_nodes.get(e[1].label, e[1]))
    for n in new_nodes:
        g.remove_node(graph.get_node(n))

    return Graph(label=label, graph=g)
