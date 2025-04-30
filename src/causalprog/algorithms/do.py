"""Algorithms for applying do to a graph."""

import numpy.typing as npt

from causalprog.graph import Graph, ParameterNode


def do(graph: Graph, node: str, value: float, label: str | None = None) -> Graph:
    if label is None:
        label = f"{graph.label}|do({node}={value})"

    g = graph._graph.copy()

    new_node = ParameterNode(node, value=value)
    g.add_node(new_node)

    edges_to_remove = []
    for e in graph._graph.edges:
        if e[0].label == node:
            g.add_edge(new_node, e[1])
            g.remove_edge(*e)
        if e[1].label == node:
            g.add_edge(e[0], new_node)
            g.remove_edge(*e)

    g.remove_node(graph.get_node(node))


    from IPython import embed; embed()

    return Graph(label, g)
