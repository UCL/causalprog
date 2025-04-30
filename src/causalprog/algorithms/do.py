"""Algorithms for applying do to a graph."""

import numpy.typing as npt

from causalprog.graph import Graph


def do(graph: Graph, node: str, value: float, label: str | None = None) -> Graph:
    if label is None:
        label = f"{graph.label}|do({node}={value})"

    g = graph._graph.copy()
    return Graph(label, g)

    from IPython import embed; embed()

    return graph.copy()
