"""Graph storage."""

import networkx as nx

from causalprog._abc.labelled import Labelled

from .node import Node


class Graph(Labelled):
    """A directed acyclic graph that represents a causality tree."""

    def __init__(self, graph: nx.Graph, label: str) -> None:
        """Initialise a graph from a NetworkX graph."""
        super().__init__(label=label)

        for node in graph.nodes:
            if not isinstance(node, Node):
                msg = f"Invalid node: {node}"
                raise TypeError(msg)

        self._graph = graph.copy()
        self._nodes = list(graph.nodes())
        self._depth_first_nodes = list(nx.algorithms.dfs_postorder_nodes(graph))

        outcomes = [node for node in self._nodes if node.is_outcome]
        if len(outcomes) == 0:
            msg = "Cannot create graph with no outcome nodes"
            raise ValueError(msg)
        if len(outcomes) > 1:
            msg = "Cannot yet create graph with multiple outcome nodes"
            raise ValueError(msg)
        self._outcome = outcomes[0]
