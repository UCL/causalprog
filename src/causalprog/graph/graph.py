"""Graph storage."""

import networkx as nx
from .node import Node


class Graph(object):
    """A directed acyclic graph that represents a causality tree."""

    def __init__(self, graph: nx.Graph):
        """Initialise a graph from a NetworkX graph."""
        for node in graph.nodes:
            if not isinstance(node, Node):
                raise ValueError(f"Invalid node: {node}")

        self._graph = graph
        self._nodes = list(graph.nodes())
        self._depth_first_nodes = list(nx.algorithms.dfs_postorder_nodes(graph))

        self._predecessors = nx.algorithms.dfs_predecessors(graph)
        self._successors = nx.algorithms.dfs_successors(graph)

        for n in self._nodes:
            if n not in self._predecessors:
                self._predecessors[n] = []
            if n not in self._successors:
                self._successors[n] = []
