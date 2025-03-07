"""Graph storage."""

import networkx as nx
from .node import Node

g_index = 0


class Graph(object):
    """A directed acyclic graph that represents a causality tree."""

    def __init__(self, graph: nx.Graph, label: str | None = None):
        """Initialise a graph from a NetworkX graph."""
        global g_index
        for node in graph.nodes:
            if not isinstance(node, Node):
                raise ValueError(f"Invalid node: {node}")

        if label is None:
            self._label = f"Graph{g_index}"
            g_index += 1
        else:
            self.label = label

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

        outcomes = [node for node in self._nodes if node.is_outcome]
        if len(outcomes) == 0:
            raise ValueError("Cannot create graph with no outcome nodes")
        if len(outcomes) > 1:
            raise ValueError("Cannot yes create graph with multiple outcome nodes")
        self._outcome = outcomes[0]

    @property
    def label(self) -> str:
        """The label of the graph."""
        return self._label
