"""Graph storage."""

import networkx as nx

from .node import Node


class Graph:
    """A directed acyclic graph that represents a causality tree."""

    def __init__(self, graph: nx.Graph, label: str) -> None:
        """Initialise a graph from a NetworkX graph."""
        for node in graph.nodes:
            if not isinstance(node, Node):
                msg = f"Invalid node: {node}"
                raise TypeError(msg)

        self._label = label
        self._graph = graph.copy()

    def get_node(self, label: str) -> Node:
        """Get a node from its label."""
        for node in self._graph.nodes():
            if node.label == label:
                return node
        msg = f'Node not found with label "{label}"'
        raise ValueError(msg)

    @property
    def predecessors(self) -> dict[Node, list[Node]]:
        """Get predecessors of every node."""
        return nx.algorithms.dfs_predecessors(self._graph)

    @property
    def successors(self) -> dict[Node, list[Node]]:
        """Get successors of every node."""
        return nx.algorithms.dfs_successors(self._graph)

    @property
    def outcome(self) -> Node:
        """The outcome node of the graph."""
        outcomes = [node for node in self.nodes if node.is_outcome]
        if len(outcomes) == 0:
            msg = "Cannot create graph with no outcome nodes"
            raise ValueError(msg)
        if len(outcomes) > 1:
            msg = "Cannot yet create graph with multiple outcome nodes"
            raise ValueError(msg)
        return outcomes[0]

    @property
    def nodes(self) -> list[Node]:
        """The nodes of the graph."""
        return list(self._graph.nodes())

    @property
    def depth_first_nodes(self) -> list[Node]:
        """The nodes of the graph in depth first order."""
        return list(nx.algorithms.dfs_postorder_nodes(self._graph))

    @property
    def label(self) -> str:
        """The label of the graph."""
        return self._label
