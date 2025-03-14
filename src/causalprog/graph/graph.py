"""Graph storage."""

import networkx as nx

from .node import Node


class Graph:
    """A directed acyclic graph that represents a causality tree."""

    _nodes_by_label: dict[str, Node]

    def __init__(self, label: str) -> None:
        """Create end empty graph."""
        self._label = label
        self._graph = nx.DiGraph()
        self._nodes_by_label = {}
        self._node_index = 0

    def get_node(self, label: str) -> Node:
        """Get a node from its label."""
        try:
            return self._nodes_by_label[label]
        except KeyError as e:
            msg = f'Node not found with label "{label}"'
            raise ValueError(msg) from e

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        if node._label is None:  # noqa: SLF001
            while f"node{self._node_index}" in self._nodes_by_label:
                self._node_index += 1
            node._label = f"node{self._node_index}"  # noqa: SLF001
        if node.label in self._nodes_by_label:
            msg = f"Duplicate node label: {node.label}"
            raise ValueError(msg)
        self._nodes_by_label[node.label] = node
        self._graph.add_node(node)

    def add_edge(self, first_node: Node | str, second_node: Node | str) -> None:
        """Add an edge to the graph."""
        if isinstance(first_node, str):
            first_node = self.get_node(first_node)
        if isinstance(second_node, str):
            second_node = self.get_node(second_node)
        if first_node.label not in self._nodes_by_label:
            self.add_node(first_node)
        if second_node.label not in self._nodes_by_label:
            self.add_node(second_node)
        if first_node != self._nodes_by_label[first_node.label]:
            msg = "Invalid node"
            raise ValueError(msg)
        if second_node != self._nodes_by_label[second_node.label]:
            msg = "Invalid node"
            raise ValueError(msg)
        self._graph.add_edge(first_node, second_node)

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
