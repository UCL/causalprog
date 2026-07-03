"""Graph storage."""

from __future__ import annotations

import typing

import networkx as nx

if typing.TYPE_CHECKING:
    import numpy.typing as npt

from causalprog._abc.labelled import Labelled
from causalprog.graph.node import DataNode, DistributionNode, Node


class Graph(Labelled):
    """A directed acyclic graph that represents a causality tree."""

    _nodes_by_label: dict[str, Node]

    def __init__(self, *, label: str, graph: nx.DiGraph | None = None) -> None:
        """
        Create a graph.

        Args:
            label: A label to identify the graph
            graph: A networkx graph to base this graph on

        """
        super().__init__(label=label)
        self._nodes_by_label = {}
        if graph is None:
            graph = nx.DiGraph()

        self._graph = graph
        for node in graph.nodes:
            self._nodes_by_label[node.label] = node

    def copy(self, *, label: str | None = None) -> Graph:
        """Create a copy of a graph."""
        if label is None:
            label = f"{self.label}_copy"
        return Graph(label=label, graph=self._graph.copy())

    def get_node(self, label: str) -> Node:
        """
        Get a node from its label.

        Args:
            label: The label

        Returns:
            The node

        """
        node = self._nodes_by_label.get(label, None)
        if not node:
            msg = f'Node not found with label "{label}"'
            raise KeyError(msg)
        return node

    def add_node(self, node: Node) -> None:
        """
        Add a node to the graph.

        Args:
            node: The node to add

        """
        if node.label in self._nodes_by_label:
            msg = f"Duplicate node label: {node.label}"
            raise ValueError(msg)
        self._nodes_by_label[node.label] = node
        self._graph.add_node(node)
        for p in node.parents:
            self.add_edge(p, node.label)

    def remove_node(self, node: str | Node) -> None:
        """
        Remove a node from the graph.

        Args:
            node: The node to remove

        """
        if isinstance(node, str):
            node = self.get_node(node)
        if node.label not in self._nodes_by_label:
            msg = f"Cannot remove node: {node.label}"
            raise ValueError(msg)
        for start, end in self.edges:
            if node.label in {start.label, end.label}:
                msg = "Cannot remove node that is attached to edges"
                raise ValueError(msg)
        del self._nodes_by_label[node.label]
        self._graph.remove_node(node)

    def add_edge(self, start_node: Node | str, end_node: Node | str) -> None:
        """
        Add a directed edge to the graph.

        Adding an edge between nodes not currently in the graph,
        will cause said nodes to be added to the graph along with
        the edge.

        Args:
            start_node: The node that the edge points from
            end_node: The node that the edge points to

        """
        if isinstance(start_node, str):
            start_node = self.get_node(start_node)
        if isinstance(end_node, str):
            end_node = self.get_node(end_node)
        if start_node.label not in self._nodes_by_label:
            self.add_node(start_node)
        if end_node.label not in self._nodes_by_label:
            self.add_node(end_node)
        for node_to_check in (start_node, end_node):
            if node_to_check != self._nodes_by_label[node_to_check.label]:
                msg = "Invalid node: {node_to_check}"
                raise ValueError(msg)
        self._graph.add_edge(start_node, end_node)

    def remove_edge(self, start_node: Node | str, end_node: Node | str) -> None:
        """
        Remove a directed edge from the graph.

        Args:
            start_node: The node that the edge points from
            end_node: The node that the edge points to

        """
        if isinstance(start_node, str):
            start_node = self.get_node(start_node)
        if isinstance(end_node, str):
            end_node = self.get_node(end_node)
        try:
            self._graph.remove_edge(start_node, end_node)
        except nx.exception.NetworkXError as err:
            msg = "Cannot remove edge that is not in graph"
            raise ValueError(msg) from err

    @property
    def root_nodes(self) -> tuple[Node, ...]:
        """
        Returns all root nodes in the graph.

        Root nodes are nodes with no parents.

        The returned tuple uses the `ordered_nodes` property to obtain the root
        nodes so that a natural "fixed order" is given to the roots. When root
        values are given as inputs to the causal estimand and / or constraint functions,
        they will ideally be given as a single vector of root values, in which case
        a fixed ordering for the leaves is necessary to make an association to the
        components of the given input vector.

        Returns:
            Root nodes

        """
        return tuple(node for node in self.ordered_nodes if len(node.parents) == 0)

    @property
    def leaf_nodes(self) -> tuple[Node, ...]:
        """
        Returns all leaf nodes in the graph.

        Root nodes are nodes with no children.

        The returned tuple uses the `ordered_nodes` property to obtain the leaf
        nodes so that a natural "fixed order" is given to the leaves.

        Returns:
            Leaf nodes

        """
        labels = [node.label for node in self.ordered_nodes]
        for node in self.nodes:
            for p in node.parents:
                if p in labels:
                    labels.remove(p)
        return tuple(self.get_node(label) for label in labels)

    @property
    def predecessors(self) -> dict[Node, tuple[Node, ...]]:
        """
        Get predecessors of every node.

        Returns:
            Mapping of each Node to its predecessor Nodes

        """
        return {node: tuple(self._graph.predecessors(node)) for node in self.nodes}

    @property
    def successors(self) -> dict[Node, tuple[Node, ...]]:
        """
        Get successors of every node.

        Returns:
            Mapping of each Node to its successor Nodes.

        """
        return {node: tuple(self._graph.successors(node)) for node in self.nodes}

    @property
    def nodes(self) -> tuple[Node, ...]:
        """
        Get the nodes of the graph, with no enforced ordering.

        Returns:
            A list of all the nodes in the graph.

        See Also:
            ordered_nodes: Fetch an ordered list of the nodes in the graph.

        """
        return tuple(self._graph.nodes())

    @property
    def edges(self) -> tuple[tuple[Node, Node], ...]:
        """
        Get the edges of the graph.

        Returns:
            A tuple of all the edges in the graph.

        """
        return tuple(self._graph.edges())

    @property
    def ordered_nodes(self) -> tuple[Node, ...]:
        """
        Nodes ordered so that each node appears after its dependencies.

        Returns:
            A list of all the nodes, ordered such that each node
                appears after all its dependencies.

        """
        if not nx.is_directed_acyclic_graph(self._graph):
            msg = "Graph is not acyclic."
            raise RuntimeError(msg)
        return tuple(nx.topological_sort(self._graph))

    def roots_down_to_outcome(
        self,
        outcome_node_label: str,
    ) -> tuple[Node, ...]:
        """
        Get ordered list of nodes that outcome depends on.

        Nodes are ordered so that each node appears after its dependencies.

        Args:
            outcome_node_label: The label of the outcome node

        Returns:
            A list of the nodes, ordered from root nodes to the outcome Node.

        """
        outcome = self.get_node(outcome_node_label)
        ancestors = nx.ancestors(self._graph, outcome)
        return tuple(
            node for node in self.ordered_nodes if node == outcome or node in ancestors
        )

    def model(self, **parameter_values: npt.ArrayLike) -> dict[str, npt.ArrayLike]:
        """
        Model corresponding to the `Graph`'s structure.

        The model created takes values of the nodes that are parameter as keyword
        arguments. Names of the keyword arguments should match the labels of the
        `DataNode`s, and their values should be the values of those parameters.

        The method returns a dictionary recording the mode sites that are created.
        This means that the model can be 'extended' further by defining additional
        sites in a wrapper around this method.

        Args:
            parameter_values: Names of the keyword arguments should match the labels
                of the `DataNode`s, and their values should be the values of those
                parameters.

        Returns:
            Mapping of non-`DataNode` `Node` labels to the site objects created
                for these nodes.

        """
        # Confirm that all `DataNode`s have been assigned a value.
        for node in self.nodes:
            if isinstance(node, DataNode) and node.label not in parameter_values:
                msg = f"DataNode '{node.label}' not assigned"
                raise KeyError(msg)

        # Build model sequentially, using the node_order to inform the
        # construction process.
        node_record: dict[str, npt.ArrayLike] = {}
        for node in self.ordered_nodes:
            if isinstance(node, DistributionNode):
                node_record[node.label] = node.create_model_site(
                    **parameter_values,  # All nodes require knowledge of the parameters
                    **node_record,  # and any dependent nodes we have already visited
                )

        return node_record

    def is_dag(self) -> bool:
        """Check if this graph is a directed acyclic graph."""
        return nx.is_directed_acyclic_graph(self._graph)
