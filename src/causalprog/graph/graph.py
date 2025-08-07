"""Graph storage."""

import networkx as nx

from causalprog._abc.labelled import Labelled

from .node import Node, ParameterNode


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

    def add_edge(self, first_node: Node | str, second_node: Node | str) -> None:
        """
        Add a directed edge to the graph.

        Adding an edge between nodes not currently in the graph,
        will cause said nodes to be added to the graph along with
        the edge.

        Args:
            first_node: The node that the edge points from
            second_node: The node that the edge points to

        """
        if isinstance(first_node, str):
            first_node = self.get_node(first_node)
        if isinstance(second_node, str):
            second_node = self.get_node(second_node)
        if first_node.label not in self._nodes_by_label:
            self.add_node(first_node)
        if second_node.label not in self._nodes_by_label:
            self.add_node(second_node)
        for node_to_check in (first_node, second_node):
            if node_to_check != self._nodes_by_label[node_to_check.label]:
                msg = "Invalid node: {node_to_check}"
                raise ValueError(msg)
        self._graph.add_edge(first_node, second_node)

    def set_parameters(self, **parameter_values: float | None) -> None:
        """
        Set the current value of all given parameter nodes to the new values.

        Parameter nodes are identified by variable name. Absent parameters retain their
        current value. Names that correspond to nodes which are not parameter nodes
        raise `TypeError`s.

        Args:
            parameter_values: The parameters and values to set them to

        """
        for name, new_value in parameter_values.items():
            node = self.get_node(name)
            if not isinstance(node, ParameterNode):
                msg = f"Node {name} is not a parameter node."
                raise TypeError(msg)
            node.value = new_value

    @property
    def parameter_nodes(self) -> tuple[ParameterNode, ...]:
        """
        Returns all parameter nodes in the graph.

        The returned tuple uses the `ordered_nodes` property to obtain the parameter
        nodes so that a natural "fixed order" is given to the parameters. When parameter
        values are given as inputs to the causal estimand and / or constraint functions,
        they will ideally be given as a single vector of parameter values, in which case
        a fixed ordering for the parameters is necessary to make an association to the
        components of the given input vector.

        Returns:
            Parameter nodes

        """
        return tuple(node for node in self.ordered_nodes if node.is_parameter)

    @property
    def predecessors(self) -> dict[Node, list[Node]]:
        """
        Get predecessors of every node.

        Returns:
            Mapping of each Node to its predecessor Nodes

        """
        return {node: list(self._graph.predecessors(node)) for node in self.nodes}

    @property
    def successors(self) -> dict[Node, list[Node]]:
        """
        Get successors of every node.

        Returns:
            Mapping of each Node to its successor Nodes.

        """
        return {node: list(self._graph.successors(node)) for node in self.nodes}

    @property
    def outcome(self) -> Node:
        """
        The outcome node of the graph.

        Will raise a ValueError if there is not exactly one outcome node.

        Returns:
            Outcome node

        """
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
        """
        The nodes of the graph. Ordering is not enforced in any particular manner.

        Returns:
            A list of all the nodes in the graph.
        
        See also:
            ordered_nodes: Fetch an ordered list of the nodes in the graph.

        """
        return list(self._graph.nodes())

    @property
    def ordered_nodes(self) -> list[Node]:
        """
        Nodes ordered so that each node appears after its dependencies.

        Returns:
            A list of all the nodes

        """
        if not nx.is_directed_acyclic_graph(self._graph):
            msg = "Graph is not acyclic."
            raise RuntimeError(msg)
        return list(nx.topological_sort(self._graph))

    def roots_down_to_outcome(
        self,
        outcome_node_label: str,
    ) -> list[Node]:
        """
        Get ordered list of nodes that outcome depends on.

        Nodes are ordered so that each node appears after its dependencies.

        Args:
            outcome_node_label: The label of the outcome node

        Returns:
            A list of the nodes

        """
        outcome = self.get_node(outcome_node_label)
        ancestors = nx.ancestors(self._graph, outcome)
        return [
            node for node in self.ordered_nodes if node == outcome or node in ancestors
        ]
