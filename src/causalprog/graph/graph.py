"""Graph storage."""

from collections.abc import Callable

import networkx as nx
import numpy.typing as npt

from causalprog._abc.labelled import Labelled
from causalprog.graph.node import DistributionNode, Node, ParameterNode


class Graph(Labelled):
    """A directed acyclic graph that represents a causality tree."""

    _nodes_by_label: dict[str, Node]

    def __init__(self, label: str) -> None:
        """Create end empty graph."""
        super().__init__(label=label)
        self._graph = nx.DiGraph()
        self._nodes_by_label = {}

    def get_node(self, label: str) -> Node:
        """Get a node from its label."""
        node = self._nodes_by_label.get(label, None)
        if not node:
            msg = f'Node not found with label "{label}"'
            raise KeyError(msg)
        return node

    def add_node(self, node: Node) -> None:
        """Add a node to the graph."""
        if node.label in self._nodes_by_label:
            msg = f"Duplicate node label: {node.label}"
            raise ValueError(msg)
        self._nodes_by_label[node.label] = node
        self._graph.add_node(node)

    def add_edge(self, first_node: Node | str, second_node: Node | str) -> None:
        """
        Add an edge to the graph.

        Adding an edge between nodes not currently in the graph,
        will cause said nodes to be added to the graph along with
        the edge.
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
        """
        return tuple(node for node in self.ordered_nodes if node.is_parameter)

    @property
    def predecessors(self) -> dict[Node, Node]:
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
    def ordered_nodes(self) -> list[Node]:
        """`Node`s ordered so that each node appears after its dependencies."""
        if not nx.is_directed_acyclic_graph(self._graph):
            msg = "Graph is not acyclic."
            raise RuntimeError(msg)
        return list(nx.topological_sort(self._graph))

    @property
    def ordered_dist_nodes(self) -> list[DistributionNode]:
        """
        `DistributionNode`s in dependency order.

        Each `DistributionNode` in the returned list appears after all its
        dependencies. Order is derived from `self.ordered_nodes`, with the
        `ParameterNode`s removed.
        """
        return [node for node in self.ordered_nodes if not node.is_parameter]

    def roots_down_to_outcome(
        self,
        outcome_node_label: str,
    ) -> list[Node]:
        """
        Get ordered list of nodes that outcome depends on.

        Nodes are ordered so that each node appears after its dependencies.
        """
        outcome = self.get_node(outcome_node_label)
        ancestors = nx.ancestors(self._graph, outcome)
        return [
            node for node in self.ordered_nodes if node == outcome or node in ancestors
        ]

    def build_model(self) -> Callable[..., None]:
        """
        Return a function that constructs the causal model defined by the Graph.

        The model that is defined by a graph is a function of the parameter values,
        or in this implementation the values of the `ParameterNode`s. This means that
        a model can only be specified (and then used to sample from, for example) once
        values for the `ParameterNode`s have been given. However, each of these models
        is built in the same way, and all that needs to be done is substitute the new
        parameter values for the old.
        """

        def _model(**parameter_values: npt.ArrayLike) -> None:
            """
            Create the model corresponding to the graph structure.

            The model created is a function of the values that the `ParameterNode`s in
            the graph take, which is what must be passed into the model as keyword
            arguments. Names of the keyword arguments should match the labels of the
            `ParameterNode`s, and their values should be the values of those parameters.
            """
            # Initialise node values for the `ParameterNode`s, which should have been
            # passed in via the keyword arguments.
            node_record = dict(parameter_values)
            # Confirm that all `ParameterNode`s have been assigned a value.
            for node in self.parameter_nodes:
                if node.label not in node_record:
                    msg = f"ParameterNode {node.label} not assigned."
                    raise KeyError(msg)

            # Build model sequentially, using the node_order to inform the construction
            # process.
            for node in self.ordered_dist_nodes:
                node_record[node.label] = node.create_model_site(**node_record)

        return _model
