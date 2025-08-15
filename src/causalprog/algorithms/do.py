"""Algorithms for applying do to a graph."""

from copy import deepcopy

from causalprog.graph import Graph, Node


def get_included_excluded_successors(
    graph: Graph, nodes: dict[str, Node], node: str
) -> tuple[list[str], list[str]]:
    """
    Get list of successors that are included in and excluded from nodes dictionary.

    Args:
        graph: The graph
        nodes: A dictionary of nodes, indexed by label
        node: The node to check the successors of

    Returns:
        Lists of included and excluded nodes

    """
    included = []
    excluded = []
    for n in graph.successors[graph.get_node(node)]:
        if n in nodes:
            included.append(n)
        else:
            excluded.append(n)
    return included, excluded


def removable_nodes(graph: Graph, nodes: dict[str, Node]) -> list[str]:
    """
    Generate list of nodes that can be removed from the graph.

    Args:
        graph: The graph
        nodes: A dictionary of nodes, indexed by label

    Returns:
        List of labels of removable nodes

    """
    removable = []
    for n in nodes:
        included, excluded = get_included_excluded_successors(graph, nodes, n)
        if len(excluded) > 0 and len(included) == 0:
            removable.append(n)
    return removable


def do(graph: Graph, node: str, value: float, label: str | None = None) -> Graph:
    """
    Apply do to a graph.

    Args:
        graph: The graph to apply do to. This will be copied.
        node: The label of the node to apply do to.
        value: The value to set the node to.
        label: The label of the new graph

    Returns:
        A copy of the graph with do applied

    """
    if label is None:
        label = f"{graph.label}|do({node}={value})"

    nodes = {n.label: n for n in graph.nodes}
    del nodes[node]

    # Search through the old graph, identifying nodes that had parameters which were
    # defined by the node being fixed in the DO operation.
    # We recreate these nodes, but replace each such parameter we encounter with
    # a constant parameter equal that takes the fixed value given as an input.
    for original_node in graph.nodes:
        new_n = None
        for parameter_name, parameter_target_node in original_node.parameters.items():
            if parameter_target_node == node:
                # If this parameter in the original_node was determined by the node we
                # are fixing with DO.
                if new_n is None:
                    new_n = deepcopy(original_node)
                # Swap the parameter to a constant parameter, giving it the fixed value
                new_n.constant_parameters[parameter_name] = value
                # Remove the parameter from the node's record of non-constant parameters
                new_n.parameters.pop(parameter_name)
        # If we had to recreate a new node, replace it in the nodes list
        if new_n is not None:
            nodes[original_node.label] = new_n

    # Recursively remove nodes that are predecessors of removed nodes
    nodes_to_remove = [node]
    while len(nodes_to_remove) > 0:
        nodes_to_remove = removable_nodes(graph, nodes)
        for n in removable_nodes(graph, nodes):
            del nodes[n]

    # Check for nodes that are predecessors of both a removed node and a remaining node
    # and throw an error if one of these is found
    for n in nodes:
        _, excluded = get_included_excluded_successors(graph, nodes, n)
        if len(excluded) > 0:
            msg = (
                "Node that is predecessor of node set by do and "
                'nodes that are not removed found ("{n}")'
            )
            raise ValueError(msg)

    g = Graph(label=label)
    for n in nodes.values():
        g.add_node(n)

    # Any nodes whose counterparts connect to other nodes in the network need
    # to mimic these links.
    for edge in [e for e in graph.edges if e[0].label in nodes and e[1].label in nodes]:
        g.add_edge(nodes[edge[0].label], nodes[edge[1].label])

    return g
