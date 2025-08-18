"""Algorithms for applying do to a graph."""

from copy import deepcopy

from causalprog.graph import Graph, Node


def get_included_excluded_successors(
    graph: Graph, nodes_to_split: dict[str, Node], successors_of: str
) -> tuple[list[str], list[str]]:
    """
    Split nodes into two groups; those that are successors or another node, and those that are not.

    Args:
        graph: The graph
        nodes_to_split: A dictionary of nodes, indexed by label
        successor_of: The node to check the successors of

    Returns:
        Lists of included and excluded nodes

    """
    included = []
    excluded = []
    for n in graph.successors[graph.get_node(node)]:
        if n.label in nodes:
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

    nodes = {n.label: deepcopy(n) for n in graph.nodes if n.label != node}

    # Search through the old graph, identifying nodes that had parameters which were
    # defined by the node being fixed in the DO operation.
    # We recreate these nodes, but replace each such parameter we encounter with
    # a constant parameter equal that takes the fixed value given as an input.
    for n in nodes.values():
        params = tuple(n.parameters.keys())
        for parameter_name in params:
            if n.parameters[parameter_name] == node:
                # Swap the parameter to a constant parameter, giving it the fixed value
                n.constant_parameters[parameter_name] = value
                # Remove the parameter from the node's record of non-constant parameters
                n.parameters.pop(parameter_name)

    # Recursively remove nodes that are predecessors of removed nodes
    nodes_to_remove = [node]
    while len(nodes_to_remove) > 0:
        nodes_to_remove = removable_nodes(graph, nodes)
        for n in removable_nodes(graph, nodes):
            nodes.pop(n)

    # Check for nodes that are predecessors of both a removed node and a remaining node
    # and throw an error if one of these is found
    for n in nodes:
        _, excluded = get_included_excluded_successors(graph, nodes, n)
        if len(excluded) > 0:
            msg = (
                "Node that is predecessor of node set by do and "
                f'nodes that are not removed found ("{n}")'
            )
            raise ValueError(msg)

    g = Graph(label=f"{label}|do[{node}={value}]")
    for n in nodes.values():
        g.add_node(n)

    # Any nodes whose counterparts connect to other nodes in the network need
    # to mimic these links.
    for edge in graph.edges:
        if edge[0].label in nodes and edge[1].label in nodes:
            g.add_edge(edge[0].label, edge[1].label)

    return g
