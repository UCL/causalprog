"""Algorithms for applying do to a graph."""

from copy import deepcopy

from causalprog.graph import Graph


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

    old_g = graph._graph  # noqa: SLF001
    g = deepcopy(old_g)

    nodes_by_label = {n.label: n for n in g.nodes}
    g.remove_node(nodes_by_label[node])

    new_nodes = {}
    # Search through the old graph, identifying nodes that had parameters which were
    # defined by the node being fixed in the DO operation.
    # We recreate these nodes, but replace each such parameter we encounter with
    # a constant parameter equal that takes the fixed value given as an input.
    for original_node in old_g.nodes:
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
        # If we had to recreate a new node, add it to the new (Di)Graph.
        # Also record the name of the node that it is set to replace
        if new_n is not None:
            g.add_node(new_n)
            new_nodes[original_node.label] = new_n

    # Any new_nodes whose counterparts connect to other nodes in the network need
    # to mimic these links.
    for edge in old_g.edges:
        labels = [e.label for e in edge]
        if node in labels:
            continue
        for label in labels:
            if label in new_nodes:
                g.add_edge(*labels)
                break
    # Now that the new_nodes are present in the graph, and correctly connected, remove
    # their counterparts from the graph.
    for original_node in new_nodes:
        g.remove_node(nodes_by_label[original_node])

    return Graph(label=label, graph=g)
