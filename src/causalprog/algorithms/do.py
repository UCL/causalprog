"""Algorithms for applying do to a graph."""

from causalprog.graph import Graph, Node


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
    g = old_g.copy()

    g.remove_node(graph.get_node(node))

    new_nodes: dict[str, Node] = {}
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
                    new_n = original_node.copy()
                # Swap the parameter to a constant parameter, giving it the fixed value
                new_n.constant_parameters[parameter_name] = value
                # Remove the parameter from the node's record of non-constant parameters
                new_n.parameters.pop(parameter_name)
        # If we had to recreate a new node, add it to the new (Di)Graph.
        # Also record the name of the node that it is set to replace
        if new_n is not None:
            g.add_node(new_n)
            # new_nodes[original_node.label] = new_node ?

    # Any new_nodes whose counterparts connect to other nodes in the network need
    # to mimic these links.
    for edge in old_g.edges:
        if edge[0].label in new_nodes or edge[1].label in new_nodes:
            g.add_edge(
                new_nodes.get(edge[0].label, edge[0]),
                new_nodes.get(edge[1].label, edge[1]),
            )
    # Now that the new_nodes are present in the graph, and correctly connected, remove
    # their counterparts from the graph.
    for original_node in new_nodes:
        g.remove_node(graph.get_node(original_node))

    return Graph(label=label, graph=g)
