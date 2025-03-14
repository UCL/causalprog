"""Algorithms for iteration."""

from causalprog.graph import Graph, Node


def roots_down_to_outcome(
    graph: Graph,
    outcome_node_label: str,
) -> list[Node]:
    """
    Get ordered list of nodes that outcome depends on.

    Nodes are ordered so that each node appears after its dependencies.
    """
    pre = graph.predecessors

    nodes_need_sampling = [graph.get_node(outcome_node_label)]
    n = 0
    while n < len(nodes_need_sampling):
        new_n = len(nodes_need_sampling)
        for node in nodes_need_sampling[n:]:
            if node in pre and pre[node] not in nodes_need_sampling:
                nodes_need_sampling.append(pre[node])
        n = new_n

    return [
        node for node in graph.depth_first_nodes[::-1] if node in nodes_need_sampling
    ]
