"""Algorithms for evaluating a graph node."""

import numpy.typing as npt

from causalprog.graph import Graph


def evaluate_down_to(
    graph: Graph, outcome_node_label: str, **values: float | npt.NDArray[float]
) -> dict[str, float | npt.NDArray[float]]:
    """
    Evaluate all nodes down to a particular node.

    Args:
        graph: The graph that the node is contained in.
        outcome_node_label: The label of the node to evaluate down to.
        values: Values taken by nodes whose value is given

    Returns:
        A dictionary of the values of all the nodes that are ancestors of the input node
    """
    computed_values = {key: value for key, value in values.items()}
    nodes_to_evaluate = [
        n
        for n in graph.roots_down_to_outcome(outcome_node_label)
        if n.label not in values
    ]
    for node in nodes_to_evaluate:
        computed_values[node.label] = node.evaluate(**computed_values)
    return computed_values


def evaluate(
    graph: Graph, outcome_node_label: str, **values: float | npt.NDArray[float]
) -> float | npt.NDArray[float]:
    """
    Evaluate a node.

    Args:
        graph: The graph that the node is contained in.
        outcome_node_label: The label of the node to evaluate.
        values: Values taken by nodes whose value is given

    Returns:
        The evaluation of the node

    """
    return evaluate_down_to(graph, outcome_node_label, **values)[outcome_node_label]
