"""Algorithms for evaluating a graph node."""

import numpy.typing as npt

from causalprog.graph import Graph


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
    nodes_to_evaluate = [
        n
        for n in graph.roots_down_to_outcome(outcome_node_label)
        if n.label not in values
    ]
    for node in nodes_to_evaluate:
        values[node.label] = node.evaluate(**values)
    return values[outcome_node_label]
