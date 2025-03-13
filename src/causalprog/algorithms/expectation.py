"""Algorithm for estimating the expectation of a process represented by a graph."""

from ..graph import Graph
from .iteration import roots_down_to_outcome


def expectation(
    graph: Graph,
    outcome_node_label: str | None = None,
    samples: int = 1000,
) -> float:
    """Estimate the expectation of a graph."""
    if outcome_node_label is None:
        outcome_node_label = graph.outcome.label

    nodes = roots_down_to_outcome(graph, outcome_node_label)

    mean = 0.0
    for _ in range(samples):
        values = {}
        for node in nodes:
            values[node.label] = node.sample(values)
        mean += values[outcome_node_label] / samples
    return mean
