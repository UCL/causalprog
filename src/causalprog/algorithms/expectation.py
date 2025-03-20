"""Algorithms for estimating the expectation and standard deviation."""

import numpy as np
import numpy.typing as npt

from causalprog.graph import Graph


def sample(
    graph: Graph,
    outcome_node_label: str | None = None,
    samples: int = 1000,
) -> npt.NDArray[float]:
    """Sample data from a graph."""
    if outcome_node_label is None:
        outcome_node_label = graph.outcome.label

    nodes = graph.roots_down_to_outcome(outcome_node_label)

    values: dict[str, npt.NDArray[float]] = {}
    for node in nodes:
        values[node.label] = node.sample(values, samples)
    return values[outcome_node_label]


def expectation(
    graph: Graph,
    outcome_node_label: str | None = None,
    samples: int = 1000,
) -> float:
    """Estimate the expectation of a graph."""
    return sample(graph, outcome_node_label, samples).mean()


def standard_deviation(
    graph: Graph,
    outcome_node_label: str | None = None,
    samples: int = 1000,
) -> float:
    """Estimate the standard deviation of a graph."""
    return np.std(sample(graph, outcome_node_label, samples))
