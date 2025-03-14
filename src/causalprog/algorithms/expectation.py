"""Algorithm for estimating the expectation of a process represented by a graph."""

import typing

if typing.TYPE_CHECKING:
    import numpy.typing as npt

from causalprog.graph import Graph

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

    values: dict[str, npt.NDArray[float]] = {}
    for node in nodes:
        values[node.label] = node.sample(values, samples)
    return values[outcome_node_label].sum() / samples
