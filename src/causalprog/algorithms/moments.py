"""Algorithms for estimating the expectation and standard deviation."""

import jax
import numpy.typing as npt

from causalprog.graph import Graph


def sample(
    graph: Graph,
    outcome_node_label: str,
    samples: int,
    *,
    rng_key: jax.Array,
) -> npt.NDArray[float]:
    """Sample data from a graph."""
    nodes = graph.roots_down_to_outcome(outcome_node_label)

    values: dict[str, npt.NDArray[float]] = {}
    keys = jax.random.split(rng_key, len(nodes))

    for node, key in zip(nodes, keys, strict=False):
        values[node.label] = node.sample(values, samples, rng_key=key)
    return values[outcome_node_label]


def expectation(
    graph: Graph,
    outcome_node_label: str,
    samples: int,
    *,
    rng_key: jax.Array,
) -> float:
    """Estimate the expectation of a graph."""
    return moment(1, graph, outcome_node_label, samples, rng_key=rng_key)


def standard_deviation(
    graph: Graph,
    outcome_node_label: str,
    samples: int,
    *,
    rng_key: jax.Array,
    rng_key_first_moment: jax.Array | None = None,
) -> float:
    """Estimate the standard deviation of a graph."""
    return (
        moment(2, graph, outcome_node_label, samples, rng_key=rng_key)
        - moment(
            1,
            graph,
            outcome_node_label,
            samples,
            rng_key=rng_key if rng_key_first_moment is None else rng_key_first_moment,
        )
        ** 2
    ) ** 0.5


def moment(
    order: int,
    graph: Graph,
    outcome_node_label: str,
    samples: int,
    *,
    rng_key: jax.Array,
) -> float:
    """Estimate moment of the given order of the data."""
    return (
        sum(sample(graph, outcome_node_label, samples, rng_key=rng_key) ** order)
        / samples
    )
