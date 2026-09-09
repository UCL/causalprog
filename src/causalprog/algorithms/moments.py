"""Algorithms for estimating the expectation and standard deviation."""

import jax

from causalprog.graph import Graph


def sample(
    graph: Graph,
    outcome_node_label: str,
    samples: int,
    *,
    parameter_values: dict[str, float] | None = None,
    rng_key: jax.Array,
) -> jax.Array:
    """
    Sample data from (a random variable attached to) a node in a graph.

    Args:
        graph: The graph from which to sample.
        outcome_node_label: The label of the node to sample from.
        samples: Number of desired samples.
        parameter_values: Values to be taken by node parameters.
        rng_key: PRNG key to use to generate samples.

    Returns:
        Array of `samples` elements, containing the random samples.

    """
    nodes = graph.roots_down_to_outcome(outcome_node_label)

    values: dict[str, jax.Array] = {}
    keys = jax.random.split(rng_key, len(nodes))

    for node, key in zip(nodes, keys, strict=False):
        values[node.label] = node.sample(
            parameter_values or {},
            values,
            samples,
            rng_key=key,
        )
    return values[outcome_node_label]


def expectation(
    graph: Graph,
    outcome_node_label: str,
    samples: int,
    *,
    parameter_values: dict[str, float] | None = None,
    rng_key: jax.Array,
) -> float:
    """
    Estimate the expectation of (a random variable attached to) a node in a graph.

    Args:
        graph: The graph containing the node.
        outcome_node_label: The label of the node to compute the expectation of.
        samples: Number of samples to use to estimate the expectation.
        parameter_values: Values to be taken by node parameters.
        rng_key: PRNG key to use to generate samples.

    Returns:
        Approximation to the expectation of `outcome_node_label`.

    """
    return moment(
        1,
        graph,
        outcome_node_label,
        samples,
        rng_key=rng_key,
        parameter_values=parameter_values,
    )


def standard_deviation(
    graph: Graph,
    outcome_node_label: str,
    samples: int,
    *,
    parameter_values: dict[str, float] | None = None,
    rng_key: jax.Array,
    rng_key_first_moment: jax.Array | None = None,
) -> float:
    r"""
    Estimate the standard deviation of (a RV attached to) a node in a graph.

    The method computes the standard deviation of node $X$ via the formula

    $$ \sqrt{\mathrm{Var}(X)} = \sqrt{\mathbb{E}[X^2] - \mathbb{E}[X]^2}. $$

    Args:
        graph: The graph containing the node.
        outcome_node_label: The label of the node to compute the standard deviation of.
        samples: Number of samples to use to estimate the standard deviation.
        parameter_values: Values to be taken by node parameters.
        rng_key: PRNG key to use to generate samples.
        rng_key_first_moment: PRNG key that will be used to approximate the expectation,
            used in the formula to calculate the standard deviation.

    Returns:
        Approximation to the standard deviation of `outcome_node_label`.

    """
    return (
        moment(
            2,
            graph,
            outcome_node_label,
            samples,
            rng_key=rng_key,
            parameter_values=parameter_values,
        )
        - moment(
            1,
            graph,
            outcome_node_label,
            samples,
            rng_key=rng_key if rng_key_first_moment is None else rng_key_first_moment,
            parameter_values=parameter_values,
        )
        ** 2
    ) ** 0.5


def moment(
    order: int,
    graph: Graph,
    outcome_node_label: str,
    samples: int,
    *,
    parameter_values: dict[str, float] | None = None,
    rng_key: jax.Array,
) -> float:
    """
    Estimate a moment of (a random variable attached to) a node in a graph.

    Args:
        order: Order of the moment to estimate.
        graph: The graph containing the node.
        outcome_node_label: The label of the node to compute the moment of.
        samples: Number of samples to be used to estimate the moment.
        parameter_values: Values to be taken by node parameters.
        rng_key: PRNG key to use to generate samples.

    Returns:
        Approximation to the `order` moment of `outcome_node_label`.

    """
    return (
        sum(
            sample(
                graph,
                outcome_node_label,
                samples,
                rng_key=rng_key,
                parameter_values=parameter_values,
            )
            ** order
        )
        / samples
    )
