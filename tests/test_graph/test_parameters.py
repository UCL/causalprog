"""Tests for graph module."""

from typing import Literal, TypeAlias

import jax.numpy as jnp

from causalprog.graph import DistributionNode, ParameterNode

NormalGraphNodeNames: TypeAlias = Literal["mean", "cov", "X"]
NormalGraphNodes: TypeAlias = dict[
    NormalGraphNodeNames, DistributionNode | ParameterNode
]


def test_parameter_node(rng_key, raises_context):
    node = ParameterNode(label="mu")

    with raises_context(ValueError("Missing input for parameter")):
        node.sample({}, {}, 1, rng_key=rng_key)

    assert jnp.allclose(
        node.sample({node.label: 0.3}, {}, 10, rng_key=rng_key)[0], [0.3] * 10
    )
