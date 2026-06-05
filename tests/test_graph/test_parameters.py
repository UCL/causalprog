"""Tests for graph module."""

import jax.numpy as jnp

from causalprog.graph import DataNode


def test_data_node(rng_key, raises_context):
    node = DataNode(label="mu")

    with raises_context(ValueError("Missing input for node")):
        node.sample({}, {}, 1, rng_key=rng_key)

    assert jnp.allclose(
        node.sample({node.label: 0.3}, {}, 10, rng_key=rng_key), jnp.full((10,), 0.3)
    )
