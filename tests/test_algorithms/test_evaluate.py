"""Tests for evaluate algorithms."""

import numpy as np
import jax.numpy as jnp
import pytest
from causalprog.graph import DataNode


def test_evaluate_node(raises_context):
    node = DataNode(label="A")
    assert np.isclose(node.evaluate(A=2.0), 2.0)

    with raises_context(ValueError("Missing input for node: A")):
        node.evaluate()
