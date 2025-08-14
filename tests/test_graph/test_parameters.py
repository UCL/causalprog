"""Tests for graph module."""

from typing import Literal, TypeAlias

import numpy as np
import pytest

from causalprog.graph import DistributionNode, ParameterNode

NormalGraphNodeNames: TypeAlias = Literal["mean", "cov", "X"]
NormalGraphNodes: TypeAlias = dict[
    NormalGraphNodeNames, DistributionNode | ParameterNode
]


def test_parameter_node(rng_key):
    node = ParameterNode(label="mu")

    with pytest.raises(ValueError, match="Missing input for parameter"):
        node.sample({}, {}, 1, rng_key=rng_key)

    assert np.allclose(
        node.sample({node.label: 0.3}, {}, 10, rng_key=rng_key)[0], [0.3] * 10
    )
