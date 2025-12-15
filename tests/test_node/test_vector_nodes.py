"""Test for vector-valued nodes."""

import pytest
from numpyro.distributions import Normal

import causalprog
from causalprog.graph import DistributionNode


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (0,),
        (1,),
        (4,),
        (10,),
        (2, 2),
        (3, 1, 4, 1, 5),
    ],
)
def test_vector_node(rng_key, shape):
    node = DistributionNode(Normal, shape=shape, label="X")
    assert node.sample({}, {}, 100, rng_key=rng_key).shape == (100,) + shape
