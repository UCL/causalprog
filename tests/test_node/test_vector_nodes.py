"""Test for vector-valued nodes."""

import pytest
from numpyro.distributions import Normal

from causalprog.graph import ComponentNode, DistributionNode


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
def test_sample_shape(rng_key, shape):
    node = DistributionNode(Normal, shape=shape, label="X")
    assert node.sample({}, {}, 100, rng_key=rng_key).shape == (100, *shape)


def test_component_node(rng_key):
    node = DistributionNode(Normal, shape=(4, 5), label="X")
    node2 = ComponentNode(node.label, (0, 0), label="Y")
    s = node.sample({}, {}, 100, rng_key=rng_key)
    assert node2.sample({}, {"X": s}, 100, rng_key=rng_key).shape == (100,)


def test_another_component_node(rng_key):
    node = DistributionNode(Normal, shape=(4, 5), label="X")
    node2 = ComponentNode(node.label, (0,), shape=(5), label="Y")
    s = node.sample({}, {}, 100, rng_key=rng_key)
    assert node2.sample({}, {"X": s}, 100, rng_key=rng_key).shape == (100, 5)


def test_get_component_node(rng_key):
    node = DistributionNode(Normal, shape=(4, 5), label="X")
    s = node.sample({}, {}, 100, rng_key=rng_key)

    s0 = node[0].sample({}, {"X": s}, 100, rng_key=rng_key)
    assert s0.shape == (100, 5)
    assert node[:, 1].sample({}, {"X": s}, 100, rng_key=rng_key).shape == (100, 4)
    assert node[0, 1].sample({}, {"X": s}, 100, rng_key=rng_key).shape == (100,)
    assert node[0][1].sample({}, {"X": s, "X[0]": s0}, 100, rng_key=rng_key).shape == (
        100,
    )
