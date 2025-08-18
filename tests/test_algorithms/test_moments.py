"""Tests for moment algorithms."""

import numpy as np
import pytest

from causalprog import algorithms

max_samples = 10**5


@pytest.mark.parametrize(
    ("mean", "stdev", "samples", "rtol"),
    [
        pytest.param(1.0, 1.0, 10, 1, id="N(mean=1, stdev=1), 10 samples"),
        pytest.param(2.0, 0.8, 1000, 1e-1, id="N(mean=2, stdev=0.8), 1000 samples"),
        pytest.param(1.0, 0.8, 100000, 1e-2, id="N(mean=1, stdev=0.8), 10^5 samples"),
        pytest.param(1.0, 1.2, 10000000, 1e-3, id="N(mean=1, stdev=1.2), 10^7 samples"),
    ],
)
def test_expectation_stdev_single_normal_node(
    normal_graph, samples, rtol, mean, stdev, rng_key
):
    if samples > max_samples:
        pytest.xfail("Test currently too slow")

    graph = normal_graph(mean, stdev)

    # Check within hand-computation
    assert np.isclose(
        algorithms.expectation(
            graph, outcome_node_label="X", samples=samples, rng_key=rng_key
        ),
        mean,
        rtol=rtol,
    )
    assert np.isclose(
        algorithms.standard_deviation(
            graph, outcome_node_label="X", samples=samples, rng_key=rng_key
        ),
        stdev,
        rtol=rtol,
    )


@pytest.mark.parametrize(
    ("mean", "stdev", "stdev2", "samples", "rtol"),
    [
        pytest.param(
            1.0,
            1.0,
            0.8,
            100,
            1,
            id="N(mean=N(mean=0, stdev=1), stdev=0.8), 100 samples",
        ),
        pytest.param(
            3.0,
            0.5,
            1.0,
            10000,
            1e-1,
            id="N(mean=N(mean=3, stdev=0.5), stdev=1), 10^4 samples",
        ),
        pytest.param(
            2.0,
            0.7,
            0.8,
            1000000,
            1e-2,
            id="N(mean=N(mean=2, stdev=0.7), stdev=0.8), 10^6 samples",
        ),
    ],
)
def test_mean_stdev_two_node_graph(
    two_normal_graph, samples, rtol, mean, stdev, stdev2, rng_key
):
    if samples > max_samples:
        pytest.xfail("Test currently too slow")

    graph = two_normal_graph(mean=mean, cov=stdev, cov2=stdev2)

    assert np.isclose(
        algorithms.expectation(
            graph, outcome_node_label="X", samples=samples, rng_key=rng_key
        ),
        mean,
        rtol=rtol,
    )
    assert np.isclose(
        algorithms.standard_deviation(
            graph, outcome_node_label="X", samples=samples, rng_key=rng_key
        ),
        np.sqrt(stdev**2 + stdev2**2),
        rtol=rtol,
    )


@pytest.mark.parametrize(
    ("samples", "rtol"),
    [
        pytest.param(100, 1, id="100 samples"),
        pytest.param(10000, 1e-1, id="10^4 samples"),
        pytest.param(1000000, 1e-2, id="10^6 samples"),
    ],
)
def test_expectation(two_normal_graph, rng_key, samples, rtol):
    if samples > max_samples:
        pytest.xfail("Test currently too slow")
    graph = two_normal_graph(1.0, 1.2, 0.8)

    assert np.isclose(
        algorithms.expectation(
            graph, outcome_node_label="X", samples=samples, rng_key=rng_key
        ),
        algorithms.moments.sample(
            graph, outcome_node_label="X", samples=samples, rng_key=rng_key
        ).mean(),
        rtol=rtol,
    )


@pytest.mark.parametrize(
    ("samples", "rtol"),
    [
        pytest.param(100, 1, id="100 samples"),
        pytest.param(10000, 1e-1, id="10^4 samples"),
        pytest.param(1000000, 1e-2, id="10^6 samples"),
    ],
)
def test_stdev(two_normal_graph, rng_key, samples, rtol):
    if samples > max_samples:
        pytest.xfail("Test currently too slow")
    graph = two_normal_graph(1.0, 1.2, 0.8)

    assert np.isclose(
        algorithms.standard_deviation(
            graph, outcome_node_label="X", samples=samples, rng_key=rng_key
        ),
        algorithms.moments.sample(
            graph, outcome_node_label="X", samples=samples, rng_key=rng_key
        ).std(),
        rtol=rtol,
    )


@pytest.mark.parametrize("samples", [1, 2, 10, 100])
def test_sample_shape(two_normal_graph, rng_key, samples):
    graph = two_normal_graph(1.0, 1.2, 0.8)

    s1 = algorithms.moments.sample(graph, "X", samples, rng_key=rng_key)
    assert s1.shape == () if samples == 1 else (samples,)

    s2 = algorithms.moments.sample(graph, "UX", samples, rng_key=rng_key)
    assert s2.shape == () if samples == 1 else (samples,)
