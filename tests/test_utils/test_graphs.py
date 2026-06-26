import numpy as np

from causalprog.algorithms import evaluate
from causalprog.utils.graphs import example_model


def test_example_model():
    graph = example_model(
        compute_u_x=lambda C: C,
        compute_u_y=lambda C: C + 1,
        compute_phi_x=lambda L: L[0],
        compute_x=lambda Z, PhiX, UX: Z[0] + UX - PhiX,
        compute_y=lambda X, UY: X * UY,
    )
    assert len(graph.nodes) == 8

    data = {
        "L": np.array([5.5]),
        "Z": np.array([2.0]),
        "C": 4.0,
    }

    assert np.allclose(evaluate(graph, "L", **data), np.array([5.5]))
    assert np.allclose(evaluate(graph, "Z", **data), np.array([2.0]))
    assert np.allclose(evaluate(graph, "C", **data), 4.0)
    assert np.allclose(evaluate(graph, "UX", **data), 4.0)
    assert np.allclose(evaluate(graph, "UY", **data), 5.0)
    assert np.allclose(evaluate(graph, "PhiX", **data), 5.5)
    assert np.allclose(evaluate(graph, "X", **data), 0.5)
    assert np.isclose(evaluate(graph, "Y", **data), 2.5)
