""""""

import jax
import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from causalprog.algorithms import expectation
from causalprog.causal_problem import CausalProblem
from causalprog.distribution.normal import NormalFamily
from causalprog.graph import DistributionNode, Graph, ParameterNode


def test_two_normal_example(
    rng_key: jax.Array,
    n_samples: int = 100,
    nu_x: float = 1.0,
    nu_y: float = 1.0,
    epsilon: float = 1.0,
    data: tuple[float, ...] = (2.0,),
    x0: tuple[float, ...] = (1.25,),
) -> None:
    """"""
    data = np.array(data, ndmin=1)
    x0 = np.array(x0, ndmin=1)
    true_analytic_value = np.array(data) - epsilon

    mu = ParameterNode("mu")
    x = DistributionNode(
        NormalFamily(),
        label="x",
        parameters={"mean": "mu"},
        constant_parameters={"cov": nu_x**2},
        is_outcome=True,
    )
    y = DistributionNode(
        NormalFamily(),
        label="y",
        parameters={"mean": "x"},
        constant_parameters={"cov": nu_y**2},
        is_outcome=True,
    )

    graph = Graph(label="G")
    graph.add_edge(mu, x)
    graph.add_edge(x, y)

    def expectation_with_n_samples():
        def _inner(g: Graph, rv: DistributionNode) -> float:
            return expectation(g, rv.label, samples=n_samples, rng_key=rng_key)

        return _inner

    # Solve everything analytically first. That is, use the analytic formula for
    # the CE, and for the Constraints, and solve the resulting problem.
    def analytic_ce(p):
        return p

    def analytic_con(p):
        return np.abs(p - data)

    analytic_constraint = NonlinearConstraint(analytic_con, lb=-np.inf, ub=epsilon)
    analytic_result = minimize(analytic_ce, x0, constraints=[analytic_constraint])
    assert np.isclose(analytic_result.x, true_analytic_value)

    # Setup the CausalProblem instance.

    cp = CausalProblem(graph, label="CP")
    cp.set_causal_estimand(
        expectation_with_n_samples(),
        rvs_to_nodes={"rv": "y"},
        graph_argument="g",
    )
    cp.set_constraints(
        expectation_with_n_samples(),
        rvs_to_nodes={"rv": "x"},
        graph_argument="g",
    )

    def ce(p):
        return cp.causal_estimand(p)

    def con(p):
        return np.abs(cp.constraints(p) - data)

    # Prior to solving, check that evaluating the CE and constraints bears some
    # resemblance to their analytic counterparts.
    range_check = np.linspace(0.0, 5.0, num=50)
    for value in range_check:
        v = np.atleast_1d(value)
        assert np.isclose(ce(v), analytic_ce(v), atol=5 / n_samples)
        assert np.isclose(con(v), analytic_con(v), atol=5 / n_samples)

    # Alright, now try solving the actual problem
    nlc = NonlinearConstraint(con, lb=-np.inf, ub=epsilon)
    result = minimize(
        ce,
        x0,
        constraints=[nlc],
        options={"disp": True},
        jac=lambda *p: np.atleast_1d(1.0),
    )

    assert np.isclose(result.x, analytic_result.x)
