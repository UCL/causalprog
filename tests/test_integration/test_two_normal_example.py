""""""

from collections.abc import Callable

import jax
import numpy as np
import pytest
from scipy.optimize import NonlinearConstraint, minimize

from causalprog.algorithms import expectation
from causalprog.causal_problem import CausalProblem
from causalprog.distribution.normal import NormalFamily
from causalprog.graph import DistributionNode, Graph, ParameterNode


@pytest.mark.parametrize(
    ("n_samples"),
    [
        pytest.param(1e3, id="1e3 samples"),
        pytest.param(1e6, id="1e6 samples"),
        pytest.param(1e8, id="1e8 samples"),
    ],
)
def test_two_normal_example(
    n_samples: int,
    rng_key: jax.Array,
    nu_x: float = 1.0,
    nu_y: float = 1.0,
    epsilon: float = 1.0,
    data: tuple[float, ...] = (2.0,),
    x0: tuple[float, ...] = (1.1,),
) -> None:
    """"""
    n_samples = int(n_samples)
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

    def expectation_with_n_samples() -> Callable[[Graph, DistributionNode], float]:
        def _inner(g: Graph, rv: DistributionNode) -> float:
            return expectation(g, rv.label, samples=n_samples, rng_key=rng_key)

        return _inner

    # Solve everything analytically first. That is, use the analytic formula for
    # the CE, and for the Constraints, and solve the resulting problem.
    # This will flag if we have setup our problem incorrectly, or changed something
    # which affects the problem further down the line.
    def analytic_ce(p):
        return p

    def analytic_con(p):
        return np.abs(p - data)

    analytic_constraint = NonlinearConstraint(analytic_con, lb=-np.inf, ub=epsilon)
    analytic_result = minimize(
        analytic_ce, x0, constraints=[analytic_constraint], options={"disp": True}
    )
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

    def ce_jacobian(*p):
        return 1.0

    def con(p):
        return np.abs(cp.constraints(p) - data)

    def con_jacobian(p):
        return -1.0 * (p < data) + (p > data)

    nlc = NonlinearConstraint(con, lb=-np.inf, ub=epsilon, jac=con_jacobian)
    result = minimize(
        ce,
        x0,
        constraints=[nlc],
        options={"disp": True},
        jac=ce_jacobian,
    )

    # When providing both Jacobians, error seems to scale with
    # inverse square-root of number of samples.
    # Use np.floor to provide more leeway in solution.
    atol = 10 ** (-np.floor(np.log10(n_samples) / 2.0))
    assert np.isclose(result.x, analytic_result.x, atol=atol)
