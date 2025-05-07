""""""

from collections.abc import Callable
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from causalprog.algorithms import expectation
from causalprog.causal_problem import CausalProblem
from causalprog.distribution.normal import NormalFamily
from causalprog.graph import DistributionNode, Graph, ParameterNode


def test_two_normal_example(  # noqa: PLR0915
    rng_key: jax.Array,
    n_samples: int = 1000,
    nu_x: float = 1.0,
    nu_y: float = 1.0,
    epsilon: float = 1.0,
    data: tuple[float, ...] = (2.0,),
    x0: tuple[float, ...] = (1.0,),
    *,
    plotting: bool = False,
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

    def expectation_with_n_samples() -> Callable[[Graph, DistributionNode], float]:
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
    analytic_result = minimize(
        analytic_ce, x0, constraints=[analytic_constraint], options={"disp": True}
    )
    assert np.isclose(analytic_result.x, true_analytic_value)

    # Setup the CausalProblem instance.

    cp = CausalProblem(graph, label="CP")
    cp.set_causal_estimand(
        expectation_with_n_samples(),
        rvs_to_nodes={"rv": "y"},
        # rvs_to_nodes={"rv": "mu"},
        graph_argument="g",
    )
    cp.set_constraints(
        expectation_with_n_samples(),
        rvs_to_nodes={"rv": "x"},
        # rvs_to_nodes={"rv": "mu"},
        graph_argument="g",
    )

    def ce(p):
        return cp.causal_estimand(p)

    def con(p):
        return np.abs(cp.constraints(p) - data)

    # Alright, now try solving the actual problem
    nlc = NonlinearConstraint(con, lb=-np.inf, ub=epsilon)
    result = minimize(
        ce,
        x0,
        constraints=[nlc],
        options={"disp": True},
        jac=lambda *p: np.atleast_1d(1.0),
    )

    if plotting:
        # Debug part of the test to check what the functions we are evaluating look like
        param_values = np.linspace(0.0, 3.0, num=500, endpoint=True)
        f_evals = np.zeros_like(param_values)
        c_evals = np.zeros_like(param_values)
        for i, val in enumerate(param_values):
            f_evals[i] = ce(np.atleast_1d(val))
            c_evals[i] = con(np.atleast_1d(val))[0]

        f_diff: np.ndarray = f_evals - analytic_ce(param_values)
        c_diff: np.ndarray = c_evals - analytic_con(param_values)

        n_rows = 2
        n_cols = 2
        fig, ax = plt.subplots(n_rows, n_cols)
        for i in range(n_rows):
            for j in range(n_cols):
                ax[i, j].set_xlabel(r"$\mu$")

        ax[0, 0].plot(param_values, f_evals, color="blue", label="E[Y]")
        ax[0, 0].set_ylabel("Function")
        ax[0, 0].plot(result.x, ce(result.x), color="red", marker="o")
        ax[0, 0].plot(analytic_result.x, ce(analytic_result.x), color="red", marker="x")

        ax[0, 1].plot(param_values, c_evals)
        ax[0, 1].set_ylabel("Constraint")
        ax[0, 1].plot(result.x, con(result.x), color="red", marker="o")
        ax[0, 1].plot(
            param_values,
            np.ones_like(param_values) * epsilon,
            color="red",
            linestyle="dashed",
        )
        ax[0, 1].plot(
            analytic_result.x, con(analytic_result.x), color="red", marker="x"
        )

        ax[1, 0].plot(param_values, f_diff)
        ax[1, 0].set_ylabel("Function difference")

        ax[1, 1].plot(param_values, c_diff)
        ax[1, 1].set_ylabel("Constraint difference")

        fig.tight_layout()

        save_loc = (Path(__file__).parent / ".." / ".." / ".vscode").resolve()
        fig.savefig(save_loc / "_two_normal_plot.png")

        print("Min / max function diff:", f_diff.min(), f_diff.max())  # noqa: T201
        print("Min / max constraint diff:", c_diff.min(), c_diff.max())  # noqa: T201

    assert np.isclose(result.x, analytic_result.x)
