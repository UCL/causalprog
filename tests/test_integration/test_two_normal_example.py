r"""Integration test for the two-normal example, <https://github.com/UCL/causalprog/issues/38>.

In this test, we have the following setup:

- A single parameter $\mu$.
- A random variable $X$ distributed as $X \sim \mathcal{N}(\mu, \nu_X^2)$ for a fixed
    $\nu_X$.
- A random variable $Y$ distributed as $Y \vert X \sim \mathcal{N}(X, \nu_Y^2)$ for a
    fixed $\nu_Y$.
- The causal estimand of interest $\sigma = \mathbb{E}[Y]$.
- The constraints function is $\vert \mathbb{E}[X] - \phi \vert$ for an empirical
    observation of the expectation of $X$, $\phi$.

For a fixed tolerance $\epsilon > 0$, we are thus we are looking to solve the
following optimisation problem (1):

$$ \text{min / max}_\mu \mathbb{E}[Y], $$
$$ \text{subject to } \vert \mathbb{E}[X] - \phi \vert \leq \epsilon. $$

It should be noted that analytically, $\mathbb{E}[X] = \mathbb{E}[Y] = \mu$.
This means that we are effectively solving (2):

$$ \text{min / max}_\mu \mu, $$
$$ \text{subject to } \vert \mu - \phi \vert \leq \epsilon. $$

which we can immediately spot has solutions $\mu = \phi \pm \epsilon$ (larger value in
the case of maximisation).
"""

from collections.abc import Callable
from typing import Literal

import jax
import numpy as np
import pytest
from scipy.optimize import NonlinearConstraint, minimize

from causalprog.algorithms import expectation
from causalprog.causal_problem import CausalProblem
from causalprog.distribution.normal import NormalFamily
from causalprog.graph import DistributionNode, Graph, ParameterNode


@pytest.mark.parametrize(
    ("n_samples", "max_or_min"),
    [
        pytest.param(1e3, "min", id="[Min] 1e3 samples"),
        pytest.param(1e6, "min", id="[Min] 1e6 samples"),
        pytest.param(1e8, "min", id="[Min] 1e8 samples"),
        pytest.param(1e3, "max", id="[Max] 1e3 samples"),
        pytest.param(1e6, "max", id="[Max] 1e6 samples"),
        pytest.param(1e8, "max", id="[Max] 1e8 samples"),
    ],
)
def test_two_normal_example(
    n_samples: int,
    max_or_min: Literal["max", "min"],
    rng_key: jax.Array,
    nu_x: float = 1.0,
    nu_y: float = 1.0,
    epsilon: float = 1.0,
    data: tuple[float, ...] = (2.0,),
    x0: tuple[float, ...] = (1.1,),
) -> None:
    r"""Integration test for the two normal example.

    0) Record the analytic answer we expect, `true_analytic_value`.
    1) Compute the result of solving (1), `analytic_result` via the optimiser, to ensure
        that we have setup and understood the problem (and analytical answer) correctly.
        This also makes us robust against behaviour changes in the `causal_estimand` and
        `constraints` methods of the `CausalProblem` class.
    2) Check that `analytic_result` is close to `true_analytic_value`.
    3) Compute the result of (2) via the optimiser, `result`.
    4) Check that `result` is close to both `true_analytic_result` and `analytic_result`
        (see below for tolerances used).

    Empirical experiments suggest that the absolute difference between the `result` and
    `analytic_result` scales in proportion to the inverse square of the number of
    samples used;

    $$ \mathrm{atol} \propto \mathrm{samples}^{-0.5}, $$

    so to be generous, we use `atol = 10 ** (-np.floor(np.log10(n_samples) / 2.)))`.

    Finally, it should be noted that in order to obtain a good answer (in any case), we
    need to provide the Jacobians of the causal estimand and constraints functions to
    the solver. Without these, the results are poor (if the optimiser converges at all).
    """
    minimise_options = {"disp": False, "maxiter": 20}
    # Maximisation is minimisation of the negation of the objective function.
    prefactor = 1.0 if max_or_min == "min" else -1.0

    n_samples = int(n_samples)
    data = np.array(data, ndmin=1)
    x0 = np.array(x0, ndmin=1)
    true_analytic_value = np.array(data) - prefactor * epsilon

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
        return prefactor * cp.causal_estimand(p)

    def ce_jacobian(*p):
        # Gradient is prefactor * 1.0 since we're effectively minimising y = x
        return prefactor

    def con(p):
        return np.abs(cp.constraints(p) - data)

    def con_jacobian(p):
        return -1.0 * (p < data) + (p > data)

    # 1) Analytic solve
    def analytic_ce(p):
        return prefactor * p

    def analytic_con(p):
        return np.abs(p - data)

    analytic_constraint = NonlinearConstraint(analytic_con, lb=-np.inf, ub=epsilon)
    analytic_result = minimize(
        analytic_ce, x0, constraints=[analytic_constraint], options=minimise_options
    )

    # 2) Check analytic solve.
    assert np.isclose(analytic_result.x, true_analytic_value)

    # 3) Solve (1) via the CausalProblem class methods.
    nlc = NonlinearConstraint(con, lb=-np.inf, ub=epsilon, jac=con_jacobian)
    result = minimize(
        ce,
        x0,
        constraints=[nlc],
        options=minimise_options,
        jac=ce_jacobian,
    )

    # 4) Check proximity to correct solution.
    atol = 10.0 ** (-np.floor(np.log10(n_samples) / 2.0))
    assert np.isclose(result.x, analytic_result.x, atol=atol)
    assert np.isclose(result.x, true_analytic_value, atol=atol)
