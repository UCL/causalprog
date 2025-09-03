from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest

from causalprog.causal_problem.causal_problem import CausalProblem
from causalprog.causal_problem.components import CausalEstimand, Constraint
from causalprog.graph import Graph
from causalprog.solvers.sgd import stochastic_gradient_descent
from causalprog.utils.norms import l2_normsq


@pytest.mark.parametrize(
    "is_solving_max",
    [
        pytest.param(False, id="Minimise"),
        pytest.param(True, id="Maximise"),
    ],
)
def test_two_normal_example(
    rng_key: jax.Array,
    two_normal_graph: Callable[..., Graph],
    adams_learning_rate: float = 1.0e-1,
    n_samples: int = 500,
    phi_observed: float = 0.0,  # The observed data
    epsilon: float = 1.0,  # The tolerance in the observed data
    nu_x_starting_value: float = 1.0,  # Where to start nu_x in the solver initial guess
    lagrange_mult_sol: float = 1.0,  # Solution value of the lagrange multiplier
    maxiter: int = 100,  # Max iterations to allow (~100 sufficient for test cases)
    # Threshold for minimisation function value being considered 0
    minimisation_tolerance: float = 1.0e-6,
    *,
    is_solving_max: bool,
):
    r"""Solves the 'two normal' graph example problem.

    We use the `two_normal_graph` with `cov=1.0`. For the purposes of this test, we will
    write $\mu_{ux}$ for the parameter `mean`, and $\nu_{x}$ for the parameter `cov2`,
    giving  us the following model:

    $$
    \mu_{ux} \rightarrow UX \sim \mathcal{N}(\mu_{ux}, 1.0)
    \rightarrow X, X \vert UX \sim \mathcal{N}(UX, \nu_{x})
    \leftarrow \nu_{x}.
    $$

    We will be interested in the causal estimand

    $$ \sigma(\mu_{ux}, \nu_{x}) = \mathbb{E}[X] = \mu_{ux}, $$

    with observed data (constraints)

    $$ \phi(\mu_{ux}, \nu_{x}) = \mathbb{E}[UX] = \mu_{ux}. $$

    With observed data $\phi_{obs}$, and tolerance in the data $\epsilon$, we are
    effectively looking to solve the minimisation problem;

    $$ \mathrm{min}_{\mu_{ux}, \nu_{x}} \mu_{ux}, \quad
    \text{subject to } \vert \mu_{ux} - \phi_{obs} \vert \leq \epsilon.
    $$

    The solution to this is $\mu_{ux}^{*} = \mu_{ux} \pm \phi_{obs}$ ($+$ in the
    maximisation case). The value of $\nu_{x}$ can be any positive value, since in this
    setup both $\phi$ and $\sigma$ are independent of it.

    The corresponding Lagrangian that we will form will be

    $$ \mathcal{L}(\mu_{ux}, \nu_{x}, \lambda) = \pm \mu_{ux}
    + \lambda(\vert \mu_{ux} - \phi_{obs} \vert - \epsilon), $$

    (again with $+\mu_{ux}$ in the maximisation case). In both cases, $\mathcal{L}$ is
    minimised at $(\mu_{ux}^{*}, \nu_x, 1)$.
    """
    ce_prefactor = 1.0 if not is_solving_max else -1.0
    mu_x_sol = phi_observed - ce_prefactor * epsilon

    # Setup the optimisation problem from the graph
    ce = CausalEstimand(do_with_samples=lambda **pv: pv["X"].mean())
    con = Constraint(
        model_quantity=lambda **pv: pv["UX"].mean(),
        data=phi_observed,
        tolerance=epsilon,
    )
    cp = CausalProblem(
        two_normal_graph(cov=1.0),
        con,
        causal_estimand=ce,
    )
    lagrangian = cp.lagrangian(n_samples=n_samples, maximum_problem=is_solving_max)

    # We'll be seeking stationary points of the Lagrangian, using the
    # naive approach of minimising the norm of its gradient. We will need to
    # ensure we "converge" to a minimum value suitably close to 0.
    def objective(x, key):
        v = jax.grad(lagrangian, argnums=(0, 1))(*x, rng_key=key)
        return l2_normsq(v)

    # Choose a starting guess that is at the optimal solution, in the hopes that
    # SGD converges quickly. We almost certainly will not have this luxury in general.
    # The value of nu_x is free; the Lagrangian is independent of it.
    # As such, it can take any value and should not change during the optimisation
    # iterations.
    params = {
        "mean": mu_x_sol,
        "cov2": nu_x_starting_value,
    }
    l_mult = jnp.atleast_1d(lagrange_mult_sol)

    opt_params, _, _, _ = stochastic_gradient_descent(
        objective,
        (params, l_mult),
        convergence_criteria=lambda x, _: jnp.abs(x),
        fn_kwargs={"key": rng_key},
        learning_rate=adams_learning_rate,
        maxiter=maxiter,
        tolerance=minimisation_tolerance,
    )
    # Unpack concatenated arguments
    params, l_mult = opt_params

    # The lagrangian is independent of nu_x, thus it should not have changed value.
    assert jnp.isclose(params["cov2"], nu_x_starting_value), (
        "nu_x has changed significantly from the starting value."
    )

    # Confirm that we found a minimiser that does satisfy the inequality constraints.
    assert jnp.all(l_mult > 0.0), (
        f"Converged, but not to a minimiser (lagrange multiplier = {l_mult})"
    )

    # Give a generous error margin in mu_ux and the Lagrange multiplier,
    # given SGD is being used on MC-integral functions.
    rtol = jnp.sqrt(1.0 / n_samples)
    assert jnp.isclose(params["mean"], mu_x_sol, rtol=rtol)
    assert jnp.allclose(l_mult, lagrange_mult_sol, atol=rtol)
