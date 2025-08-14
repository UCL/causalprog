from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy.typing as npt
import optax
import pytest
from numpyro.infer import Predictive

from causalprog.graph import Graph


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
    # Setup the optimisation problem from the graph
    g = two_normal_graph(cov=1.0)
    predictive_model = Predictive(g.model, num_samples=n_samples)

    def lagrangian(
        parameter_values: dict[str, npt.ArrayLike],
        predictive_model: Predictive,
        rng_key: jax.Array,
        *,
        ce_prefactor: float,
    ):
        subkeys = jax.random.split(rng_key, predictive_model.num_samples)
        l_mult = parameter_values["_l_mult"]

        def _x_sampler(pv: dict[str, npt.ArrayLike], key: jax.Array) -> float:
            return predictive_model(key, **pv)["UX"]

        def _ce(pv, subkeys):
            return (
                ce_prefactor
                * jax.vmap(_x_sampler, in_axes=(None, 0))(pv, subkeys).mean()
            )

        def _ux_sampler(pv: dict[str, npt.ArrayLike], key: jax.Array) -> float:
            return predictive_model(key, **pv)["X"]

        def _constraint(pv, subkeys):
            return (
                jnp.abs(
                    jax.vmap(_ux_sampler, in_axes=(None, 0))(pv, subkeys).mean()
                    - phi_observed
                )
                - epsilon
            )

        return _ce(parameter_values, subkeys) + l_mult * _constraint(
            parameter_values, subkeys
        )

    # In both cases, the Lagrange multiplier has the value 1.0 at the minimum.
    lambda_sol = 1.0
    ce_prefactor = 1.0 if not is_solving_max else -1.0
    mu_x_sol = phi_observed - ce_prefactor * epsilon

    # We'll be seeking stationary points of the Lagrangian, using the
    # naive approach of minimising the norm of its gradient. We will need to
    # ensure we "converge" to a minimum value suitably close to 0.
    def objective(params, predictive, key, ce_prefactor=ce_prefactor):
        v = jax.grad(lagrangian)(params, predictive, key, ce_prefactor=ce_prefactor)
        return sum(value**2 for value in v.values())

    # Choose a starting guess that is at the optimal solution, in the hopes that
    # SGD converges quickly. We almost certainly will not have this luxury in general.
    # The value of nu_x is free; the Lagrangian is independent of it.
    # As such, it can take any value and should not change during the optimisation
    # iterations.
    params = {
        "mean": mu_x_sol,
        "cov2": nu_x_starting_value,
        "_l_mult": lambda_sol,
    }
    # Setup SGD optimiser
    optimiser = optax.adam(adams_learning_rate)
    opt_state = optimiser.init(params)

    converged = False
    for _ in range(maxiter):
        # Actual iteration loop
        grads = jax.jacobian(objective)(
            params, predictive_model, rng_key, ce_prefactor=ce_prefactor
        )
        updates, opt_state = optimiser.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        # Convergence "check" and progress update
        objective_value = objective(
            params, predictive_model, rng_key, ce_prefactor=ce_prefactor
        )

        if jnp.abs(objective_value) <= minimisation_tolerance:
            converged = True
            break

    assert converged, f"Did not converge, final objective value: {objective_value}"

    # Confirm that we found a minimiser that does satisfy the inequality constraints.
    assert params["_l_mult"] > 0.0, (
        f"Converged, but not to a minimiser (lagrange multiplier = {params['_l_mult']})"
    )

    # Give a generous error margin in mu_ux and the Lagrange multiplier,
    # given SGD is being used on MC-integral functions.
    rtol = jnp.sqrt(1.0 / n_samples)
    assert jnp.isclose(params["mean"], mu_x_sol, rtol=rtol)
    assert jnp.isclose(params["_l_mult"], lagrange_mult_sol, atol=rtol)
