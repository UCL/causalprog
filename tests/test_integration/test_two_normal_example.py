import sys
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
    two_normal_graph_parametrized_mean: Callable[[], Graph],
    adams_learning_rate: float = 1.0e-1,
    n_samples: int = 500,  # 1000 causes LLVM memory error... check cleanup of mem
    phi_observed: float = 0.0,  # The observed data
    epsilon: float = 1.0,  # The tolerance in the observed data
    nu_y_starting_value: float = 1.0,  # Where to start nu_y, the independent parameter
    lagrange_mult_sol: float = 1.0,  # Solution value of the lagrange multiplier
    maxiter: int = 100,  # Max iterations to allow (~100 sufficient for test cases)
    # Threshold for minimisation function value being considered 0
    minimisation_tolerance: float = 1.0e-6,
    *,
    is_solving_max: bool,
):
    """Solves the 'two normal' graph example problem.

    Assume we have the following model:
    mu_x -> X ~ N(mu_x, 1.0)
                |
                v
    nu_y -> Y ~ N(X, nu_y)

    and are interested in the causal estimand

    sigma(mu_x, nu_y) = E[Y] = mu_x,

    with constraints

    phi(mu_x, nu_y) = E[X] = mu_x.

    With observed data phi_observed, and tolerance in the data epsilon, we are
    effectively looking to solve the minimisation problem;

    min_{mu_x, nu_y} mu_x, subject to |mu_x - phi_observed| <= epsilon.

    The solution to this is mu_x^* = mu_x +/- phi_observed (+ in the maximisation case).
    The value of nu_y can be any positive value.

    The corresponding Lagrangian that we will form will be

    L(mu_x, nu_y, l_mult) = +/- mu_x + l_mult * (|mu_x - phi_observed| - epsilon)

    (again with + in the max case). In both cases, this is minimised at
    L(mu_x^*, nu_y, 1).
    """
    # Setup the optimisation problem from the graph
    g = two_normal_graph_parametrized_mean()
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
            return predictive_model(key, **pv)["X"]

        def _ce(pv, subkeys):
            return (
                ce_prefactor
                * jax.vmap(_x_sampler, in_axes=(None, 0))(pv, subkeys).mean()
            )

        def _ux_sampler(pv: dict[str, npt.ArrayLike], key: jax.Array) -> float:
            return predictive_model(key, **pv)["UX"]

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
    # The value of nu_y is free; the Lagrangian is independent of it.
    # As such, it can take any value and should not change during the optimisation
    # iterations.
    params = {
        "mu_x": mu_x_sol,
        "nu_y": nu_y_starting_value,
        "_l_mult": lambda_sol,
    }
    # Setup SGD optimiser
    optimiser = optax.adam(adams_learning_rate)
    opt_state = optimiser.init(params)

    converged = False
    for i in range(maxiter):
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

        sys.stdout.write(
            f"\n\t{i}, F_val={objective_value:.2e}, "
            f"mu_x={params['mu_x']:.3e}, l_mult={params['_l_mult']:.3e}"
        )

        if jnp.abs(objective_value) <= minimisation_tolerance:
            converged = True
            sys.stdout.write("CONVERGED - ")
            break

    sys.stdout.write("END ITERATIONS\n")

    # Confirm that nu_y has not changed, being an independent variable.
    assert jnp.isclose(nu_y_starting_value, params["nu_y"]), (
        "nu_y value has changed, despite gradient being independent of it"
    )
    assert converged, f"Did not converge, final objective value: {objective_value}"

    sys.stdout.write(
        f"Converged at: mu_x={params['mu_x']:.5e}, l_mult={params['_l_mult']:.5e}"
    )

    # Confirm that we found a minimiser that does satisfy the inequality constraints.
    assert params["_l_mult"] > 0.0, (
        f"Converged, but not to a minimiser (lagrange multiplier = {params['_l_mult']})"
    )

    # Give a generous error margin in mu_x and the Lagrange multiplier,
    # given SGD is being used on MC-integral functions.
    rtol = jnp.sqrt(1.0 / n_samples)
    assert jnp.isclose(params["mu_x"], mu_x_sol, rtol=rtol)
    assert jnp.isclose(params["_l_mult"], lagrange_mult_sol, atol=rtol)
