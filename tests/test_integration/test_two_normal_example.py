import sys
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy.typing as npt
import optax
from numpyro.infer import Predictive

from causalprog.graph import Graph


def test_two_normal_example(
    rng_key: jax.Array,
    two_normal_graph_parametrized_mean: Callable[[], Graph],
    n_samples: int = 500,  # 1000 causes LLVM memory error... check cleanup of mem
    phi_observed: float = 0.0,
    epsilon: float = 1.0,
    nu_y_starting_value: float = 1.0,
    lagrange_mult_sol: float = 1.0,  # Solution value of the lagrange multiplier
    maxiter: int = 200,
    minimisation_tolerance: float = 1.0e-6,
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

    The solution to this is mu_x^* = mu_x - phi_observed. The value of nu_y can be any
    positive value.

    The corresponding Lagrangian that we will form will be

    L(mu_x, nu_y, l_mult) = mu_x + l_mult * (|mu_x - phi_observed| - epsilon)

    which has stationary points when mu_x = mu_x^* and l_mult = +/ 1.

    TODO: solve max problem too....?
    """
    g = two_normal_graph_parametrized_mean()
    predictive_model = Predictive(g.model, num_samples=n_samples)

    def lagrangian(
        parameter_values: dict[str, npt.ArrayLike],
        predictive_model: Predictive,
        rng_key: jax.Array,
    ):
        subkeys = jax.random.split(rng_key, predictive_model.num_samples)
        l_mult = parameter_values["_l_mult"]

        def _x_sampler(pv: dict[str, npt.ArrayLike], key: jax.Array) -> float:
            return predictive_model(key, **pv)["X"]

        def _ce(pv, subkeys):
            return jax.vmap(_x_sampler, in_axes=(None, 0))(pv, subkeys).mean()

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

    # objective is euclidean norm of the gradient of the lagrangian
    def objective(params, predictive, key):
        v = jax.grad(lagrangian)(params, predictive, key)
        return sum(value**2 for value in v.values())

    # Try starting close to the optimal parameter values
    mu_x_sol = phi_observed - epsilon
    nu_y_sol = nu_y_starting_value  # nu_y is free - it does not affect the outcome
    lambda_sol = lagrange_mult_sol
    params = {
        "mu_x": mu_x_sol,
        "nu_y": nu_y_sol,
        "_l_mult": lambda_sol,
    }

    # Setup optimiser
    adams_learning_rate = 1.0e-1
    optimiser = optax.adam(adams_learning_rate)
    opt_state = optimiser.init(params)

    converged = False
    for i in range(maxiter):
        sys.stdout.write(f"{i}, ")
        # Actual iteration loop
        grads = jax.jacobian(objective)(params, predictive_model, rng_key)
        updates, opt_state = optimiser.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        # Early break if needed
        objective_value = objective(params, predictive_model, rng_key)
        if jnp.abs(objective_value) <= minimisation_tolerance:
            converged = True
            sys.stdout.write("CONVERGED - ")
            break
    sys.stdout.write("END ITERATIONS\n")

    assert jnp.isclose(nu_y_starting_value, params["nu_y"]), (
        "nu_y value has changed, despite gradient being independent of it"
    )
    assert converged, f"Did not converge, final objective value: {objective_value}"

    sys.stdout.write(
        f"Converged at: mu_x={params['mu_x']:.5e}, nu_y={params['nu_y']:.5e}"
    )

    assert params["_l_mult"] > 0.0, (
        f"Converged, but not to a minimiser (lagrange multiplier = {params['_l_mult']})"
    )
    rtol = jnp.sqrt(1.0 / n_samples)
    assert jnp.isclose(params["mu_x"], mu_x_sol, rtol=rtol)
    assert jnp.isclose(params["_l_mult"], lagrange_mult_sol, atol=rtol)
