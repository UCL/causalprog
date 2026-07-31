"""Augmented Lagrangian solvers."""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from causalprog.solvers.sgd import stochastic_gradient_descent
from causalprog.solvers.solver_result import SolverResult


def minimise(
    f: Callable[[dict[str, jax.Array], dict[str, jax.Array]], jax.Array],
    bounds: Callable[[dict[str, jax.Array], dict[str, jax.Array]], jax.Array],
    *,
    bounds_epsilon: float | None = None,
    initial_guess: dict[str, jax.Array] | None = None,
    variables: list[str] | None = None,
    n_iter: int = 10,
    initial_mu: float = 1.0,
    update_mu: Callable[[float], float] = lambda mu: 10 * mu,
    initial_learning_rate: float = 0.1,
    update_learning_rate: Callable[[float, float, float], float] = lambda lr, mu, _: (
        lr / mu
    ),
    parameter_values: dict[str, jax.Array] | None = None,
) -> SolverResult:
    """
    Augmented Lagrangian minimisation solver.

    Args:
        f: Function to minimise
        bounds: Function that evaluates bounds on the minimisation problem
        bounds_epsilon: Value of epsilon to use for the bounds. Any value smaller than
                        this will be treated as equal to 0.
        initial_guess: An inital guess for the solution
        variables: List of variable names to include in solution
        n_iter: Number of iterations
        initial_mu: Starting value for mu
        update_mu: Function to update mu after each gradient descent solve
        initial_learning_rate: Learning rate for the 1st stochastic gradient descent
        update_learning_rate: Function to update the learning rate after each gradient
                              descent solve. Takes 3 positional arguments corresponding
                              to the current values of the learning rate, mu, and lamb,
                              in that order.
        parameter_values: Parameter values to pass into the f and bounds functions

    """
    if parameter_values is None:
        parameter_values = {}
    if initial_guess is None:
        if variables is None:
            msg = (
                "List of variable names must be provided if initial guess not provided."
            )
            raise ValueError(msg)
        initial_guess = dict.fromkeys(variables, 0.0)
    elif variables is not None:
        if set(variables) != set(initial_guess.keys()):
            msg = "List of variable names must match variables in initial guess."
            raise ValueError(msg)
    else:
        variables = list(initial_guess.keys())

    if bounds_epsilon is None:

        def bounds_(
            params: dict[str, jax.Array], guess: dict[str, jax.Array]
        ) -> jax.Array:
            return jnp.maximum(bounds(params, guess), 0.0)
    else:

        def bounds_(
            params: dict[str, jax.Array], guess: dict[str, jax.Array]
        ) -> jax.Array:
            return jnp.maximum(bounds(params, guess) - bounds_epsilon, 0.0)

    mu = initial_mu
    lamb = jnp.zeros_like(bounds(parameter_values, initial_guess))
    learning_rate = initial_learning_rate

    for _ in range(n_iter):

        def fun(
            s: dict[str, jax.Array],
            mu: float = mu,
            lamb: float = lamb,
        ) -> jax.Array:
            bound_values = bounds_(parameter_values, s)
            return (
                f(parameter_values, s)
                + mu / 2 * jnp.dot(bound_values, bound_values)
                + jnp.dot(lamb, bound_values)
            )

        solution = stochastic_gradient_descent(
            fun, initial_guess, learning_rate=learning_rate
        )

        lamb += mu * bounds_(parameter_values, solution.fn_args)
        mu = update_mu(mu)
        learning_rate = update_learning_rate(learning_rate, mu, lamb)

    return solution


def maximise(
    f: Callable[[dict[str, jax.Array], dict[str, jax.Array]], jax.Array],
    *args,
    **kwargs,
) -> dict[str, jax.Array]:
    """
    Augmented Lagrangian maximisation solver.

    Args:
        f: Function to minimise

    """
    return minimise(
        lambda a, b: -f(a, b),
        *args,
        **kwargs,
    )
