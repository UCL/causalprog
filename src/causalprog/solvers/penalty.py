"""Penalty method solvers."""

from collections.abc import Callable
from copy import deepcopy
from typing import Literal

import jax
import jax.numpy as jnp

from causalprog._types import PyTree
from causalprog.solvers.iteration_result import IterationResult
from causalprog.solvers.sgd import stochastic_gradient_descent
from causalprog.solvers.solver_callbacks import _normalise_callbacks, _run_callbacks
from causalprog.solvers.solver_result import SolverResult
from causalprog.utils.norms import l2_normsq


def penalty_method(
    obj_fn: Callable[[PyTree], jax.Array],
    initial_guess: PyTree,
    bounds: Callable[[PyTree], jax.Array],
    *,
    initial_mu: float = 1.0,
    update_mu: Callable[[float], float] = lambda mu: 10 * mu,
    initial_learning_rate: float = 0.1,
    update_learning_rate: Callable[[float, float], float] = lambda _, mu: 1.0 / mu,
    bounds_epsilon: float = 0.0,
    convergence_criterion: Callable[[PyTree, PyTree], jax.Array] | None = None,
    fn_args: tuple = (),
    fn_kwargs: dict | None = None,
    max_or_min: Literal["max", "min"] = "min",
    maxiter: int = 10,
    tolerance: float = 1.0e-8,
    history_logging_interval: int = -1,
    callbacks: Callable[[IterationResult], None]
    | list[Callable[[IterationResult], None]]
    | None = None,
) -> SolverResult:
    """
    Minimise a function using a penalty method.

    Args:
        obj_fn: Function to minimise
        initial_guess: An inital guess for the solution
        bounds: Function that evaluates bounds on the minimisation problem
        bounds_epsilon: Value of epsilon to use for the bounds. Any value smaller than
                        this will be treated as equal to 0.
        maxiter: Maximum number of iterations
        initial_mu: Starting value for mu
        update_mu: Function to update mu after each gradient descent solve
        convergence_criterion: The quantity that will be tested against `tolerance`, to
            determine whether the method has converged to a minimum. It should be a
            `callable` that takes the current value of `obj_fn` as its first argument
            and the solution at the previous iteration as its second argument. The
            default criterion is the l2-norm of the difference between the two
            solutions.
        initial_learning_rate: Learning rate to use in the first gradient descent solve
        update_learning_rate: Function to update the learning rate after each gradient
                              descent solve. Should take 2 positional arguments; the
                              current learning rate and the value of mu to be used in
                              the next iteration, in that order.
        fn_args: Positional arguments to be passed to `obj_fn`, and held constant.
        fn_kwargs: Keyword arguments to be passed to `obj_fn`, and held constant.
        max_or_min: Whether to minimise or maximise `obj_fn`.
        maxiter: Maximum number of iterations to perform. An error will be reported if
            this number of iterations is exceeded.
        tolerance: `tolerance` used when determining if a minimum has been found.
        history_logging_interval: Interval (in number of iterations) at which to log
            the history of optimisation. If history_logging_interval <= 0, no
            history is logged.
        callbacks: A `callable` or list of `callables` that take an
            `IterationResult` as their only argument, and return `None`.
            These will be called at the end of each iteration of the optimisation
            procedure.

    Returns:
        Result of the optimisation procedure.

    """
    if fn_kwargs is None:
        fn_kwargs = {}
    if convergence_criterion is None:
        convergence_criterion = lambda a, b: jnp.sqrt(  # noqa: E731
            sum(l2_normsq(b[i] - a[i]) for i in b)
        )
    obj_prefactor = -1.0 if max_or_min == "max" else 1.0

    if bounds_epsilon < 0:
        msg = "Epsilon cannot be negative."
        raise ValueError(msg)

    def evaluate_bounds(guess: dict[str, jax.Array]) -> jax.Array:
        return jnp.maximum(bounds(guess, *fn_args, **fn_kwargs) - bounds_epsilon, 0.0)

    callbacks = _normalise_callbacks(callbacks)

    def evaluate_obj_fun(x: PyTree) -> jax.Array:
        return obj_fn(x, *fn_args, **fn_kwargs)

    def objective(x: PyTree, mu: float) -> jax.Array:
        bound_values = evaluate_bounds(x)
        return obj_prefactor * evaluate_obj_fun(x) + mu / 2 * jnp.dot(
            bound_values, bound_values
        )

    def is_converged(x: PyTree, dx: PyTree) -> bool:
        return convergence_criterion(x, dx) < tolerance

    mu = initial_mu
    current_solution = deepcopy(initial_guess)
    learning_rate = initial_learning_rate

    iter_result = IterationResult(
        fn_args=current_solution,
        iters=0,
        obj_val=evaluate_obj_fun(current_solution),
        history_logging_interval=history_logging_interval,
    )

    for current_iter in range(maxiter):
        previous_solution = current_solution
        current_solution = stochastic_gradient_descent(
            objective,
            current_solution,
            fn_kwargs={"mu": mu},
            learning_rate=learning_rate,
        ).fn_args
        mu = update_mu(mu)
        learning_rate = update_learning_rate(learning_rate, mu)

        iter_result.update(
            current_params=current_solution,
            iters=current_iter,
            objective_value=evaluate_obj_fun(current_solution),
        )

        _run_callbacks(iter_result, callbacks)

        if converged := is_converged(current_solution, previous_solution):
            break

    iters_used = current_iter
    reason_msg = (
        f"Did not converge after {iters_used} iterations" if not converged else ""
    )

    return SolverResult(
        fn_args=current_solution,
        iters=iters_used,
        maxiter=maxiter,
        obj_val=evaluate_obj_fun(current_solution),
        reason=reason_msg,
        successful=converged,
        iter_history=iter_result.iter_history,
        fn_args_history=iter_result.fn_args_history,
        obj_val_history=iter_result.obj_val_history,
    )
