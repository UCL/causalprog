"""Minimisation via Stochastic Gradient Descent."""

from collections.abc import Callable
from copy import deepcopy

import jax
import jax.numpy as jnp
import optax
from jax.typing import ArrayLike

from causalprog._types import PyTree
from causalprog.solvers.iteration_result import IterationResult
from causalprog.solvers.solver_callbacks import _normalise_callbacks, _run_callbacks
from causalprog.solvers.solver_result import SolverResult
from causalprog.utils.norms import l2_normsq


def stochastic_gradient_descent(
    obj_fn: Callable[[PyTree], jax.Array],
    initial_guess: PyTree,
    *,
    convergence_criterion: Callable[[PyTree, PyTree], jax.Array] | None = None,
    fn_args: tuple = (),
    fn_kwargs: dict | None = None,
    learning_rate: float = 1.0e-1,
    maxiter: int = 1000,
    optimiser: optax.GradientTransformationExtraArgs | None = None,
    tolerance: float = 1.0e-8,
    history_logging_interval: int = -1,
    callbacks: Callable[[IterationResult], None]
    | list[Callable[[IterationResult], None]]
    | None = None,
) -> SolverResult:
    """
    Minimise a function of one argument using Stochastic Gradient Descent (SGD).

    The `obj_fn` provided will be minimised over its first argument. If you wish to
    minimise a function over a different argument, or multiple arguments, wrap it in a
    suitable `lambda` expression that has the correct call signature. For example, to
    minimise a function `f(x, y, z)` over `y` and `z`, use
    `g = lambda yz, x: f(x, yz[0], yz[1])`, and pass `g` in as `obj_fn`. Note that
    you will also need to provide a constant value for `x` via `fn_args` or `fn_kwargs`.

    The `fn_args` and `fn_kwargs` keys can be used to supply additional parameters that
    need to be passed to `obj_fn`, but which should be held constant.

    SGD terminates when the `convergence_criterion` is found to be smaller than the
    `tolerance`. That is, when
    `convergence_criterion(objective_value, gradient_value) <= tolerance` is found to
    be `True`, the algorithm considers a minimum to have been found. The default
    condition under which the algorithm terminates is when the norm of the gradient
    at the current argument value is smaller than the provided `tolerance`.

    The optimiser to use can be selected by passing in a suitable `optax` optimiser
    via the `optimiser` command. By default, `optax.adams` is used with the supplied
    `learning_rate`. Providing an explicit value for `optimiser` will result in the
    `learning_rate` argument being ignored.

    Args:
        obj_fn: Function to be minimised over its first argument.
        initial_guess: Initial guess for the minimising argument.
        convergence_criterion: The quantity that will be tested against `tolerance`, to
            determine whether the method has converged to a minimum. It should be a
            `callable` that takes the current value of `obj_fn` as its 1st argument, and
            the current value of the gradient of `obj_fn` as its 2nd argument. The
            default criterion is the l2-norm of the gradient.
        fn_args: Positional arguments to be passed to `obj_fn`, and held constant.
        fn_kwargs: Keyword arguments to be passed to `obj_fn`, and held constant.
        learning_rate: Default learning rate (or step size) to use when using the
            default `optimiser`. No effect if `optimiser` is provided explicitly.
        maxiter: Maximum number of iterations to perform. An error will be reported if
            this number of iterations is exceeded.
        optimiser: The `optax` optimiser to use during the update step.
        tolerance: `tolerance` used when determining if a minimum has been found.
        history_logging_interval: Interval (in number of iterations) at which to log
            the history of optimisation. If history_logging_interval <= 0, no
            history is logged.
        callbacks: A `callable` or list of `callables` that take an
            `IterationResult` as their only argument, and return `None`.
            These will be called at the end of each iteration of the optimisation
            procedure.


    Returns:
        SolverResult: Result of the optimisation procedure.

    """
    if fn_kwargs is None:
        fn_kwargs = {}
    if convergence_criterion is None:
        convergence_criterion = lambda _, dx: jnp.sqrt(l2_normsq(dx))  # noqa: E731
    if optimiser is None:
        optimiser = optax.adam(learning_rate)

    callbacks = _normalise_callbacks(callbacks)

    def objective(x: ArrayLike) -> jax.Array:
        return obj_fn(x, *fn_args, **fn_kwargs)

    def is_converged(x: ArrayLike, dx: ArrayLike) -> bool:
        return convergence_criterion(x, dx) < tolerance

    value_and_grad_fn = jax.jit(jax.value_and_grad(objective))

    # init state
    opt_state = optimiser.init(initial_guess)
    current_params = deepcopy(initial_guess)
    converged = False
    objective_value, gradient_value = value_and_grad_fn(current_params)

    iter_result = IterationResult(
        fn_args=current_params,
        grad_val=gradient_value,
        iters=0,
        obj_val=objective_value,
        history_logging_interval=history_logging_interval,
    )

    for current_iter in range(maxiter):
        iter_result.update(
            current_params=current_params,
            gradient_value=gradient_value,
            iters=current_iter,
            objective_value=objective_value,
        )

        _run_callbacks(iter_result, callbacks)

        if converged := is_converged(objective_value, gradient_value):
            break

        updates, opt_state = optimiser.update(gradient_value, opt_state)
        current_params = optax.apply_updates(current_params, updates)

        objective_value, gradient_value = value_and_grad_fn(current_params)

    iters_used = current_iter if converged else maxiter
    reason_msg = (
        f"Did not converge after {iters_used} iterations" if not converged else ""
    )

    return SolverResult(
        fn_args=current_params,
        grad_val=gradient_value,
        iters=iters_used,
        maxiter=maxiter,
        obj_val=objective_value,
        reason=reason_msg,
        successful=converged,
        iter_history=iter_result.iter_history,
        fn_args_history=iter_result.fn_args_history,
        grad_val_history=iter_result.grad_val_history,
        obj_val_history=iter_result.obj_val_history,
    )
