"""Minimisation via Stochastic Gradient Descent."""

from collections.abc import Callable
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy.typing as npt
import optax

from causalprog.solvers.solver_result import SolverResult
from causalprog.utils.norms import PyTree, l2_normsq


def stochastic_gradient_descent(
    obj_fn: Callable[[PyTree], npt.ArrayLike],
    initial_guess: PyTree,
    *,
    convergence_criteria: Callable[[PyTree, PyTree], npt.ArrayLike] | None = None,
    fn_args: tuple | None = None,
    fn_kwargs: dict | None = None,
    learning_rate: float = 1.0e-1,
    maxiter: int = 1000,
    optimiser: optax.GradientTransformationExtraArgs | None = None,
    tolerance: float = 1.0e-8,
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

    SGD terminates when the `convergence_criteria` is found to be smaller than the
    `tolerance`. That is, when
    `convergence_criteria(objective_value, gradient_value) <= tolerance` is found to
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
        convergence_criteria: The quantity that will be tested against `tolerance`, to
            determine whether the method has converged to a minimum. It should be a
            `callable` that takes the current value of `obj_fn` as its 1st argument, and
            the current value of the gradient of `obj_fn` as its 2nd argument. The
            default criteria is the l2-norm of the gradient.
        fn_args: Positional arguments to be passed to `obj_fn`, and held constant.
        fn_kwargs: Keyword arguments to be passed to `obj_fn`, and held constant.
        learning_rate: Default learning rate (or step size) to use when using the
            default `optimiser`. No effect if `optimiser` is provided explicitly.
        maxiter: Maximum number of iterations to perform. An error will be reported if
            this number of iterations is exceeded.
        optimiser: The `optax` optimiser to use during the update step.
        tolerance: `tolerance` used when determining if a minimum has been found.

    Returns:
        Minimising argument of `obj_fn`.
        Value of `obj_fn` at the minimum.
        Gradient of `obj_fn` at the minimum.
        Number of iterations performed.

    """
    if not fn_args:
        fn_args = ()
    if not fn_kwargs:
        fn_kwargs = {}
    if not convergence_criteria:
        convergence_criteria = lambda _, dx: jnp.sqrt(l2_normsq(dx))  # noqa: E731
    if not optimiser:
        optimiser = optax.adam(learning_rate)

    def objective(x: npt.ArrayLike) -> npt.ArrayLike:
        return obj_fn(x, *fn_args, **fn_kwargs)

    def is_converged(x: npt.ArrayLike, dx: npt.ArrayLike) -> bool:
        return convergence_criteria(x, dx) < tolerance

    converged = False

    opt_state = optimiser.init(initial_guess)
    current_params = deepcopy(initial_guess)
    gradient = jax.grad(objective)

    for _ in range(maxiter + 1):
        objective_value = objective(current_params)
        gradient_value = gradient(current_params)

        if converged := is_converged(objective_value, gradient_value):
            break

        updates, opt_state = optimiser.update(gradient_value, opt_state)
        current_params = optax.apply_updates(current_params, updates)

    iters_used = _
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
    )
