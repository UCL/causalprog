"""Minimisation via Stochastic Gradient Descent."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy.typing as npt
import optax

from causalprog.utils.norms import PyTree, l2_normsq


def minimise(
    obj_fn: Callable,
    initial_guess: npt.ArrayLike,  # should be a pytree really
    *,
    convergence_criteria: Callable[[PyTree, PyTree], npt.ArrayLike] | None,
    fn_args: tuple | None = None,
    fn_kwargs: dict | None = None,
    learning_rate: float = 1.0e-1,
    maxiter: int = 100,
    optimiser: optax.GradientTransformationExtraArgs | None = None,
    tolerance: float = 1.0e-8,
) -> npt.ArrayLike:
    """Minimise a function of one argument using Stochastic Gradient Descent."""
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
        return convergence_criteria(x, dx) <= tolerance

    gradient = jax.grad(objective)

    opt_state = optimiser.init(initial_guess)

    current_params = initial_guess.copy(deep=True)
    for _ in range(maxiter):
        grads = gradient(current_params)
        updates, opt_state = optimiser.update(grads, opt_state)
        current_params = optax.apply_updates(current_params, updates)

        objective_value = objective(current_params)
        gradient_value = gradient(current_params)

        if is_converged(objective_value, gradient_value):
            return current_params

    msg = f"Did not converge after {_} iterations."
    raise RuntimeError(msg)
