"""Container classes for outputs from each iteration of solver methods."""

from dataclasses import dataclass, field

import numpy.typing as npt

from causalprog.utils.norms import PyTree


@dataclass(frozen=False)
class IterationResult:
    """
    Container class storing state of solvers at iteration `iters`.

    Args:
        fn_args: Argument to the objective function at final iteration (the solution,
            if `successful is `True`).
        grad_val: Value of the gradient of the objective function at the `fn_args`.
        iters: Number of iterations performed.
        obj_val: Value of the objective function at `fn_args`.
        iter_history: List of iteration numbers at which history was logged.
        fn_args_history: List of `fn_args` at each logged iteration.
        grad_val_history: List of `grad_val` at each logged iteration.
        obj_val_history: List of `obj_val` at each logged iteration.

    """

    fn_args: PyTree
    grad_val: PyTree
    iters: int
    obj_val: npt.ArrayLike

    iter_history: list[int] = field(default_factory=list)
    fn_args_history: list[PyTree] = field(default_factory=list)
    grad_val_history: list[PyTree] = field(default_factory=list)
    obj_val_history: list[npt.ArrayLike] = field(default_factory=list)


def _update_iteration_result(
    iter_result: IterationResult,
    current_params: PyTree,
    gradient_value: PyTree,
    iters: int,
    objective_value: npt.ArrayLike,
    history_logging_interval: int,
) -> None:
    """
    Update the `IterationResult` object with current iteration data.

    Only updates the history if `history_logging_interval` is positive and
    the current iteration is a multiple of `history_logging_interval`.

    """
    iter_result.fn_args = current_params
    iter_result.grad_val = gradient_value
    iter_result.iters = iters
    iter_result.obj_val = objective_value

    if history_logging_interval > 0 and iters % history_logging_interval == 0:
        iter_result.iter_history.append(iters)
        iter_result.fn_args_history.append(current_params)
        iter_result.grad_val_history.append(gradient_value)
        iter_result.obj_val_history.append(objective_value)
