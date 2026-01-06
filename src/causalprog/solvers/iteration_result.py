"""Container classes for outputs from each iteration of solver methods."""

from dataclasses import dataclass, field

import numpy.typing as npt

from causalprog.utils.norms import PyTree


@dataclass(frozen=False, slots=True)
class IterationResult:
    """
    Result container for iterative solvers with optional history logging.

    Stores the latest iterate and if `history_logging_interval > 0`, `update` appends
    snapshots of the iterate to corresponding history lists each time the iteration
    number is a multiple of `history_logging_interval`.
    Instances are mutable but do not allow dynamic attribute creation.

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
        history_logging_interval: Interval at which to log history. If
            `history_logging_interval <= 0`, then no history is logged.

    """

    fn_args: PyTree
    grad_val: PyTree
    iters: int
    obj_val: npt.ArrayLike
    history_logging_interval: int = 0

    iter_history: list[int] = field(default_factory=list)
    fn_args_history: list[PyTree] = field(default_factory=list)
    grad_val_history: list[PyTree] = field(default_factory=list)
    obj_val_history: list[npt.ArrayLike] = field(default_factory=list)

    _log_enabled: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._log_enabled = self.history_logging_interval > 0

    def update(
        self,
        current_params: PyTree,
        gradient_value: PyTree,
        iters: int,
        objective_value: npt.ArrayLike,
    ) -> None:
        """
        Update the `IterationResult` object with current iteration data.

        Only updates the history if `history_logging_interval` is positive and
        the current iteration is a multiple of `history_logging_interval`.

        """
        self.fn_args = current_params
        self.grad_val = gradient_value
        self.iters = iters
        self.obj_val = objective_value

        if self._log_enabled and iters % self.history_logging_interval == 0:
            self.iter_history.append(iters)
            self.fn_args_history.append(current_params)
            self.grad_val_history.append(gradient_value)
            self.obj_val_history.append(objective_value)
