"""Module for callback functions for solvers."""

from collections.abc import Callable

from tqdm.auto import tqdm

from causalprog.solvers.iteration_result import IterationResult


def _normalise_callbacks(
    callbacks: Callable[[IterationResult], None]
    | list[Callable[[IterationResult], None]]
    | None = None,
) -> list[Callable[[IterationResult], None]]:
    if callbacks is None:
        return []
    if callable(callbacks):
        return [callbacks]
    if isinstance(callbacks, list) and all(callable(cb) for cb in callbacks):
        return callbacks

    msg = "Callbacks must be a callable or a sequence of callables"
    raise TypeError(msg)


def _run_callbacks(
    iter_result: IterationResult,
    callbacks: list[Callable[[IterationResult], None]],
) -> None:
    for cb in callbacks:
        cb(iter_result)


def tqdm_callback(total: int) -> Callable[[IterationResult], None]:
    """
    Progress bar callback using `tqdm`.

    Creates a callback function that can be passed to solvers to display a progress bar
    during optimization. The progress bar updates based on the number of iterations and
    also displays the current objective value.

    Args:
        total: Total number of iterations for the progress bar.

    Returns:
        Callback function that updates the progress bar.

    """
    bar = tqdm(total=total)
    last_it = {"i": 0}

    def cb(ir: IterationResult) -> None:
        step = ir.iters - last_it["i"]
        if step > 0:
            bar.update(step)

            # Show objective and grad norm
            bar.set_postfix(obj=float(ir.obj_val))
            last_it["i"] = ir.iters

    return cb
