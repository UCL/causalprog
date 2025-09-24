"""Container class for outputs from solver methods."""

from dataclasses import dataclass

import numpy.typing as npt

from causalprog.utils.norms import PyTree


@dataclass(frozen=True)
class SolverResult:
    """
    Container class for outputs from solver methods.

    Instances of this class provide a container for useful information that
    comes out of running one of the solver methods on a causal problem.

    Attributes:
        fn_args: Argument to the objective function at final iteration (the solution,
            if `successful is `True`).
        grad_val: Value of the gradient of the objective function at the `fn_args`.
        iters: Number of iterations performed.
        maxiter: Maximum number of iterations the solver was permitted to perform.
        obj_val: Value of the objective function at `fn_args`.
        reason: Human-readable string explaining success or reasons for solver failure.
        successful: `True` if solver converged, in which case `fn_args` is the
            argument to the objective function at the solution of the problem being
            solved. `False` otherwise.

    """

    fn_args: PyTree
    grad_val: PyTree
    iters: int
    maxiter: int
    obj_val: npt.ArrayLike
    reason: str
    successful: bool
