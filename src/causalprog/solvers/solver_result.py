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
    """

    successful: bool
    reason: str
    parameters: PyTree
    obj_val: npt.ArrayLike
    grad_val: PyTree
    iters: int
    maxiters: int
