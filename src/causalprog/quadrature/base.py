"""Base quadrature class."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Concatenate, ParamSpec, TypeAlias

import numpy.typing as npt

IntegrandArgs = ParamSpec("IntegrandArgs")
Integrand: TypeAlias = Callable[Concatenate[float, IntegrandArgs], float]


class QuadratureMethod(ABC):
    """An abstract quadrature method."""

    def __init__(self, npoints: int) -> None:
        """
        Initialise.

        Args:
            npoints: The number of quadrature points

        """
        self._npts = npoints

    @property
    def npoints(self) -> int:
        """Number of quadrature points."""
        return self._npts

    @abstractmethod
    def integrate(
        self,
        integrand: Integrand,
        a: float = -1.0,
        b: float = 1.0,
        *integrand_args: IntegrandArgs.args,
        **integrand_kwargs: IntegrandArgs.kwargs,
    ) -> float:
        """
        Integrate the `integrand` over `[a,b]` using the `QuadratureMethod`.

        Subclasses should implement specific details.
        """

    @abstractmethod
    def points_and_weights(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Get the quadrature points and weights."""

    def pts_wts_tuples(self) -> list[tuple[float, float]]:
        """Get `(point, weight)` pairs as a list of tuples."""
        return list(zip(*self.points_and_weights(), strict=True))
