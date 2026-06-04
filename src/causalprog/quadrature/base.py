"""Base quadrature class."""

from abc import ABC, abstractmethod

import numpy.typing as npt


class QuadratureMethod(ABC):
    """An abstract quadrature method."""

    def __init__(self, npoints: int):
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
    def points_and_weights(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Get the quadrature points and weights."""
