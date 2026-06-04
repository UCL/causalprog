"""Gaussian quadrature."""


import numpy.typing as npt
import quadraturerules

from .base import QuadratureMethod


class GaussianQuadrature(QuadratureMethod):
    """A Gaussian quadrature rule."""

    def __init__(self, npoints: int):
        """
        Initialise.

        Args:
            npoints: The number of quadrature points

        """
        super().__init__(npoints)
        pts, wts = quadraturerules.single_integral_quadrature(
            quadraturerules.QuadratureRule.GaussLegendre,
            quadraturerules.Domain.Interval,
            npoints,
        )
        self._pts = pts[:, 1] - pts[:, 0]
        self._wts = wts

    def points_and_weights(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Get the quadrature points and weights."""
        return self._pts, self._wts
