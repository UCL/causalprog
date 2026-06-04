"""Base quadrature class."""

from .base import QuadratureMethod

import quadraturerules
import typing
import numpy.typing as npt


class GaussianQuadrature(QuadratureMethod):
    """An abstract quadrature method."""

    def __init__(self, npoints: int):
        """Initialise.

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
        self._wts = 2 * wts

    def points_and_weights(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Get the quadrature points and weights."""
        return self._pts, self._wts
