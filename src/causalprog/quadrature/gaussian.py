"""Gaussian quadrature."""

import numpy.typing as npt
import quadraturerules
from typing_extensions import override

from .base import Integrand, IntegrandArgs, QuadratureMethod


class GaussianQuadrature(QuadratureMethod):
    r"""
    A Gaussian quadrature rule.

    The domain of integration for the points $p_i$ and weights $w_i$ is $[-1,1]$.
    This means that to integrate an integrand $f$ over the interval $[a,b]$, the
    approximation

    $$
    \int_a^b f(x) dx \approx
    \frac{b - a}{2}\sum_{w_i}f\left( \frac{b - a}{2} p_i + \frac{b + a}{2}\right)
    $$

    is used.
    """

    _pts: npt.NDArray
    _wts: npt.NDArray

    @override
    def __init__(self, n_points: int) -> None:
        super().__init__(n_points)
        pts, wts = quadraturerules.single_integral_quadrature(
            quadraturerules.QuadratureRule.GaussLegendre,
            quadraturerules.Domain.Interval,
            n_points,
        )
        self._pts = pts[:, 1] - pts[:, 0]
        self._wts = wts * 2.0

    def integrate(
        self,
        integrand: Integrand,
        a: float = -1,
        b: float = 1,
        *integrand_args: IntegrandArgs.args,
        **integrand_kwargs: IntegrandArgs.kwargs,
    ) -> float:
        """Integrate the `integrand` over $[a,b]$ via Gaussian quadrature."""
        result = 0.0
        for p_i, w_i in self.pts_wts_tuples(a=a, b=b):
            result += w_i * integrand(
                p_i,
                *integrand_args,
                **integrand_kwargs,
            )

        return result * (b - a) / 2

    def points_and_weights(
        self, a: float = -1.0, b: float = 1.0
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Get quadrature points and weights for performing integration on $[a,b]$."""
        change_of_vars_derivative = (b - a) / 2.0
        interval_midpoint = (b + a) / 2.0
        return self._pts * change_of_vars_derivative + interval_midpoint, self._wts
