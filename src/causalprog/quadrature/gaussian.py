"""Gaussian quadrature."""

import numpy.typing as npt
import quadraturerules

from .base import Integrand, IntegrandArgs, QuadratureMethod


class GaussianQuadrature(QuadratureMethod):
    r"""
    A Gaussian quadrature rule.

    The domain of integration for the points $p_i$ and weights $w_i$ is $[-1,1]$.
    This means that to integrate an integrand $f$ over the interval $[a,b]$, the
    approximation

    $$
    \int_a^b f(x) dx \approx
    \frac{b - a}{2}\sum_{w_i}f\left \frac{b - a}{2} p_i + \frac{b + a}{2}\right)
    $$

    is used.
    """

    def __init__(self, npoints: int) -> None:
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
        self._wts = wts * 2.0

    def integrate(
        self,
        integrand: Integrand,
        a: float = -1,
        b: float = 1,
        *integrand_args: IntegrandArgs.args,
        **integrand_kwargs: IntegrandArgs.kwargs,
    ) -> float:
        """Integrate the `integrand` over `[a,b]` via Gaussian quadrature."""
        change_of_vars_derivative = (b - a) / 2
        interval_midpoint = (b + a) / 2

        # Ideally, we would be able to assume that the integrand is vectorised
        # in it's first argument (Callable[[ArrayLike, ...], ArrayLike]).
        # Then we could do without the for loop here.
        result = 0.0
        for p_i, w_i in self.pts_wts_tuples():
            result += w_i * integrand(
                change_of_vars_derivative * p_i + interval_midpoint,
                *integrand_args,
                **integrand_kwargs,
            )

        return result * change_of_vars_derivative

    def points_and_weights(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Get the quadrature points and weights."""
        return self._pts, self._wts
