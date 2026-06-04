"""Monte Carlo quadrature."""

import jax
import numpy as np
import numpy.typing as npt

from .base import QuadratureMethod


class MonteCarloGaussianQuadrature(QuadratureMethod):
    """Monte Carlo quadrature sampled from standard Gaussian."""

    def __init__(self, npoints: int, *, rng_key: jax.Array) -> None:
        """
        Initialise.

        Args:
            npoints: The number of quadrature points

        """
        super().__init__(npoints)
        self.rng_key = rng_key

    def points_and_weights(self) -> tuple[npt.NDArray, npt.NDArray]:
        """Get the quadrature points and weights."""
        return jax.random.normal(self.rng_key, (self.npoints,)), np.ones(
            self.npoints
        ) / self.npoints
