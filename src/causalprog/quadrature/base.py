"""Base quadrature class."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Concatenate, ParamSpec, TypeAlias

import numpy.typing as npt
from jax import Array

IntegrandArgs = ParamSpec("IntegrandArgs")
Integrand: TypeAlias = Callable[Concatenate[float, IntegrandArgs], float]


class QuadratureMethod(ABC):
    """
    An abstract quadrature method.

    All `QuadratureMethod`s are required to provide a means of obtaining the
    points and weights that they use, accessible through the `points_and_weights`
    method of an instance.

    Instances also provide an `integrate` method, to perform
    numerical integration of an integrand.
    """

    def __init__(self, n_points: int) -> None:
        """
        Initialise.

        Args:
            npoints: The number of quadrature points

        """
        self._npts = n_points

    @property
    def n_points(self) -> int:
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

        Ideally, we would be able to assume that the integrand is vectorised
        in it's first argument (Callable[[ArrayLike, ...], ArrayLike]).
        Then we could do without the for-loop in each of the subclass implementations.
        """

    @abstractmethod
    def points_and_weights(
        self, a: float = -1.0, b: float = 1.0
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """Get quadrature points and weights for performing integration on $[a,b]$."""

    def pts_wts_tuples(
        self, a: float = -1.0, b: float = 1.0
    ) -> list[tuple[float, float]]:
        """Get `(point, weight)` pairs as a list of tuples."""
        return list(zip(*self.points_and_weights(a=a, b=b), strict=True))


class RNGQuadratureMethod(QuadratureMethod):
    """
    An abstract quadrature method, that relies on RNG.

    The only difference from the base `QuadratureMethod` class is the requirement
    that an `rng_key` be provided to the instance at creation.
    """

    rng_key: Array

    def __init__(self, n_points: int, *, rng_key: Array) -> None:
        """
        Initialise.

        Args:
            npoints: The number of quadrature points
            rng_key: PRNG key used for sample generation

        """
        super().__init__(n_points)

        self.rng_key = rng_key
