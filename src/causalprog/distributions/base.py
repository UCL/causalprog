"""Base class for backend-agnostic distributions."""

from collections.abc import Callable
from typing import Generic, ParamSpec, TypeVar

SupportsSampling = TypeVar("SupportsSampling")
P = ParamSpec("P")


class Distribution(Generic[SupportsSampling]):
    """Distributions."""

    _dist: SupportsSampling

    def __init__(self, backend_distribution: SupportsSampling) -> None:
        """Create a new Distribution."""
        self._dist = backend_distribution

    def sample(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Draw samples from the distribution."""


class DistributionFamily(Generic[SupportsSampling, P]):
    """
    A family of distributions, specified by common parameters.

    Essentially a factory class for `Distribution`s.
    """

    _constructor: Callable[P, SupportsSampling]

    def __init__(self, distribution_constructor: Callable[P, SupportsSampling]) -> None:
        """Specify a family of Distributions."""
        self._constructor = distribution_constructor

    def __call__(self, *args, **kwargs) -> Distribution:  # noqa: ANN002, ANN003
        """Create a Distribution from parameters."""
        return Distribution(self._constructor(*args, **kwargs))
