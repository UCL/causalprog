"""Base class for backend-agnostic distributions."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

SupportsSampling = TypeVar("SupportsSampling")


class Distribution(ABC, Generic[SupportsSampling]):
    """Distributions."""

    _dist: SupportsSampling

    def __init__(self, backend_distribution: SupportsSampling) -> None:
        """Create a new Distribution."""
        self._dist = backend_distribution

    @abstractmethod
    def sample(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        """Draw samples from the distribution."""
