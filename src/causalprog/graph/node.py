"""Graph nodes."""

from __future__ import annotations

import typing
from abc import ABC, abstractmethod

import numpy as np

if typing.TYPE_CHECKING:
    import numpy.typing as npt


class Distribution(ABC):
    """Placeholder class."""

    @abstractmethod
    def sample(
        self, sampled_dependencies: dict[str, npt.NDArray[float]], samples: int
    ) -> npt.NDArray[float]:
        """Sample."""


class NormalDistribution(Distribution):
    """Normal distribution."""

    def __init__(self, mean: str | float = 0.0, std_dev: str | float = 1.0) -> None:
        """Initialise."""
        self.mean = mean
        self.std_dev = std_dev

    def sample(
        self, sampled_dependencies: dict[str, npt.NDArray[float]], samples: int
    ) -> npt.NDArray[float]:
        """Sample a normal distribution with mean 1."""
        values = np.random.normal(0.0, 1.0, samples)  # noqa: NPY002
        if isinstance(self.std_dev, str):
            values *= sampled_dependencies[self.std_dev]
        else:
            values *= self.std_dev
        if isinstance(self.mean, str):
            values += sampled_dependencies[self.mean]
        else:
            values += self.mean
        return values


class Node(ABC):
    """An abstract node in a graph."""

    def __init__(self, label: str | None, *, is_outcome: bool) -> None:
        """Initialise."""
        self._label = label
        self._is_outcome = is_outcome

    @property
    def label(self) -> str:
        """The label of the node."""
        if self._label is None:
            msg = "Node has no label."
            raise ValueError(msg)
        return self._label

    @abstractmethod
    def sample(
        self, sampled_dependencies: dict[str, npt.NDArray[float]], samples: int
    ) -> float:
        """Sample a value from the node."""

    @property
    def is_outcome(self) -> bool:
        """Identify if the node is an outcome."""
        return self._is_outcome


class DistributionNode(Node):
    """A node containing a distribution."""

    def __init__(
        self,
        distribution: Distribution,
        label: str | None = None,
        *,
        is_outcome: bool = False,
    ) -> None:
        """Initialise."""
        self._dist = distribution
        super().__init__(label, is_outcome=is_outcome)

    def sample(
        self, sampled_dependencies: dict[str, npt.NDArray[float]], samples: int
    ) -> float:
        """Sample a value from the node."""
        return self._dist.sample(sampled_dependencies, samples)

    def __repr__(self) -> str:
        return f'DistributionNode("{self.label}")'
