"""Graph nodes."""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable

import numpy as np


class DistributionFamily:
    """Placeholder class."""


class Distribution:
    """Placeholder class."""

    def sample(self) -> float:
        """Sample a normal distribution with mean 1."""
        return np.random.normal(1.0)  # noqa: NPY002


@runtime_checkable
class Node(Protocol):
    """An abstract node in a graph."""

    @property
    @abstractmethod
    def label(self) -> str:
        """The label of the node."""

    @abstractmethod
    def sample(self, sampled_dependencies: dict[str, float]) -> float:
        """Sample a value from the node."""

    @property
    @abstractmethod
    def is_root(self) -> bool:
        """Identify if the node is a root."""

    @property
    @abstractmethod
    def is_outcome(self) -> bool:
        """Identify if the node is an outcome."""


class RootDistributionNode:
    """A root node containing a distribution family."""

    def __init__(
        self,
        distribution: Distribution,
        label: str,
        *,
        is_outcome: bool = False,
    ) -> None:
        """Initialise the node."""
        self._dist = distribution
        self._label = label
        self._outcome = is_outcome

    def sample(self, _sampled_dependencies: dict[str, float]) -> float:
        """Sample a value from the node."""
        return self._dist.sample()

    def __repr__(self) -> str:
        return f'RootDistributionNode("{self._label}")'

    @property
    def label(self) -> str:
        """The label of the node."""
        return self._label

    @property
    def is_root(self) -> bool:
        """Identify if the node is a root."""
        return True

    @property
    def is_outcome(self) -> bool:
        """Identify if the node is an outcome."""
        return self._outcome


class DistributionNode:
    """A node containing a distribution family that depends on its parents."""

    def __init__(
        self,
        family: DistributionFamily,
        label: str,
        *,
        is_outcome: bool = False,
    ) -> None:
        """Initialise the node."""
        self._dfamily = family
        self._label = label
        self._outcome = is_outcome

    def sample(self, sampled_dependencies: dict[str, float]) -> float:
        """Sample a value from the node."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'DistributionNode("{self._label}")'

    @property
    def label(self) -> str:
        """The label of the node."""
        return self._label

    @property
    def is_root(self) -> bool:
        """Identify if the node is a root."""
        return False

    @property
    def is_outcome(self) -> bool:
        """Identify if the node is an outcome."""
        return self._outcome
