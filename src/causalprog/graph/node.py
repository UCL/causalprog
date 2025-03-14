"""Graph nodes."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class DistributionFamily:
    """Placeholder class."""


class Distribution:
    """Placeholder class."""

    def sample(self) -> float:
        """Sample a normal distribution with mean 1."""
        return np.random.normal(1.0)  # noqa: NPY002


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
    def sample(self, sampled_dependencies: dict[str, float]) -> float:
        """Sample a value from the node."""

    @property
    @abstractmethod
    def is_root(self) -> bool:
        """Identify if the node is a root."""

    @property
    def is_outcome(self) -> bool:
        """Identify if the node is an outcome."""
        return self._is_outcome


class RootDistributionNode(Node):
    """A root node containing a distribution family."""

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

    def sample(self, _sampled_dependencies: dict[str, float]) -> float:
        """Sample a value from the node."""
        return self._dist.sample()

    def __repr__(self) -> str:
        return f'RootDistributionNode("{self.label}")'

    @property
    def is_root(self) -> bool:
        """Identify if the node is a root."""
        return True


class DistributionNode(Node):
    """A node containing a distribution family that depends on its parents."""

    def __init__(
        self,
        family: DistributionFamily,
        label: str | None = None,
        *,
        is_outcome: bool = False,
    ) -> None:
        """Initialise."""
        self._dfamily = family
        super().__init__(label, is_outcome=is_outcome)

    def sample(self, sampled_dependencies: dict[str, float]) -> float:
        """Sample a value from the node."""
        raise NotImplementedError

    def __repr__(self) -> str:
        return f'DistributionNode("{self.label}")'

    @property
    def is_root(self) -> bool:
        """Identify if the node is a root."""
        return False
