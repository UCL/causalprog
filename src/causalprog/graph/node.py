"""Graph nodes."""

from __future__ import annotations

from abc import abstractproperty
from typing import Protocol, runtime_checkable


class DistributionFamily:
    """Placeholder class."""


class Distribution:
    """Placeholder class."""


@runtime_checkable
class Node(Protocol):
    """An abstract node in a graph."""

    @abstractproperty
    def label(self) -> str:
        """The label of the node."""

    @abstractproperty
    def is_root(self) -> bool:
        """Identify if the node is a root."""

    @abstractproperty
    def is_outcome(self) -> bool:
        """Identify if the node is an outcome."""

    @abstractproperty
    def is_intermediary(self) -> bool:
        """Identify if the node is an intermediary."""


class RootDistributionNode:
    """A root node containing a distribution family."""

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

    def __repr__(self) -> str:
        """Representation."""
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

    @property
    def is_intermediary(self) -> bool:
        """Identify if the node is an intermediary."""
        return False


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

    def __repr__(self) -> str:
        """Representation."""
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

    @property
    def is_intermediary(self) -> bool:
        """Identify if the node is an intermediary."""
        return not self._outcome
