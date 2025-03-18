"""Graph nodes."""

from __future__ import annotations

from abc import abstractmethod
from typing import Protocol, runtime_checkable

from causalprog._abc.labelled import Labelled


class DistributionFamily:
    """Placeholder class."""


class Distribution:
    """Placeholder class."""


@runtime_checkable
class Node(Protocol):
    """An abstract node in a graph."""

    @property
    @abstractmethod
    def label(self) -> str:
        """The label of the node."""

    @property
    @abstractmethod
    def is_root(self) -> bool:
        """Identify if the node is a root."""

    @property
    @abstractmethod
    def is_outcome(self) -> bool:
        """Identify if the node is an outcome."""


class RootDistributionNode(Labelled):
    """A root node containing a distribution family."""

    def __init__(
        self,
        family: DistributionFamily,
        label: str,
        *,
        is_outcome: bool = False,
    ) -> None:
        """Initialise the node."""
        super().__init__(label=label)

        self._dfamily = family
        self._outcome = is_outcome

    def __repr__(self) -> str:
        """Representation."""
        return f'RootDistributionNode("{self._label}")'

    @property
    def is_root(self) -> bool:
        """Identify if the node is a root."""
        return True

    @property
    def is_outcome(self) -> bool:
        """Identify if the node is an outcome."""
        return self._outcome


class DistributionNode(Labelled):
    """A node containing a distribution family that depends on its parents."""

    def __init__(
        self,
        family: DistributionFamily,
        label: str,
        *,
        is_outcome: bool = False,
    ) -> None:
        """Initialise the node."""
        super().__init__(label=label)

        self._dfamily = family
        self._outcome = is_outcome

    def __repr__(self) -> str:
        """Representation."""
        return f'DistributionNode("{self._label}")'

    @property
    def is_root(self) -> bool:
        """Identify if the node is a root."""
        return False

    @property
    def is_outcome(self) -> bool:
        """Identify if the node is an outcome."""
        return self._outcome
