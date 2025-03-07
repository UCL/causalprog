"""Graph nodes."""

from __future__ import annotations
from typing import Protocol, runtime_checkable
from abc import abstractproperty


class DistributionFamily:  # TODO: import from elsewhere once it exists
    """Placeholder class."""


class Distribution:  # TODO: import from elsewhere once it exists
    """Placeholder class."""


@runtime_checkable
class Node(Protocol):
    """An abstract node in a graph."""

    @abstractproperty
    def label(self) -> str:
        """The label of the node."""

    @abstractproperty
    def is_root(self) -> bool:
        """Is this node a root?"""

    @abstractproperty
    def is_outcome(self) -> bool:
        """Is this node an outcome?"""

    @abstractproperty
    def is_intermediary(self) -> bool:
        """Is this node an intermediary?"""


class RootDistributionNode(object):
    """A root node containing a distribution family."""

    def __init__(
        self,
        family: DistributionFamily,
        label: str,
        is_outcome: bool = False,
    ):
        """Initialise the node."""
        self._dfamily = family
        self._label = label
        self._outcome = is_outcome

    def __repr__(self) -> str:
        return f'RootDistributionNode("{self._label}")'

    @property
    def label(self) -> str:
        """The label of the node."""
        return self._label

    @property
    def is_root(self) -> bool:
        """Is this node a root?"""
        return True

    @property
    def is_outcome(self) -> bool:
        """Is this node an outcome?"""
        return self._outcome

    @property
    def is_intermediary(self) -> bool:
        """Is this node an intermediary?"""
        return False


class DistributionNode(object):
    """A node containing a distribution family that depends on its parents."""

    def __init__(
        self,
        family: DistributionFamily,
        label: str,
        is_outcome: bool = False,
    ):
        """Initialise the node."""
        self._dfamily = family
        self._label = label
        self._outcome = is_outcome

    def __repr__(self) -> str:
        return f'DistributionNode("{self._label}")'

    @property
    def label(self) -> str:
        """The label of the node."""
        return self._label

    @property
    def is_root(self) -> bool:
        """Is this node a root?"""
        return False

    @property
    def is_outcome(self) -> bool:
        """Is this node an outcome?"""
        return self._outcome

    @property
    def is_intermediary(self) -> bool:
        """Is this node an intermediary?"""
        return not self._outcome
