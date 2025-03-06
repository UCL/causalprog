"""Graph nodes."""

from __future__ import annotations
from typing import Protocol, runtime_checkable
from abc import abstractproperty

root_node_index = 0


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
        self, family: DistributionFamily, label: str = None, is_outcome: bool = False
    ):
        """Initialise the node."""
        global root_node_index
        self._dfamily = family

        if label is None:
            self._label = f"RootNode{root_node_index}"
            root_node_index += 1
        else:
            self._label = label
        self._outcome = is_outcome

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
