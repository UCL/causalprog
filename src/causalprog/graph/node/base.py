"""Base graph node."""

from __future__ import annotations

import typing
from abc import abstractmethod

if typing.TYPE_CHECKING:
    import jax
    import numpy.typing as npt

from causalprog._abc.labelled import Labelled


class Node(Labelled):
    """An abstract node in a graph."""

    def __init__(
        self,
        *,
        label: str,
        is_parameter: bool = False,
        is_distribution: bool = False,
    ) -> None:
        """
        Initialise.

        Parameters (equivalently `ParameterNode`s) represent Nodes that do not have
        random variables attached. Instead, these nodes represent values that are passed
        to nodes that _do_ have distributions attached, and the value of the "parameter"
        node is used as a fixed value when constructing the dependent node's
        distribution. The set of parameter nodes is the collection of "parameter"s over
        which one should want to optimise the causal estimand (subject to any
        constraints), and as such the value that a "parameter node" passes to its
        dependent nodes will vary as the optimiser runs and explores the solution space.

        Note that a "constant parameter" is distinct from a "parameter" in the sense
        that a constant parameter is _not_ added to the collection of parameters over
        which we will want to optimise (it is a hard-coded, fixed value).

        Distributions (equivalently `DistributionNode`s) are Nodes that represent
        random variables described by probability distributions.

        Args:
            label: A unique label to identify the node
            is_parameter: Is the node a parameter?
            is_distribution: Is the node a distribution?

        """
        super().__init__(label=label)
        self._is_parameter = is_parameter
        self._is_distribution = is_distribution

    @abstractmethod
    def sample(
        self,
        parameter_values: dict[str, float],
        sampled_dependencies: dict[str, npt.NDArray[float]],
        samples: int,
        *,
        rng_key: jax.Array,
    ) -> float:
        """
        Sample a value from the node.

        Args:
            parameter_values: Values to be taken by parameters
            sampled_dependencies: Values taken by dependencies of this node
            samples: Number of samples
            rng_key: Random key

        Returns:
            Sample value of this node

        """

    @abstractmethod
    def copy(self) -> Node:
        """
        Make a copy of a node.

        Some inner objects stored inside the node may not be copied when this is called.
        Modifying some inner objects of a copy made using this may affect the original
        node.

        Returns:
            A copy of the node

        """

    @property
    def is_parameter(self) -> bool:
        """
        Identify if the node is an parameter.

        Returns:
            True if the node is an parameter

        """
        return self._is_parameter

    @property
    def is_distribution(self) -> bool:
        """
        Identify if the node is an distribution.

        Returns:
            True if the node is an distribution

        """
        return self._is_distribution

    @property
    @abstractmethod
    def constant_parameters(self) -> dict[str, float]:
        """
        Named constants that this node depends on.

        Returns:
            A dictionary of the constant parameter names (keys) and their corresponding
            values

        """

    @property
    @abstractmethod
    def parameters(self) -> dict[str, str]:
        """
        Mapping of distribution parameter names to the nodes they are represented by.

        Returns:
            Mapping of distribution parameters (keys) to the corresponding label of the
            node that represents this parameter (value).

        """
