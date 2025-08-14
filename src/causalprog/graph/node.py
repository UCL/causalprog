"""Graph nodes."""

from __future__ import annotations

import typing
from abc import abstractmethod

import numpy as np
import numpyro
from typing_extensions import override

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


class DistributionNode(Node):
    """A node containing a distribution."""

    def __init__(
        self,
        distribution: type,
        *,
        label: str,
        parameters: dict[str, str] | None = None,
        constant_parameters: dict[str, float] | None = None,
    ) -> None:
        """
        Initialise.

        Args:
            distribution: The distribution
            label: A unique label to identify the node
            parameters: A dictionary of parameters
            constant_parameters: A dictionary of constant parameters

        """
        self._dist = distribution
        self._constant_parameters = constant_parameters if constant_parameters else {}
        self._parameters = parameters if parameters else {}
        super().__init__(label=label, is_distribution=True)

    @override
    def sample(
        self,
        parameter_values: dict[str, float],
        sampled_dependencies: dict[str, npt.NDArray[float]],
        samples: int,
        *,
        rng_key: jax.Array,
    ) -> npt.NDArray[float]:
        d = self._dist(
            # Pass in node values derived from construction so far
            **{
                native_name: sampled_dependencies[node_name]
                for native_name, node_name in self.parameters.items()
            },
            # Pass in any constant parameters this node sets
            **self.constant_parameters,
        )
        return numpyro.sample(
            self.label,
            d,
            rng_key=rng_key,
            sample_shape=(samples,) if d.batch_shape == () and samples > 1 else (),
        )

    @override
    def copy(self) -> Node:
        return DistributionNode(
            self._dist,
            label=self.label,
            parameters=dict(self._parameters),
            constant_parameters=dict(self._constant_parameters.items()),
        )

    @override
    def __repr__(self) -> str:
        return f'DistributionNode("{self.label}")'

    @override
    @property
    def constant_parameters(self) -> dict[str, float]:
        return self._constant_parameters

    @override
    @property
    def parameters(self) -> dict[str, str]:
        return self._parameters

    def create_model_site(self, **dependent_nodes: jax.Array) -> npt.ArrayLike:
        """
        Create a model site for the (conditional) distribution attached to this node.

        `dependent_nodes` should contain keyword arguments mapping dependent node names
        to the values that those nodes are taking (`ParameterNode`s), or the sampling
        object for those nodes (`DistributionNode`s). These are passed to
        `self._dist` as keyword arguments to construct the sample-able object
        representing this node.
        """
        return numpyro.sample(
            self.label,
            self._dist(
                # Pass in node values derived from construction so far
                **{
                    native_name: dependent_nodes[node_name]
                    for native_name, node_name in self.parameters.items()
                },
                # Pass in any constant parameters this node sets
                **self.constant_parameters,
            ),
        )


class ParameterNode(Node):
    """
    A node containing a parameter.

    `ParameterNode`s differ from `DistributionNode`s in that they do not have an
    attached distribution, but rather represent a parameter that contributes
    to the shape of one (or more) `DistributionNode`s.

    The collection of parameters described by `ParameterNode`s forms the set of
    variables that will be optimised over in the corresponding `CausalProblem`.

    `ParameterNode`s should not be used to encode constant values used by
    `DistributionNode`s. Such constant values should be given to the necessary
    `DistributionNode`s directly as `constant_parameters`.
    """

    def __init__(self, *, label: str) -> None:
        """
        Initialise.

        Args:
            label: A unique label to identify the node

        """
        super().__init__(label=label, is_parameter=True)

    @override
    def sample(
        self,
        parameter_values: dict[str, float],
        sampled_dependencies: dict[str, npt.NDArray[float]],
        samples: int,
        *,
        rng_key: jax.Array,
    ) -> npt.NDArray[float]:
        if self.label not in parameter_values:
            msg = f"Missing input for parameter node: {self.label}."
            raise ValueError(msg)
        return np.full(samples, parameter_values[self.label])

    @override
    def copy(self) -> Node:
        return ParameterNode(label=self.label)

    @override
    def __repr__(self) -> str:
        return f'ParameterNode("{self.label}")'

    @override
    @property
    def constant_parameters(self) -> dict[str, float]:
        return {}

    @override
    @property
    def parameters(self) -> dict[str, str]:
        return {}
