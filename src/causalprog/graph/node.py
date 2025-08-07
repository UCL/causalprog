"""Graph nodes."""

from __future__ import annotations

import typing
from abc import abstractmethod

import jax
import numpy as np
import numpyro
from typing_extensions import override

if typing.TYPE_CHECKING:
    import numpy.typing as npt

    from causalprog.distribution.family import DistributionFamily

from causalprog._abc.labelled import Labelled


class Node(Labelled):
    """An abstract node in a graph."""

    def __init__(
        self,
        *,
        label: str,
        is_parameter: bool = False,
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

        Args:
            label: A unique label to identify the node
            is_parameter: Is the node a parameter?

        """
        super().__init__(label=label)
        self._is_parameter = is_parameter

    @abstractmethod
    def sample(
        self,
        sampled_dependencies: dict[str, npt.NDArray[float]],
        samples: int,
        rng_key: jax.Array,
    ) -> float:
        """
        Sample a value from the node.

        Args:
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
        distribution: DistributionFamily,
        *,
        label: str,
        parameters: dict[str, str] | None = None,
        constant_parameters: dict[str, float] | None = None,
    ) -> None:
        """
        Initialise.

        NOTE: As of [#59](https://github.com/UCL/causalprog/pull/59),
        we will be committing to using Numpyro distributions for the
        foreseeable future. We will leave the backend-agnostic
        `DistributionFamily` class here as a type-hint (until it causes
        mypy issues), however code should only be assumed to work when
        `distribution` is passed a class from `numpyro.distributions`.

        Args:
            distribution: The distribution
            label: A unique label to identify the node
            parameters: A dictionary of parameters
            constant_parameters: A dictionary of constant parameters

        """
        self._dist = distribution
        self._constant_parameters = constant_parameters if constant_parameters else {}
        self._parameters = parameters if parameters else {}
        super().__init__(label=label, is_parameter=False)

    @override
    def sample(
        self,
        sampled_dependencies: dict[str, npt.NDArray[float]],
        samples: int,
        rng_key: jax.Array,
    ) -> npt.NDArray[float]:
        if not self._parameters:
            concrete_dist = self._dist.construct(**self._constant_parameters)
            return concrete_dist.sample(rng_key, samples)
        output = np.zeros(samples)
        new_key = jax.random.split(rng_key, samples)
        for sample in range(samples):
            parameters = {
                i: sampled_dependencies[j][sample] for i, j in self._parameters.items()
            }
            concrete_dist = self._dist.construct(
                **parameters, **self._constant_parameters
            )
            output[sample] = concrete_dist.sample(new_key[sample], 1)[0][0]
        return output

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
    attached distribution (family), but rather represent a parameter that contributes
    to the shape of one (or more) `DistributionNode`s.

    The collection of parameters described by `ParameterNode`s forms the set of
    variables that will be optimised over in the corresponding `CausalProblem`.
    `ParameterNode`s have a `.value` attribute which stores the current value
    of the parameter to facilitate this - a `CausalProblem` needs to be able to
    update the values of the parameters so it can make evaluations of the causal
    estimand and constraints functions, _as if_ they were functions of the parameters,
    rather than the `DistributionNode`s.

    `ParameterNode`s should not be used to encode constant values used by
    `DistributionNode`s. Such constant values should be given to the necessary
    `DistributionNode`s directly as `constant_parameters`.
    """

    def __init__(self, *, label: str, value: float | None = None) -> None:
        """
        Initialise.

        Args:
            label: A unique label to identify the node
            value: The value taken by this parameter

        """
        super().__init__(label=label, is_parameter=True)
        self.value = value

    @override
    def sample(
        self,
        sampled_dependencies: dict[str, npt.NDArray[float]],
        samples: int,
        rng_key: jax.Array,
    ) -> npt.NDArray[float]:
        if self.value is None:
            msg = f"Cannot sample undetermined parameter node: {self.label}."
            raise ValueError(msg)
        return np.full(samples, self.value)

    @override
    def copy(self) -> Node:
        return ParameterNode(
            label=self.label,
            value=self.value,
        )

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
