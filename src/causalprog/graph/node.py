"""Graph nodes."""

from __future__ import annotations

import typing
from abc import abstractmethod

import jax
import numpy as np
import numpyro

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
        is_outcome: bool = False,
        is_parameter: bool = False,
    ) -> None:
        """Initialise."""
        super().__init__(label=label)
        self._is_outcome = is_outcome
        self._is_parameter = is_parameter

    @abstractmethod
    def sample(
        self,
        sampled_dependencies: dict[str, npt.NDArray[float]],
        samples: int,
        rng_key: jax.Array,
    ) -> float:
        """Sample a value from the node."""

    @abstractmethod
    def copy(self) -> Node:
        """Make a copy of a node."""

    @property
    def is_outcome(self) -> bool:
        """Identify if the node is an outcome."""
        return self._is_outcome

    @property
    def is_parameter(self) -> bool:
        """Identify if the node is a parameter."""
        return self._is_parameter

    @property
    @abstractmethod
    def constant_parameters(self) -> dict[str, float]:
        """Named constants that this node depends on."""

    @property
    @abstractmethod
    def parameters(self) -> dict[str, str]:
        """Nodes that this node depends on."""


class DistributionNode(Node):
    """A node containing a distribution."""

    def __init__(
        self,
        distribution: DistributionFamily,
        label: str,
        *,
        parameters: dict[str, str] | None = None,
        constant_parameters: dict[str, float] | None = None,
        is_outcome: bool = False,
    ) -> None:
        """
        Initialise.

        NOTE: As of [#59](https://github.com/UCL/causalprog/pull/59),
        we will be committing to using Numpyro distributions for the
        foreseeable future. We will leave the backend-agnostic
        `DistributionFamily` class here as a type-hint (until it causes
        mypy issues), however code should only be assumed to work when
        `distribution` is passed a class from `numpyro.distributions`.
        """
        self._dist = distribution
        self._constant_parameters = constant_parameters if constant_parameters else {}
        self._parameters = parameters if parameters else {}
        super().__init__(label=label, is_outcome=is_outcome, is_parameter=False)

    def sample(
        self,
        sampled_dependencies: dict[str, npt.NDArray[float]],
        samples: int,
        rng_key: jax.Array,
    ) -> npt.NDArray[float]:
        """Sample a value from the node."""
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

    def copy(self) -> Node:
        """Make a copy of a node."""
        return DistributionNode(
            self._dist,
            label=self.label,
            parameters=dict(self._parameters),
            constant_parameters=dict(self._constant_parameters.items()),
            is_outcome=self.is_outcome,
        )

    def __repr__(self) -> str:
        return f'DistributionNode("{self.label}")'

    @property
    def constant_parameters(self) -> dict[str, float]:
        """Named constants that this node depends on."""
        return self._constant_parameters

    @property
    def parameters(self) -> dict[str, str]:
        """Nodes that this node depends on."""
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

    def __init__(
        self, label: str, *, value: float | None = None, is_outcome: bool = False
    ) -> None:
        """Initialise."""
        super().__init__(label=label, is_outcome=is_outcome, is_parameter=True)
        self.value = value

    def sample(
        self,
        sampled_dependencies: dict[str, npt.NDArray[float]],  # noqa: ARG002
        samples: int,
        rng_key: jax.Array,  # noqa: ARG002
    ) -> npt.NDArray[float]:
        """Sample a value from the node."""
        if self.value is None:
            msg = f"Cannot sample undetermined parameter node: {self.label}."
            raise ValueError(msg)
        return np.full(samples, self.value)

    def copy(self) -> Node:
        """Make a copy of a node."""
        return ParameterNode(
            label=self.label,
            value=self.value,
            is_outcome=self.is_outcome,
        )

    def __repr__(self) -> str:
        return f'ParameterNode("{self.label}")'

    @property
    def constant_parameters(self) -> dict[str, float]:
        """Named constants that this node depends on."""
        return {}

    @property
    def parameters(self) -> dict[str, str]:
        """Nodes that this node depends on."""
        return {}
