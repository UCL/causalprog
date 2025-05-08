"""Graph nodes."""

from __future__ import annotations

import typing
from abc import abstractmethod

import jax
import numpy as np

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

    @property
    def is_outcome(self) -> bool:
        """Identify if the node is an outcome."""
        return self._is_outcome

    @property
    def is_parameter(self) -> bool:
        """Identify if the node is a parameter."""
        return self._is_parameter


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
        """Initialise."""
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

    def __repr__(self) -> str:
        return f'DistributionNode("{self.label}")'


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

    def __repr__(self) -> str:
        return f'ParameterNode("{self.label}")'
