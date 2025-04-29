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
        label: str,
        *,
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
        super().__init__(label, is_outcome=is_outcome, is_parameter=False)

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
    """A node containing a parameter."""

    def __init__(self, label: str, *, is_outcome: bool = False) -> None:
        """Initialise."""
        super().__init__(label, is_outcome=is_outcome, is_parameter=True)
        self.value: int | None = None

    def sample(
        self,
        _sampled_dependencies: dict[str, npt.NDArray[float]],
        _samples: int,
        _rng_key: jax.Array,
    ) -> npt.NDArray[float]:
        """Sample a value from the node."""
        if self.label not in sampled_dependencies:
            msg = "Cannot sample an undetermined parameter node."
            raise ValueError(msg)
        return sampled_dependencies[self.label]

    def __repr__(self) -> str:
        return f'ParameterNode("{self.label}")'
