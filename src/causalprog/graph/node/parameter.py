"""Graph nodes representing parameters."""

from __future__ import annotations

import typing

import numpy as np
from typing_extensions import override

from .base import Node

if typing.TYPE_CHECKING:
    import jax
    import numpy.typing as npt


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
        return f'ParameterNode(label="{self.label}")'

    @override
    @property
    def constant_parameters(self) -> dict[str, float]:
        return {}

    @override
    @property
    def parameters(self) -> dict[str, str]:
        return {}
