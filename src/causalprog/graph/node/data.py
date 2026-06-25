"""Graph nodes representing known of unknown data."""

import jax
import jax.numpy as jnp
import numpy.typing as npt
from typing_extensions import override

from .base import Node


class DataNode(Node):
    """
    A node containing non-stochastic data.

    `DataNode`s should not be used to encode constant values used by
    `DistributionNode`s. Such constant values should be given to the necessary
    `DistributionNode`s directly as `constant_parameters`.
    """

    def __init__(self, *, shape: tuple[int, ...] = (), label: str) -> None:
        """
        Initialise.

        Args:
            label: A unique label to identify the node

        """
        super().__init__(label=label, shape=shape, is_parameter=True)

    @override
    def sample(
        self,
        parameter_values: dict[str, float],
        sampled_dependencies: dict[str, npt.ArrayLike],
        samples: int,
        *,
        rng_key: jax.Array,
    ) -> npt.ArrayLike:
        if self.label not in parameter_values:
            msg = f"Missing input for node: {self.label}."
            raise ValueError(msg)
        return jnp.full(samples, parameter_values[self.label])

    @override
    def evaluate(
        self,
        **given_values: float | npt.NDArray[float],
    ) -> float | npt.NDArray[float]:
        if self.label not in given_values:
            msg = f"Missing input for node: {self.label}."
            raise ValueError(msg)
        value = given_values[self.label]
        if self.shape != (value.shape if hasattr(value, "shape") else ()):
            msg = f"Invalid value for node: {self.label}"
            raise ValueError(msg)
        return value

    @override
    def copy(self) -> Node:
        return DataNode(label=self.label)

    @override
    def __repr__(self) -> str:
        return f'DataNode(label="{self.label}")'

    @override
    @property
    def constant_parameters(self) -> dict[str, float]:
        return {}

    @override
    @property
    def parameters(self) -> dict[str, str]:
        return {}
