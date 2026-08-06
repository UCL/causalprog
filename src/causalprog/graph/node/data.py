"""Graph nodes representing known of unknown data."""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from typing_extensions import override

from .base import Node


class DataNode(Node):
    """
    A node containing non-stochastic data.

    `DataNode`s should not be used to encode constant values used by
    `DistributionNode`s. Such constant values should either set when
    node is initialised or be given to the necessary
    `DistributionNode`s directly as `constant_parameters`.
    """

    def __init__(
        self,
        *,
        shape: tuple[int, ...] | None = None,
        label: str,
        value: ArrayLike | None = None,
    ) -> None:
        """
        Initialise.

        Args:
            label: A unique label to identify the node
            shape: The shape of the node's value for each sample
            value: The value of this constant

        """
        if value is None:
            self._value = value
            if shape is None:
                shape = ()
        else:
            self._value = jnp.array(value)
            if shape is None:
                shape = self._value.shape
            elif shape != value.shape:
                msg = "Node value has incorect shape."
                raise ValueError(msg)

        super().__init__(label=label, shape=shape)

    @override
    def sample(
        self,
        parameter_values: dict[str, float],
        sampled_dependencies: dict[str, jax.Array],
        samples: int,
        *,
        rng_key: jax.Array,
    ) -> jax.Array:
        if self._value is None:
            if self.label not in parameter_values:
                msg = f"Missing input for node: {self.label}."
                raise ValueError(msg)
            return jnp.full(samples, parameter_values[self.label])
        else:
            return jnp.full(samples, self._value)

    @override
    def evaluate(
        self,
        given_values: dict[str, jax.Array],
    ) -> jax.Array:
        if self._value is None:
            if self.label not in given_values:
                msg = f"Missing input for node: {self.label}."
                raise ValueError(msg)
            value = given_values[self.label]
        else:
            value = self._value
        if self.shape != (value.shape if hasattr(value, "shape") else ()):
            msg = f"Invalid value for node: {self.label}"
            raise ValueError(msg)
        return value

    @override
    def copy(self) -> Node:
        return DataNode(label=self.label, shape=self.shape, value=self._value)

    @override
    def __repr__(self) -> str:
        if self._value is None:
            return f'DataNode(label="{self.label}", shape={self.shape})'
        else:
            return f'DataNode(label="{self.label}", shape={self.shape}, value={self_value})'

    @override
    @property
    def parents(self) -> list[str]:
        return []
