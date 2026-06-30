"""Graph nodes representing distributions."""

from __future__ import annotations

import typing

import jax.numpy as jnp
from typing_extensions import override

from .base import Node

if typing.TYPE_CHECKING:
    import jax
    import numpy.typing as npt


class ConstantNode(Node):
    """A node representing a constant."""

    def __init__(self, *, label: str, value: float | npt.NDArray[float]) -> None:
        """
        Initialise.

        Args:
            label: A unique label to identify the node
            value: The value of this constant

        """
        self._value = value
        super().__init__(
            shape=() if isinstance(value, float) else value.shape, label=label
        )

    @override
    def sample(
        self,
        parameter_values: dict[str, float],
        sampled_dependencies: dict[str, npt.NDArray[float]],
        samples: int,
        *,
        rng_key: jax.Array,
    ) -> npt.NDArray[float]:
        return jnp.full(samples, self._value)

    @override
    def evaluate(
        self,
        given_values: dict[str, float | npt.NDArray[float]],
    ) -> float | npt.NDArray[float]:
        return self._value

    @override
    def copy(self) -> Node:
        return ConstantNode(label=self.label, value=self._value)

    @override
    def __repr__(self) -> str:
        return f"ConstantNode({self._value})"

    @override
    @property
    def parents(self) -> list[str]:
        return []

    @override
    def is_valid_value(self, value: float | npt.NDArray[float]) -> bool:
        return True
