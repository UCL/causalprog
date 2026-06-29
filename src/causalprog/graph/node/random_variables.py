"""Graph nodes representing random variables."""

import typing
from abc import abstractmethod

import jax
import numpy as np
import numpy.typing as npt
from typing_extensions import override

from .base import Node


class RandomVariableNode(Node):
    """A node containing a random variable (RV)."""

    def __init__(
        self,
        *,
        shape: tuple[int, ...] = (),
        label: str,
        compute: typing.Callable | None = None,
        parents: list[str] | None = None,
    ) -> None:
        """
        Initialise.

        Args:
            shape: The shape of the output of the RV
            label: A unique label to identify the node
            compute: A function to compute node's value from given values of parents

        """
        super().__init__(label=label, shape=shape)
        if parents is None:
            self._parents = []
        else:
            self._parents = parents
        self._compute = compute

    @override
    def sample(
        self,
        parameter_values: dict[str, float],
        sampled_dependencies: dict[str, npt.ArrayLike],
        samples: int,
        *,
        rng_key: jax.Array,
    ) -> npt.ArrayLike:
        raise NotImplementedError

    @override
    def evaluate(
        self,
        given_values: dict[str, float | npt.NDArray[float]],
    ) -> float | npt.NDArray[float]:
        if self.label in given_values:
            value = given_values[self.label]
            self.assert_is_valid_value(value)
            return value

        if self._compute is None:
            msg = f"Missing input for node: {self.label}."
            raise ValueError(msg)
        return self._compute(given_values)

    @override
    @property
    def parents(self) -> list[str]:
        return self._parents

    @abstractmethod
    def is_valid_value(self, value: float | npt.NDArray[float]) -> bool:
        """Check if a value is valid for this node."""

    def assert_is_valid_value(self, value: float | npt.NDArray[float]) -> None:
        """Check if a value is valid for this node."""
        if not self.is_valid_value(value):
            msg = (
                f"Invalid value for {self.__class__.__name__}: "
                f"{self.label} cannot be {value}"
            )
            raise ValueError(msg)
        if self.shape != (value.shape if hasattr(value, "shape") else ()):
            msg = f"Invalid value for node: {self.label}"
            raise ValueError(msg)


class ContinuousRandomVariableNode(RandomVariableNode):
    """A node containing a continuous random variable (RV)."""

    @override
    def __repr__(self) -> str:
        return f'ContinuousRandomVariableNode(label="{self.label}")'

    @override
    def is_valid_value(self, value: float | npt.NDArray[float]) -> bool:
        return True

    @override
    def copy(self) -> Node:
        return ContinuousRandomVariableNode(
            shape=self.shape, label=self.label, compute=self._compute
        )


class DiscreteRandomVariableNode(RandomVariableNode):
    """A node containing a discrete random variable (RV)."""

    def __init__(
        self,
        *,
        values: list[float] | list[npt.NDArray[float]],
        shape: tuple[int, ...] = (),
        label: str,
        compute: typing.Callable | None = None,
    ) -> None:
        """
        Initialise.

        Args:
            shape: The shape of the output of the RV
            label: A unique label to identify the node
            compute: A function to compute node's value from given values of parents

        """
        super().__init__(label=label, shape=shape, compute=compute)
        self._values = values

    @property
    def possible_values(self) -> list[float] | list[npt.NDArray[float]]:
        """The values that this RV can take."""
        return self._values

    @override
    def __repr__(self) -> str:
        return f'DiscreteRandomVariableNode(label="{self.label}")'

    @override
    def is_valid_value(self, value: float | npt.NDArray[float]) -> bool:
        return any(np.allclose(v, value) for v in self._values)

    @override
    def copy(self) -> Node:
        return DiscreteRandomVariableNode(
            values=self._values,
            shape=self.shape,
            label=self.label,
            compute=self._compute,
        )
