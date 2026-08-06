"""Graph nodes representing random variables."""

from collections.abc import Callable

import jax
import numpy as np
from typing_extensions import override

from .base import Node


class RandomVariableNode(Node):
    """A node containing a random variable (RV)."""

    def _compute_not_set(self, *args, **kwargs) -> None:
        """
        Throw a suitable error at runtime if `compute` is not set.

        This is assigned to `RandomVariableNode._compute` if no method for
        computing / predicting the RV from its parent nodes is provided.
        It will throw an error if a user attempts to evaluate the node using
        this method, but only at runtime.

        Doing things this way means we don't have to run an `if` check every
        time the user attempts to call `.compute`, which we would otherwise be
        doing even after we have confirmed the attribute is set. Conversely,
        it also means that we can define RVs for testing purposes without needing
        to be overly verbose and specify a `compute` function that we are not going
        to use.
        """
        msg = f"Node {self.label} does not have a .compute method set"
        raise RuntimeError(msg)

    def __init__(
        self,
        *,
        shape: tuple[int, ...] = (),
        label: str,
        compute: MLPAlias | None = None,
        parents: list[str] | None = None,
    ) -> None:
        """
        Initialise.

        Args:
            shape: The shape of the output of the RV
            label: A unique label to identify the node
            compute: A function to compute node's value from given values of parents
            parents: Labels of parent nodes

        """
        super().__init__(label=label, shape=shape)
        if parents is None:
            self._parents = []
        else:
            self._parents = parents

        self._compute = compute or self._compute_not_set

    @override
    def sample(
        self,
        parameter_values: dict[str, float],
        sampled_dependencies: dict[str, jax.Array],
        samples: int,
        *,
        rng_key: jax.Array,
    ) -> jax.Array:
        raise NotImplementedError

    @override
    def evaluate(
        self,
        given_values: dict[str, jax.Array],
        parameters: ModelParam,
    ) -> jax.Array:
        if self.label in given_values:
            value = given_values[self.label]
            self.assert_is_valid_value(value)
            return value

        return self.compute(given_values, parameters)

    @override
    @property
    def parents(self) -> list[str]:
        return self._parents

    def compute(self, *args, **kwargs) -> jax.Array:
        """Directly compute the node value given values for all parents."""
        try:
            return self._compute(*args, **kwargs)
        except KeyError as e:
            if e.args[0] in self._parents:
                msg = f"Missing value for parent: {e.args[0]}"
                raise ValueError(msg) from None
            msg = f"Missing parameter: {e.args[0]}"
            raise ValueError(msg) from None

    @override
    def replace_parent(self, old_parent_label: str, new_parent_label: str) -> None:
        super().replace_parent(old_parent_label, new_parent_label)
        self._parents.remove(old_parent_label)
        self._parents.append(new_parent_label)


class ContinuousRandomVariableNode(RandomVariableNode):
    """A node containing a continuous random variable (RV)."""

    @override
    def __repr__(self) -> str:
        return f'ContinuousRandomVariableNode(label="{self.label}")'

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
        values: list[float] | list[jax.Array],
        shape: tuple[int, ...] = (),
        label: str,
        compute: MLPAlias | None = None,
        parents: list[str] | None = None,
    ) -> None:
        """
        Initialise.

        Args:
            values: A list of values that this node could take
            shape: The shape of the output of the RV
            label: A unique label to identify the node
            compute: A function to compute node's value from given values of parents
            parents: Labels of parent nodes

        """
        super().__init__(label=label, shape=shape, compute=compute, parents=parents)
        self._values = values

    @property
    def possible_values(self) -> list[float] | list[jax.Array]:
        """The values that this RV can take."""
        return self._values

    @override
    def __repr__(self) -> str:
        return f'DiscreteRandomVariableNode(label="{self.label}")'

    @override
    def is_valid_value(self, value: jax.Array) -> bool:
        return any(np.allclose(v, value) for v in self._values)

    @override
    def copy(self) -> Node:
        return DiscreteRandomVariableNode(
            values=self._values,
            shape=self.shape,
            label=self.label,
            compute=self._compute,
        )
