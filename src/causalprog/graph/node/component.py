"""Graph nodes representing distributions."""

from __future__ import annotations

import typing

from typing_extensions import override

from .base import Node

if typing.TYPE_CHECKING:
    import jax
    import numpy.typing as npt


class ComponentNode(Node):
    """A node representing a component of another node."""

    def __init__(
        self,
        node_label: str,
        component: tuple[int, ...],
        *,
        shape: tuple[int, ...] = (),
        label: str,
    ) -> None:
        """
        Initialise.

        Args:
            node: The node to take a component of
            component: The index/indices of the component
            label: A unique label to identify the node

        """
        self._component = component
        self._node_label = node_label
        super().__init__(shape=shape, label=label, is_distribution=True)

    @override
    def sample(
        self,
        parameter_values: dict[str, float],
        sampled_dependencies: dict[str, npt.NDArray[float]],
        samples: int,
        *,
        rng_key: jax.Array,
    ) -> npt.NDArray[float]:
        return sampled_dependencies[self._node_label][:, *self._component]

    @override
    def copy(self) -> Node:
        return ComponentNode(
            self._node_label,
            self._component,
            label=self.label,
            shape=self.shape,
            parameters=dict(self._parameters),
            constant_parameters=dict(self._constant_parameters.items()),
        )

    @override
    def __repr__(self) -> str:
        r = f'ComponentNode("{self._node_label}", component={self._component}'
        if len(self.shape) > 0:
            r += f", shape={self.shape}"
        if len(self._parameters) > 0:
            r += f", parameters={self._parameters}"
        if len(self._constant_parameters) > 0:
            r += f", constant_parameters={self._constant_parameters}"
        return r

    @override
    @property
    def constant_parameters(self) -> dict[str, float]:
        return {}

    @override
    @property
    def parameters(self) -> dict[str, str]:
        return {}
