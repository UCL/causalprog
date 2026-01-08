"""Graph nodes representing distributions."""

from __future__ import annotations

import typing

import numpyro
from typing_extensions import override

from .base import Node

if typing.TYPE_CHECKING:
    import jax
    import numpy.typing as npt


class DistributionNode(Node):
    """A node containing a distribution."""

    def __init__(
        self,
        distribution: type,
        *,
        label: str,
        shape: tuple[int, ...] = (),
        parameters: dict[str, str] | None = None,
        constant_parameters: dict[str, float] | None = None,
    ) -> None:
        """
        Initialise.

        Args:
            distribution: The distribution
            label: A unique label to identify the node
            shape: The shape of the value for each sample
            parameters: A dictionary of parameters
            constant_parameters: A dictionary of constant parameters

        """
        self._dist = distribution
        self._constant_parameters = constant_parameters if constant_parameters else {}
        self._parameters = parameters if parameters else {}
        super().__init__(label=label, shape=shape, is_distribution=True)

    @override
    def sample(
        self,
        parameter_values: dict[str, float],
        sampled_dependencies: dict[str, npt.NDArray[float]],
        samples: int,
        *,
        rng_key: jax.Array,
    ) -> npt.NDArray[float]:
        d = self._dist(
            # Pass in node values derived from construction so far
            **{
                native_name: sampled_dependencies[node_name]
                for native_name, node_name in self.parameters.items()
            },
            # Pass in any constant parameters this node sets
            **self.constant_parameters,
        )

        return numpyro.sample(
            self.label,
            d,
            rng_key=rng_key,
            sample_shape=(samples, *self.shape)
            if d.batch_shape == () and samples > 1
            else self.shape,
        )

    @override
    def copy(self) -> Node:
        return DistributionNode(
            self._dist,
            label=self.label,
            shape=self.shape,
            parameters=dict(self._parameters),
            constant_parameters=dict(self._constant_parameters.items()),
        )

    @override
    def __repr__(self) -> str:
        r = f'DistributionNode({self._dist.__name__}, label="{self.label}"'
        if len(self._parameters) > 0:
            r += f", parameters={self._parameters}"
        if len(self.shape) > 0:
            r += f", shape={self.shape}"
        if len(self._constant_parameters) > 0:
            r += f", constant_parameters={self._constant_parameters}"
        return r

    @override
    @property
    def constant_parameters(self) -> dict[str, float]:
        return self._constant_parameters

    @override
    @property
    def parameters(self) -> dict[str, str]:
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
