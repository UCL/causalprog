from collections.abc import Sequence

import jax
import jax.numpy as jnp

from causalprog._types import PyTree


def n_elements_in_leaves(input_data_format: int | PyTree) -> int:
    """
    Extract the total number of elements in the leaves of a `PyTree`.

    This function is primarily used to determine the size of the input
    dimension to a `FunctionalMLP`, from the specification of the input
    data format. In the event this specification is just a scalar, this
    is interpreted as a standard column vector of that many elements. In
    the event that the input is a `PyTree`, the leaves of the `PyTree` are
    interpreted as the shapes of the arrays at the corresponding leaves.

    Args:
        input_data_format: `PyTree` whose leaves contain shape information.

    Returns:
        Total number of elements in the `PyTree`, based on shape information.

    """
    if isinstance(input_data_format, int):
        return input_data_format
    elements_per_leaf = jax.tree.map(jnp.prod, input_data_format)
    return jax.tree.reduce(jnp.sum, elements_per_leaf, 0.0)


def resolve_hidden_dims(
    *,
    hidden_dims: Sequence[int] | None,
    hidden_layers: int | None,
    hidden_units: int | None,
) -> list[int]:
    if hidden_dims is not None:
        if hidden_layers is not None or hidden_units is not None:
            msg = "Pass either hidden_dims or hidden_layers/hidden_units, not both."
            raise ValueError(msg)

        if any(dim <= 0 for dim in hidden_dims):
            msg_1 = "All hidden_dims must be positive."
            raise ValueError(msg_1)

        return list(hidden_dims)

    if hidden_layers is None:
        msg = (
            "Either hidden_dims or hidden_layers must be provided. "
            "hidden_units is required when hidden_layers is positive."
        )
        raise ValueError(msg)

    if hidden_layers < 0:
        msg = "hidden_layers must be non-negative."
        raise ValueError(msg)

    if hidden_layers == 0:
        return []

    if hidden_units is None:
        msg = "hidden_units must be provided when hidden_layers is positive."
        raise ValueError(msg)

    if hidden_units <= 0:
        msg = "hidden_units must be positive."
        raise ValueError(msg)

    return [hidden_units] * hidden_layers


def validate_mlp_base_config(
    *,
    input_dim: int,
    output_dim: int,
    dropout_rate: float,
) -> None:
    if input_dim <= 0:
        msg = "input_dim must be positive."
        raise ValueError(msg)

    if output_dim <= 0:
        msg_0 = "output_dim must be positive."
        raise ValueError(msg_0)

    if dropout_rate < 0.0 or dropout_rate >= 1.0:
        msg_2 = "dropout_rate must be in [0, 1)."
        raise ValueError(msg_2)
