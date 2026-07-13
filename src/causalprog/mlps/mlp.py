"""Multi-Layer Perceptron (MLP) function builder."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import jax
from flax import nnx

from causalprog.mlps._specifiers import (
    ActivationName,
    NormName,
    resolve_activation,
    resolve_norm,
)
from causalprog.mlps._validation import resolve_hidden_dims, validate_mlp_base_config


class _MLPBlock(nnx.Module):
    """One MLP block."""

    def __init__(
        self,
        din: int,
        dout: int,
        *,
        activation: ActivationName,
        norm: NormName,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ) -> None:
        self.linear = nnx.Linear(din, dout, rngs=rngs)
        self.norm = resolve_norm(norm, dout, rngs=rngs)
        self.activation = resolve_activation(activation)
        self.dropout = nnx.Dropout(dropout_rate, deterministic=True)

    def __call__(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        return self.dropout(x, rngs=rngs)


class _StatefulMLP(nnx.Module):
    """Stateful implementation of an MLP."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        *,
        activation: ActivationName,
        norm: NormName,
        dropout_rate: float,
        rngs: nnx.Rngs,
    ) -> None:
        hidden_dims = list(hidden_dims)
        dims = [input_dim, *hidden_dims]

        self.blocks = nnx.List(
            _MLPBlock(
                din,
                dout,
                activation=activation,
                norm=norm,
                dropout_rate=dropout_rate,
                rngs=rngs,
            )
            for din, dout in pairwise(dims)
        )

        last_dim = hidden_dims[-1] if hidden_dims else input_dim
        self.output_layer = nnx.Linear(last_dim, output_dim, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        *,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        for block in self.blocks:
            x = block(x, rngs=rngs)

        return self.output_layer(x)


@dataclass(frozen=True)
class FunctionalMLP:
    """Callable functional version of an MLP."""

    graphdef: nnx.GraphDef

    def __call__(
        self,
        input_values: jax.Array,
        model_parameters: nnx.State,
        *,
        training: bool = False,
        rngs: nnx.Rngs | None = None,
    ) -> jax.Array:
        """
        Evaluate the MLP with explicit model parameters.

        Parameters
        ----------
        input_values
            Input array to pass through the MLP. The final dimension should match the
            input dimension used to build the MLP. A single input should have shape
            `(input_dim,)` and a batch of inputs should have shape
            `(..., input_dim)`.
        model_parameters
            Explicit MLP parameters, as returned by `mlp`.
        training
            If `False`, evaluate deterministically. If `True` and the MLP contains
            dropout, evaluate with dropout enabled.
        rngs
            Random number streams used by stochastic layers during training. Required
            when `training=True` and the MLP contains non-zero dropout. For dropout,
            pass an RNG stream such as `nnx.Rngs(dropout=key)`.

        Returns
        -------
        jax.Array
            The MLP output with shape `(output_dim,)` for a single input, or
            `(..., output_dim)` for batched inputs.

        """
        model = nnx.merge(self.graphdef, model_parameters)

        model = nnx.view(
            model,
            deterministic=not training,
            raise_if_not_found=False,
        )

        return model(input_values, rngs=rngs)


def mlp(
    input_dim: int,
    output_dim: int,
    *,
    hidden_layers: int | None = None,
    hidden_units: int | None = None,
    hidden_dims: Sequence[int] | None = None,
    activation: ActivationName = "gelu",
    norm: NormName = None,
    dropout_rate: float = 0.0,
    rngs: nnx.Rngs | None = None,
    seed: int = 0,
) -> tuple[FunctionalMLP, nnx.State]:
    """
    Build an explicit-parameter multilayer perceptron.

    The returned `FunctionalMLP` stores the model structure, while the trainable
    parameters are returned separately as an `nnx.State`.

    Hidden layers must be configured either with just `hidden_dims` or
    both `hidden_layers` and `hidden_units`.

    Parameters
    ----------
    input_dim
        Size of the final dimension of the input array.
    output_dim
        Size of the final dimension of the output array.
    hidden_layers
        Number of hidden layers to create when using `hidden_units`. Must be used
        together with `hidden_units`. Must not be provided if `hidden_dims` is
        provided. May be zero.
    hidden_units
        Number of units in each hidden layer when using `hidden_layers`. Must be
        used together with `hidden_layers`. Must not be provided if
        `hidden_dims` is provided.
    hidden_dims
        Explicit hidden-layer sizes. For example, `[16, 8]` creates two hidden
        layers with 16 and 8 units. Must not be provided with `hidden_layers` or
        `hidden_units`.
    activation
        Activation function used after each hidden linear layer. Options are
        `"relu"`, `"gelu"`, `"silu"`, `"tanh"`, and `"identity"`.
    norm
        Optional normalisation layer to apply after each hidden linear layer and
        before the activation. Options are `None`, `"layernorm"`, and `"rmsnorm"`.
    dropout_rate
        Dropout probability for hidden layers. Must be in the interval `[0, 1)`.
    rngs
        Random number streams used to initialise the MLP parameters. If not
        provided, `seed` is used to create parameter initialisation RNGs.
    seed
        Seed used for parameter initialisation when `rngs` is not provided.

    Returns
    -------
    FunctionalMLP
        Callable functional MLP object.
    nnx.State
        Initial trainable parameter state for the MLP.

    """
    resolved_hidden_dims = resolve_hidden_dims(
        hidden_dims=hidden_dims,
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
    )

    validate_mlp_base_config(
        input_dim=input_dim,
        output_dim=output_dim,
        dropout_rate=dropout_rate,
    )

    if rngs is None:
        rngs = nnx.Rngs(params=seed)

    model = _StatefulMLP(
        input_dim,
        output_dim,
        resolved_hidden_dims,
        activation=activation,
        norm=norm,
        dropout_rate=dropout_rate,
        rngs=rngs,
    )

    graphdef, initial_parameters = nnx.split(model, nnx.Param)

    return (
        FunctionalMLP(
            graphdef=graphdef,
        ),
        initial_parameters,
    )
