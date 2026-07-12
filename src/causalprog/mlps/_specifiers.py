from collections.abc import Callable
from typing import Literal

import jax
from flax import nnx

ActivationName = Literal["relu", "gelu", "silu", "tanh", "identity"] | None
NormName = Literal["layernorm", "rmsnorm"] | None


def resolve_activation(name: ActivationName) -> Callable[[jax.Array], jax.Array]:
    match name:
        case "relu":
            return nnx.relu
        case "gelu":
            return nnx.gelu
        case "silu":
            return nnx.silu
        case "tanh":
            return nnx.tanh
        case "identity" | None:
            return nnx.identity
        case _:
            msg = f"Unknown activation: {name}"
            raise ValueError(msg)


def resolve_norm(
    name: NormName,
    num_features: int,
    *,
    rngs: nnx.Rngs,
) -> nnx.Module | Callable[[jax.Array], jax.Array]:
    match name:
        case "layernorm":
            return nnx.LayerNorm(num_features, rngs=rngs)
        case "rmsnorm":
            return nnx.RMSNorm(num_features, rngs=rngs)
        case None:
            return nnx.identity
        case _:
            msg = f"Unknown norm: {name}"
            raise ValueError(msg)
