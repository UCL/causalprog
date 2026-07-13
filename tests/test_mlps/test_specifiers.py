from typing import cast

from flax import nnx

from causalprog.mlps._specifiers import (
    ActivationName,
    NormName,
    resolve_activation,
    resolve_norm,
)


def test_resolve_activation_rejects_unknown_activation(raises_context) -> None:
    unknown_activation = cast("ActivationName", "not_an_activation")

    with raises_context(ValueError("Unknown activation: not_an_activation")):
        resolve_activation(unknown_activation)


def test_resolve_norm_rejects_unknown_norm(raises_context) -> None:
    unknown_norm = cast("NormName", "batchnorm")

    with raises_context(ValueError("Unknown norm: batchnorm")):
        resolve_norm(
            unknown_norm,
            num_features=3,
            rngs=nnx.Rngs(params=0),
        )
