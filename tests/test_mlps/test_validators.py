from collections.abc import Sequence

import pytest

from causalprog.mlps._validation import (
    resolve_hidden_dims,
    validate_mlp_base_config,
)


@pytest.mark.parametrize(
    ("hidden_dims", "hidden_layers", "hidden_units", "message"),
    [
        ([8, 4], 2, None, "Pass either hidden_dims or hidden_layers/hidden_units"),
        ([8, 4], None, 8, "Pass either hidden_dims or hidden_layers/hidden_units"),
        ([8, 4], 2, 8, "Pass either hidden_dims or hidden_layers/hidden_units"),
        ([8, 0], None, None, "All hidden_dims must be positive"),
        (None, None, None, "Either hidden_dims or hidden_layers must be provided"),
        (None, None, 8, "Either hidden_dims or hidden_layers must be provided"),
        (None, -1, 8, "hidden_layers must be non-negative"),
        (None, 1, None, "hidden_units must be provided"),
        (None, 1, 0, "hidden_units must be positive"),
    ],
    ids=[
        "hidden-dims-with-hidden-layers",
        "hidden-dims-with-hidden-units",
        "hidden-dims-with-hidden-layers-and-units",
        "hidden-dims-contains-zero",
        "missing-hidden-configuration",
        "missing-hidden-layers-with-hidden-units",
        "negative-hidden-layers",
        "positive-hidden-layers-missing-units",
        "positive-hidden-layers-zero-units",
    ],
)
def test_resolve_hidden_dims_rejects_invalid_configuration(
    hidden_dims: Sequence[int] | None,
    hidden_layers: int | None,
    hidden_units: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_hidden_dims(
            hidden_dims=hidden_dims,
            hidden_layers=hidden_layers,
            hidden_units=hidden_units,
        )


@pytest.mark.parametrize(
    ("input_dim", "output_dim", "dropout_rate", "message"),
    [
        (0, 2, 0.0, "input_dim must be positive"),
        (3, 0, 0.0, "output_dim must be positive"),
        (3, 2, -0.1, "dropout_rate must be in"),
        (3, 2, 1.0, "dropout_rate must be in"),
    ],
    ids=[
        "zero-input-dim",
        "zero-output-dim",
        "negative-dropout-rate",
        "dropout-rate-one",
    ],
)
def test_validate_mlp_base_config_rejects_invalid_configuration(
    input_dim: int,
    output_dim: int,
    dropout_rate: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_mlp_base_config(
            input_dim=input_dim,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
        )
