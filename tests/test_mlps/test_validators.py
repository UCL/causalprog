"""Tests for mlp validators."""

from collections.abc import Sequence

import jax.numpy as jnp
import pytest

from causalprog._types import PyTree
from causalprog.mlps._validation import (
    n_elements_in_leaves,
    resolve_hidden_dims,
    validate_mlp_base_config,
)


@pytest.mark.parametrize(
    ("input_fmt", "expected_n_elements"),
    [
        pytest.param(3, 3, id="Scalar specification"),
        pytest.param(jnp.array([3]), 3, id="Column input, specified as array"),
        pytest.param(jnp.array([3, 4]), 12, id="Matrix input, specified as array"),
        pytest.param(
            {
                "a": 3,
                "b": jnp.array(
                    [
                        5,
                    ]
                ),
                "c": jnp.array([2, 7]),
            },
            3 + 5 + 2 * 7,
            id="PyTree input",
        ),
    ],
)
def test_n_elements_in_tree(input_fmt: int | PyTree, expected_n_elements: int) -> None:
    assert n_elements_in_leaves(input_fmt) == expected_n_elements


@pytest.mark.parametrize(
    ("hidden_dims", "hidden_layers", "hidden_units", "expected_error"),
    [
        (
            [8, 4],
            2,
            None,
            ValueError(
                "Pass either hidden_dims or hidden_layers/hidden_units, not both."
            ),
        ),
        (
            [8, 4],
            None,
            8,
            ValueError(
                "Pass either hidden_dims or hidden_layers/hidden_units, not both."
            ),
        ),
        (
            [8, 4],
            2,
            8,
            ValueError(
                "Pass either hidden_dims or hidden_layers/hidden_units, not both."
            ),
        ),
        (
            [8, 0],
            None,
            None,
            ValueError("All hidden_dims must be positive."),
        ),
        (
            None,
            None,
            None,
            ValueError(
                "Either hidden_dims or hidden_layers must be provided. "
                "hidden_units is required when hidden_layers is positive."
            ),
        ),
        (
            None,
            None,
            8,
            ValueError(
                "Either hidden_dims or hidden_layers must be provided. "
                "hidden_units is required when hidden_layers is positive."
            ),
        ),
        (
            None,
            -1,
            8,
            ValueError("hidden_layers must be non-negative."),
        ),
        (
            None,
            1,
            None,
            ValueError("hidden_units must be provided when hidden_layers is positive."),
        ),
        (
            None,
            1,
            0,
            ValueError("hidden_units must be positive."),
        ),
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
    expected_error: Exception,
    raises_context,
) -> None:
    with raises_context(expected_error):
        resolve_hidden_dims(
            hidden_dims=hidden_dims,
            hidden_layers=hidden_layers,
            hidden_units=hidden_units,
        )


@pytest.mark.parametrize(
    ("input_dim", "output_dim", "dropout_rate", "expected_error"),
    [
        (
            0,
            2,
            0.0,
            ValueError("input_dim must be positive."),
        ),
        (
            3,
            0,
            0.0,
            ValueError("output_dim must be positive."),
        ),
        (
            3,
            2,
            -0.1,
            ValueError("dropout_rate must be in [0, 1)."),
        ),
        (
            3,
            2,
            1.0,
            ValueError("dropout_rate must be in [0, 1)."),
        ),
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
    expected_error: Exception,
    raises_context,
) -> None:
    with raises_context(expected_error):
        validate_mlp_base_config(
            input_dim=input_dim,
            output_dim=output_dim,
            dropout_rate=dropout_rate,
        )
