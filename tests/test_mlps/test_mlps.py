"""Tests for explicit-parameter MLP builders."""

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from itertools import pairwise
from typing import Any

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from causalprog.mlps import FunctionalMLP, mlp
from causalprog.mlps._specifiers import NormName


@pytest.fixture
def x_3() -> jax.Array:
    return jnp.array([1.0, -2.0, 0.5])


def build_mlp(**overrides):
    kwargs = {
        "input_dim": 3,
        "output_dim": 2,
        "hidden_layers": 3,
        "hidden_units": 8,
    }
    kwargs.update(overrides)
    return mlp(**kwargs)


def fit_mlp_to_targets(
    f: FunctionalMLP,
    theta: nnx.State,
    x_train: jax.Array,
    y_train: jax.Array,
    *,
    learning_rate: float,
    steps: int,
    dropout_seed: int = 0,
) -> tuple[nnx.State, jax.Array, jax.Array]:
    optimiser = optax.adam(learning_rate=learning_rate)
    opt_state = optimiser.init(theta)

    def eval_loss_fn(theta: nnx.State) -> jax.Array:
        preds = f(x_train, theta)
        return jnp.mean((preds - y_train) ** 2)

    def train_loss_fn(theta: nnx.State, dropout_key: jax.Array) -> jax.Array:
        preds = f(
            x_train,
            theta,
            training=True,
            rngs=nnx.Rngs(dropout=dropout_key),
        )
        return jnp.mean((preds - y_train) ** 2)

    @jax.jit
    def train_step(
        theta: nnx.State,
        opt_state: optax.OptState,
        dropout_key: jax.Array,
    ) -> tuple[nnx.State, optax.OptState]:
        _, grads = jax.value_and_grad(train_loss_fn)(theta, dropout_key)
        updates, opt_state = optimiser.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state

    initial_loss = eval_loss_fn(theta)

    dropout_key = jax.random.key(dropout_seed)

    for _ in range(steps):
        dropout_key, step_key = jax.random.split(dropout_key)
        theta, opt_state = train_step(theta, opt_state, step_key)

    final_loss = eval_loss_fn(theta)

    return theta, initial_loss, final_loss


def test_mlp_returns_functional_mlp_and_parameters() -> None:
    f, theta = build_mlp()

    assert isinstance(f, FunctionalMLP)
    assert isinstance(theta, nnx.State)
    assert isinstance(f.graphdef, nnx.GraphDef)


def test_functional_mlp_matches_merged_stateful_mlp(x_3: jax.Array) -> None:
    f, theta = build_mlp()
    model = nnx.merge(f.graphdef, theta)

    y_functional = f(x_3, theta)
    y_stateful = model(x_3)

    assert y_functional.shape == y_stateful.shape
    assert bool(jnp.allclose(y_functional, y_stateful))


@pytest.mark.parametrize(
    ("hidden_dims", "hidden_layers", "hidden_units", "expected_hidden_dims"),
    [
        ([8, 4], None, None, [8, 4]),
        ((3, 2, 2, 8), None, None, [3, 2, 2, 8]),
        ([], None, None, []),
        (None, 0, None, []),
        (None, 0, 8, []),
        (None, 1, 8, [8]),
        (None, 2, 8, [8, 8]),
        (None, 3, 4, [4, 4, 4]),
    ],
    ids=[
        "hidden-dims-list",
        "hidden-dims-tuple",
        "empty-hidden-dims",
        "zero-hidden-layers-no-units",
        "zero-hidden-layers-with-units",
        "one-hidden-layer",
        "two-hidden-layers",
        "three-hidden-layers",
    ],
)
def test_mlp_uses_correct_hidden_configuration(
    hidden_dims: Sequence[int] | None,
    hidden_layers: int | None,
    hidden_units: int | None,
    expected_hidden_dims: list[int],
) -> None:
    f, theta = build_mlp(
        hidden_dims=hidden_dims,
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
    )
    model = nnx.merge(f.graphdef, theta)

    assert len(model.blocks) == len(expected_hidden_dims)

    hidden_block_dims = [3, *expected_hidden_dims]

    for block, (expected_in, expected_out) in zip(
        model.blocks,
        pairwise(hidden_block_dims),
        strict=True,
    ):
        assert isinstance(block.linear, nnx.Linear)
        assert block.linear.in_features == expected_in
        assert block.linear.out_features == expected_out

    output_input_dim = expected_hidden_dims[-1] if expected_hidden_dims else 3

    assert isinstance(model.output_layer, nnx.Linear)
    assert model.output_layer.in_features == output_input_dim
    assert model.output_layer.out_features == 2


@pytest.mark.parametrize(
    ("norm", "expected_norm"),
    [
        (None, nnx.identity),
        ("layernorm", nnx.LayerNorm),
        ("rmsnorm", nnx.RMSNorm),
    ],
    ids=[
        "no-norm",
        "layernorm",
        "rmsnorm",
    ],
)
def test_mlp_same_norm_accross_blocks(
    norm: NormName,
    expected_norm: Callable[..., object] | type[nnx.Module],
) -> None:
    f, theta = build_mlp(norm=norm)
    model = nnx.merge(f.graphdef, theta)

    for block in model.blocks:
        if expected_norm is nnx.identity:
            assert block.norm is nnx.identity
        else:
            assert isinstance(expected_norm, type)  # Narrow for mypy
            assert isinstance(block.norm, expected_norm)


@pytest.mark.parametrize(
    ("activation", "expected_activation"),
    [
        ("relu", nnx.relu),
        ("gelu", nnx.gelu),
        ("silu", nnx.silu),
        ("tanh", jnp.tanh),
        ("identity", None),
        (None, None),
    ],
    ids=[
        "relu",
        "gelu",
        "silu",
        "tanh",
        "identity",
        "None",
    ],
)
def test_mlp_same_activations_across_block(
    activation: str,
    expected_activation: Callable[[jax.Array], jax.Array] | None,
    x_3: jax.Array,
) -> None:
    f, theta = build_mlp(activation=activation)
    model = nnx.merge(f.graphdef, theta)

    for block in model.blocks:
        if expected_activation is None:
            assert jnp.allclose(block.activation(x_3), x_3)
        else:
            assert block.activation is expected_activation


def test_mlp_dropout_params() -> None:
    dropout_rate = 0.33

    f, theta = build_mlp(dropout_rate=dropout_rate)
    model = nnx.merge(f.graphdef, theta)

    for block in model.blocks:
        assert isinstance(block.dropout, nnx.Dropout)
        assert block.dropout.rate == dropout_rate
        assert block.dropout.deterministic is True


def test_mlp_is_deterministic_in_eval_mode_with_dropout(x_3: jax.Array) -> None:
    f, theta = build_mlp(dropout_rate=0.5)

    y1 = f(x_3, theta)
    y2 = f(x_3, theta)

    assert bool(jnp.allclose(y1, y2))


def test_mlp_training_uses_configured_dropout_rate() -> None:
    dropout_rate = 0.33
    width = 8

    f, theta = mlp(
        input_dim=width,
        output_dim=width,
        hidden_layers=1,
        hidden_units=width,
        activation="identity",
        norm=None,
        dropout_rate=dropout_rate,
        seed=0,
    )

    model = nnx.merge(f.graphdef, theta)

    model.blocks[0].linear.kernel[...] = jnp.eye(width)
    model.blocks[0].linear.bias[...] = jnp.zeros(width)
    model.output_layer.kernel[...] = jnp.eye(width)
    model.output_layer.bias[...] = jnp.zeros(width)

    _, theta = nnx.split(model, nnx.Param)

    x = jnp.ones((10_000, width))

    y = f(
        x,
        theta,
        training=True,
        rngs=nnx.Rngs(dropout=0),
    )

    dropped_fraction = jnp.mean(y == 0.0)
    retained_values = y[y != 0.0]

    assert float(dropped_fraction) == pytest.approx(dropout_rate, abs=0.02)
    assert bool(jnp.allclose(retained_values, 1.0 / (1.0 - dropout_rate)))


def test_mlp_is_jittable(x_3: jax.Array) -> None:
    f, theta = build_mlp()

    @jax.jit
    def apply(theta: nnx.State, x: jax.Array) -> jax.Array:
        return f(x, theta, training=True, rngs=nnx.Rngs(dropout=0))

    apply(theta, x_3)


@pytest.mark.parametrize(
    "use_same_seed",
    [
        True,
        False,
    ],
    ids=[
        "same-seed",
        "different-seeds",
    ],
)
def test_mlp_initialisation_depends_on_seed(
    x_3: jax.Array,
    seed: int,
    use_same_seed: bool,
) -> None:
    other_seed = seed if use_same_seed else seed + 1

    f_0, theta_0 = build_mlp(seed=seed)
    f_1, theta_1 = build_mlp(seed=other_seed)

    y_0 = f_0(x_3, theta_0)
    y_1 = f_1(x_3, theta_1)

    outputs_are_equal = bool(jnp.allclose(y_0, y_1))

    assert outputs_are_equal is use_same_seed


def test_shared_rngs_advance_between_mlp_initialisations(x_3: jax.Array) -> None:
    rngs = nnx.Rngs(params=0)

    count_before = int(rngs.params.count[...])

    f_0, theta_0 = build_mlp(rngs=rngs)

    count_after_first_mlp = int(rngs.params.count[...])

    f_1, theta_1 = build_mlp(rngs=rngs)

    count_after_second_mlp = int(rngs.params.count[...])

    y_0 = f_0(x_3, theta_0)
    y_1 = f_1(x_3, theta_1)

    assert not bool(jnp.allclose(y_0, y_1))
    assert count_after_first_mlp > count_before
    assert count_after_second_mlp > count_after_first_mlp


def test_mlp_forward_pass_calls_layers_in_expected_order(x_3: jax.Array) -> None:
    def record_call(calls, name, fn):
        def wrapped(*args, **kwargs):
            calls.append(name)
            return fn(*args, **kwargs)

        return wrapped

    f, theta = build_mlp(
        hidden_layers=2,
        activation="gelu",
        norm="layernorm",
        dropout_rate=0.5,
        seed=0,
    )

    model = nnx.merge(f.graphdef, theta)
    model = nnx.view(model, deterministic=False)

    calls: list[str] = []

    for block_index, block in enumerate(model.blocks):
        block.linear = record_call(
            calls,
            f"block_{block_index}.linear",
            block.linear,
        )
        block.norm = record_call(
            calls,
            f"block_{block_index}.norm",
            block.norm,
        )
        block.activation = record_call(
            calls,
            f"block_{block_index}.activation",
            block.activation,
        )
        block.dropout = record_call(
            calls,
            f"block_{block_index}.dropout",
            block.dropout,
        )

    model.output_layer = record_call(calls, "output_layer", model.output_layer)
    model(x_3, rngs=nnx.Rngs(dropout=1))

    assert calls == [
        "block_0.linear",
        "block_0.norm",
        "block_0.activation",
        "block_0.dropout",
        "block_1.linear",
        "block_1.norm",
        "block_1.activation",
        "block_1.dropout",
        "output_layer",
    ]


def test_mlp_training_with_dropout_requires_rngs(
    x_3: jax.Array,
    raises_context: Callable[[Exception], AbstractContextManager[Any]],
) -> None:
    f, theta = build_mlp(dropout_rate=0.5)

    with raises_context(ValueError("no `rngs` argument was provided to Dropout")):
        f(x_3, theta, training=True)


def test_mlp_rejects_unknown_activation(
    raises_context: Callable[[Exception], AbstractContextManager[Any]],
) -> None:
    with raises_context(ValueError("Unknown activation: not_an_activation")):
        build_mlp(activation="not_an_activation")


def test_mlp_rejects_unknown_norm(
    raises_context: Callable[[Exception], AbstractContextManager[Any]],
) -> None:
    with raises_context(ValueError("Unknown norm: batchnorm")):
        build_mlp(norm="batchnorm")


@pytest.mark.parametrize(
    (
        "y_func",
        "num_samples",
        "mlp_kwargs",
        "fitting_kwargs",
        "tolerance",
    ),
    [
        pytest.param(
            lambda x: 2.0 * x + 0.5,
            64,
            {
                "input_dim": 1,
                "output_dim": 1,
                "hidden_dims": [],
                "activation": "identity",
                "norm": None,
                "dropout_rate": 0.0,
            },
            {
                "learning_rate": 0.05,
                "steps": 300,
            },
            1e-5,
            id="linear-problem",
        ),
        pytest.param(
            lambda x: x**3 + 0.3 * x - 0.2,
            128,
            {
                "input_dim": 1,
                "output_dim": 1,
                "hidden_layers": 2,
                "hidden_units": 16,
                "activation": "relu",
                "norm": None,
                "dropout_rate": 0.0,
            },
            {
                "learning_rate": 0.01,
                "steps": 1_000,
            },
            1e-5,
            id="nonlinear-problem",
        ),
    ],
)
def test_mlp_learning(
    y_func: Callable[[jax.Array], jax.Array],
    num_samples: int,
    mlp_kwargs: dict[str, Any],
    fitting_kwargs: dict[str, Any],
    tolerance: float,
    seed: int,
) -> None:
    x_train = jnp.linspace(-1.0, 1.0, num_samples).reshape(-1, 1)
    y_train = y_func(x_train)

    f, theta = mlp(
        **mlp_kwargs,
        seed=seed,
    )

    _, initial_loss, final_loss = fit_mlp_to_targets(
        f,
        theta,
        x_train,
        y_train,
        **fitting_kwargs,
    )

    assert float(final_loss) < float(initial_loss)
    assert float(final_loss) < tolerance
