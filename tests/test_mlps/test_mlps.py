"""Tests for explicit-parameter MLP builders."""

from collections.abc import Callable, Sequence
from itertools import pairwise

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from causalprog.mlps import FunctionalMLP, mlp


def test_mlp_returns_functional_mlp_and_parameters() -> None:
    f, theta = mlp(
        input_dim=3,
        output_dim=2,
        hidden_layers=2,
        hidden_units=8,
        seed=0,
    )

    assert isinstance(f, FunctionalMLP)
    assert isinstance(theta, nnx.State)
    assert isinstance(f.graphdef, nnx.GraphDef)


def test_functional_mlp_matches_merged_stateful_mlp() -> None:
    f, theta = mlp(
        input_dim=3,
        output_dim=2,
        hidden_layers=2,
        hidden_units=8,
        activation="gelu",
        norm="layernorm",
        dropout_rate=0.0,
        seed=0,
    )

    x = jnp.array([1.0, 2.0, 3.0])

    model = nnx.merge(f.graphdef, theta)
    model = nnx.view(model)

    y_functional = f(x, theta)
    y_stateful = model(x)

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
    f, theta = mlp(
        input_dim=3,
        output_dim=2,
        hidden_dims=hidden_dims,
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        seed=0,
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
    ("norm", "expected_norm_type"),
    [
        (None, type(None)),
        ("layernorm", nnx.LayerNorm),
        ("rmsnorm", nnx.RMSNorm),
    ],
    ids=[
        "no-norm",
        "layernorm",
        "rmsnorm",
    ],
)
def test_mlp_norms(norm: str | None, expected_norm_type: type) -> None:
    hidden_layers = 3

    f, theta = mlp(
        input_dim=3,
        output_dim=2,
        hidden_layers=hidden_layers,
        hidden_units=8,
        norm=norm,
        seed=0,
    )

    model = nnx.merge(f.graphdef, theta)

    for block in model.blocks:
        assert isinstance(block.norm, expected_norm_type)


@pytest.mark.parametrize(
    ("activation", "expected_activation"),
    [
        ("relu", nnx.relu),
        ("gelu", nnx.gelu),
        ("silu", nnx.silu),
        ("tanh", jnp.tanh),
        ("identity", None),
    ],
    ids=[
        "relu",
        "gelu",
        "silu",
        "tanh",
        "identity",
    ],
)
def test_mlp_activations(
    activation: str,
    expected_activation: Callable[[jax.Array], jax.Array] | None,
) -> None:
    hidden_layers = 3

    f, theta = mlp(
        input_dim=3,
        output_dim=2,
        hidden_layers=hidden_layers,
        hidden_units=8,
        activation=activation,
        seed=0,
    )

    model = nnx.merge(f.graphdef, theta)

    for block in model.blocks:
        if expected_activation is None:
            x = jnp.array([-1.0, 0.0, 2.0])
            assert jnp.allclose(block.activation(x), x)
        else:
            assert block.activation is expected_activation


# TODO: Not tested non-determinism in training mode or apply_train.


def test_mlp_dropout() -> None:
    hidden_layers = 3
    dropout_rate = 0.5

    f, theta = mlp(
        input_dim=3,
        output_dim=2,
        hidden_layers=hidden_layers,
        hidden_units=8,
        dropout_rate=dropout_rate,
        seed=0,
    )

    model = nnx.merge(f.graphdef, theta)

    for block in model.blocks:
        assert isinstance(block.dropout, nnx.Dropout)
        assert block.dropout.rate == dropout_rate
        assert block.dropout.deterministic is True


def test_mlp_is_deterministic_in_eval_mode_with_dropout() -> None:
    f, theta = mlp(
        input_dim=3,
        output_dim=2,
        hidden_layers=1,
        hidden_units=8,
        dropout_rate=0.5,
        seed=0,
    )

    x = jnp.ones((3,))

    y1 = f(x, theta)
    y2 = f(x, theta)

    assert bool(jnp.allclose(y1, y2))


def test_mlp_is_jittable() -> None:
    f, theta = mlp(
        input_dim=3,
        output_dim=2,
        hidden_layers=1,
        hidden_units=8,
        seed=0,
    )

    @jax.jit
    def apply(theta: nnx.State, x: jax.Array) -> jax.Array:
        return f(x, theta)

    x = jnp.ones((3,))

    apply(theta, x)


def test_mlp_same_seed_gives_same_initialisation() -> None:
    f_0, theta_0 = mlp(
        input_dim=3,
        output_dim=2,
        hidden_layers=2,
        hidden_units=8,
        seed=0,
    )

    f_1, theta_1 = mlp(
        input_dim=3,
        output_dim=2,
        hidden_layers=2,
        hidden_units=8,
        seed=0,
    )

    x = jnp.array([1.0, -2.0, 0.5])

    y_0 = f_0(x, theta_0)
    y_1 = f_1(x, theta_1)

    assert bool(jnp.allclose(y_0, y_1))


def test_mlp_different_seeds_give_different_initialisations() -> None:
    f_0, theta_0 = mlp(
        input_dim=3,
        output_dim=2,
        hidden_layers=2,
        hidden_units=8,
        seed=0,
    )

    f_1, theta_1 = mlp(
        input_dim=3,
        output_dim=2,
        hidden_layers=2,
        hidden_units=8,
        seed=1,
    )

    x = jnp.array([1.0, -2.0, 0.5])

    y_0 = f_0(x, theta_0)
    y_1 = f_1(x, theta_1)

    assert not bool(jnp.allclose(y_0, y_1))


def test_copy_rngs_gives_same_initialisation() -> None:
    rngs_0 = nnx.Rngs(params=123)
    rngs_1 = nnx.Rngs(params=123)

    f_0, theta_0 = mlp(
        input_dim=3,
        output_dim=2,
        hidden_layers=2,
        hidden_units=8,
        rngs=rngs_0,
    )

    f_1, theta_1 = mlp(
        input_dim=3,
        output_dim=2,
        hidden_layers=2,
        hidden_units=8,
        rngs=rngs_1,
    )

    x = jnp.array([1.0, -2.0, 0.5])

    y_0 = f_0(x, theta_0)
    y_1 = f_1(x, theta_1)

    assert bool(jnp.allclose(y_0, y_1))


def test_shared_rngs_advance_between_mlp_initialisations() -> None:
    rngs = nnx.Rngs(params=0)

    count_before = int(rngs.params.count[...])

    f_0, theta_0 = mlp(
        input_dim=3,
        output_dim=2,
        hidden_layers=2,
        hidden_units=8,
        rngs=rngs,
    )

    count_after_first_mlp = int(rngs.params.count[...])

    f_1, theta_1 = mlp(
        input_dim=3,
        output_dim=2,
        hidden_layers=2,
        hidden_units=8,
        rngs=rngs,
    )

    count_after_second_mlp = int(rngs.params.count[...])

    x = jnp.array([1.0, -2.0, 0.5])

    y_0 = f_0(x, theta_0)
    y_1 = f_1(x, theta_1)

    assert not bool(jnp.allclose(y_0, y_1))
    assert count_after_first_mlp > count_before
    assert count_after_second_mlp > count_after_first_mlp


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
def test_mlp_raises_on_invalid_deep_layers(
    hidden_dims: Sequence[int] | None,
    hidden_layers: int | None,
    hidden_units: int | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        mlp(
            input_dim=3,
            output_dim=2,
            hidden_dims=hidden_dims,
            hidden_layers=hidden_layers,
            hidden_units=hidden_units,
            seed=0,
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
def test_mlp_rejects_invalid_configurations(
    input_dim: int,
    output_dim: int,
    dropout_rate: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        mlp(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=1,
            hidden_units=8,
            dropout_rate=dropout_rate,
            seed=0,
        )


def test_mlp_learns_linear_problem() -> None:
    x_train = jnp.linspace(-1.0, 1.0, 64).reshape(-1, 1)
    y_train = 2.0 * x_train + 0.5

    f, theta = mlp(
        input_dim=1,
        output_dim=1,
        hidden_dims=[],
        activation="identity",
        norm=None,
        dropout_rate=0.0,
        seed=0,
    )

    optimiser = optax.adam(learning_rate=0.05)
    opt_state = optimiser.init(theta)

    def predict_one(theta: nnx.State, x: jax.Array) -> jax.Array:
        return f(x, theta)

    def loss_fn(theta: nnx.State) -> jax.Array:
        preds = jax.vmap(lambda x: predict_one(theta, x))(x_train)
        return jnp.mean((preds - y_train) ** 2)

    @jax.jit
    def train_step(
        theta: nnx.State,
        opt_state: optax.OptState,
    ) -> tuple[nnx.State, optax.OptState, jax.Array]:
        loss, grads = jax.value_and_grad(loss_fn)(theta)
        updates, opt_state = optimiser.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, loss

    for _ in range(300):
        theta, opt_state, loss = train_step(theta, opt_state)

    assert float(loss) < 1e-4

    final_loss = loss_fn(theta)

    assert float(final_loss) < 1e-5


def test_mlp_learns_nonlinear_problem() -> None:
    x_train = jnp.linspace(-1.0, 1.0, 128).reshape(-1, 1)
    y_train = x_train**3 + 0.3 * x_train - 0.2

    f, theta = mlp(
        input_dim=1,
        output_dim=1,
        hidden_layers=2,
        hidden_units=16,
        activation="relu",
        norm=None,
        dropout_rate=0.0,
        seed=0,
    )

    optimiser = optax.adam(learning_rate=0.01)
    opt_state = optimiser.init(theta)

    def predict_one(theta: nnx.State, x: jax.Array) -> jax.Array:
        return f(x, theta)

    def loss_fn(theta: nnx.State) -> jax.Array:
        preds = jax.vmap(lambda x: predict_one(theta, x))(x_train)
        return jnp.mean((preds - y_train) ** 2)

    @jax.jit
    def train_step(
        theta: nnx.State,
        opt_state: optax.OptState,
    ) -> tuple[nnx.State, optax.OptState, jax.Array]:
        loss, grads = jax.value_and_grad(loss_fn)(theta)
        updates, opt_state = optimiser.update(grads, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        return theta, opt_state, loss

    initial_loss = loss_fn(theta)

    for _ in range(1_000):
        theta, opt_state, loss = train_step(theta, opt_state)

    final_loss = loss_fn(theta)

    assert float(final_loss) < float(initial_loss)
    assert float(final_loss) < 1e-5
