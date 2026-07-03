"""Tests for explicit-parameter MLP builders."""

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


def test_mlp_output_shape_with_hidden_layers() -> None:
    f, theta = mlp(
        input_dim=3,
        output_dim=2,
        hidden_layers=2,
        hidden_units=8,
        seed=0,
    )

    x = jnp.ones((3,))

    y = f(x, theta)

    assert y.shape == (2,)


def test_mlp_output_shape_with_explicit_hidden_dims() -> None:
    f, theta = mlp(
        input_dim=3,
        output_dim=2,
        hidden_dims=[8, 4],
        seed=0,
    )

    x = jnp.ones((3,))

    y = f(x, theta)

    assert y.shape == (2,)


def test_mlp_allows_no_hidden_layers() -> None:
    f, theta = mlp(
        input_dim=3,
        output_dim=2,
        hidden_dims=[],
        seed=0,
    )

    x = jnp.ones((3,))

    y = f(x, theta)

    assert y.shape == (2,)


@pytest.mark.parametrize("norm", [None, "layernorm", "rmsnorm"])
def test_mlp_norm_options(norm: str | None) -> None:
    f, theta = mlp(
        input_dim=3,
        output_dim=2,
        hidden_layers=2,
        hidden_units=8,
        norm=norm,
        seed=0,
    )

    x = jnp.ones((3,))

    y = f(x, theta)

    assert y.shape == (2,)


@pytest.mark.parametrize("activation", ["relu", "gelu", "silu", "tanh", "identity"])
def test_mlp_activation_options(activation: str) -> None:
    f, theta = mlp(
        input_dim=3,
        output_dim=2,
        hidden_layers=1,
        hidden_units=8,
        activation=activation,
        seed=0,
    )

    x = jnp.ones((3,))

    y = f(x, theta)

    assert y.shape == (2,)


def test_mlp_is_jittable_with_non_batched_input() -> None:
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

    y = apply(theta, x)

    assert y.shape == (2,)


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


def test_mlp_apply_train_requires_rngs_when_dropout_enabled() -> None:
    f, theta = mlp(
        input_dim=3,
        output_dim=2,
        hidden_layers=1,
        hidden_units=8,
        dropout_rate=0.5,
        seed=0,
    )

    x = jnp.ones((3,))

    with pytest.raises(ValueError, match="rngs must be provided"):
        f.apply_train(x, theta)


def test_mlp_apply_train_with_dropout_rngs() -> None:
    f, theta = mlp(
        input_dim=3,
        output_dim=2,
        hidden_layers=1,
        hidden_units=8,
        dropout_rate=0.5,
        seed=0,
    )

    x = jnp.ones((3,))

    y = f.apply_train(
        x,
        theta,
        rngs=nnx.Rngs(dropout=1),
    )

    assert y.shape == (2,)


def test_mlp_rejects_missing_hidden_configuration() -> None:
    with pytest.raises(ValueError, match="Either hidden_dims or both hidden_layers"):
        mlp(
            input_dim=3,
            output_dim=2,
        )


def test_mlp_rejects_mixed_hidden_configuration() -> None:
    with pytest.raises(ValueError, match="Pass either hidden_dims or hidden_layers"):
        mlp(
            input_dim=3,
            output_dim=2,
            hidden_dims=[8],
            hidden_layers=1,
            hidden_units=8,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"input_dim": 0, "output_dim": 1, "hidden_dims": [8]}, "input_dim"),
        ({"input_dim": 1, "output_dim": 0, "hidden_dims": [8]}, "output_dim"),
        ({"input_dim": 1, "output_dim": 1, "hidden_dims": [0]}, "hidden_dims"),
        (
            {
                "input_dim": 1,
                "output_dim": 1,
                "hidden_dims": [8],
                "dropout_rate": 1.0,
            },
            "dropout_rate",
        ),
    ],
)
def test_mlp_rejects_invalid_configuration(
    kwargs: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        mlp(**kwargs)


def test_mlp_learns_easy_linear_problem_with_non_batched_inputs() -> None:
    """Train a no-hidden-layer MLP on single-example linear regression calls."""

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
        """Apply the MLP to one non-batched input."""
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

    assert float(final_loss) < 1e-4
