"""Fixtures for MLP tests."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from causalprog.mlps import FunctionalMLP, mlp


@pytest.fixture
def build_mlp() -> Callable[..., tuple[FunctionalMLP, nnx.State]]:
    """Return a builder for an MLP with standard test defaults."""
    default_kwargs = {
        "input_dim": 3,
        "output_dim": 2,
        "hidden_layers": 3,
        "hidden_units": 8,
    }

    def _build_mlp(
        **overrides: Any,
    ) -> tuple[FunctionalMLP, nnx.State]:
        kwargs = dict(default_kwargs)
        kwargs.update(overrides)
        return mlp(**kwargs)

    return _build_mlp


@pytest.fixture
def fit_mlp_to_targets() -> Callable[
    ...,
    tuple[nnx.State, jax.Array, jax.Array],
]:
    """Return a function that fits an MLP to supplied targets."""

    def _fit_mlp_to_targets(
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

        def train_loss_fn(
            theta: nnx.State,
            dropout_key: jax.Array,
        ) -> jax.Array:
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

    return _fit_mlp_to_targets
