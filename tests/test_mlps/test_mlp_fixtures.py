import jax
import jax.numpy as jnp


def test_instrumented_mlp_matches_plain_mlp(
    build_mlp,
    fit_mlp_to_targets,
    seed: int,
) -> None:
    x_train = jax.random.normal(
        jax.random.key(seed + 1),
        shape=(32, 3),
    )
    y_train = x_train[:, :2]

    fitting_kwargs = {
        "learning_rate": 1e-2,
        "steps": 10,
        "dropout_seed": seed + 2,
    }

    plain_f, plain_theta = build_mlp(seed=seed)

    call_log: list[str] = []
    instrumented_f, instrumented_theta = build_mlp(
        listener=call_log,
        seed=seed,
    )

    plain_initial_predictions = plain_f(
        x_train,
        plain_theta,
    )
    instrumented_initial_predictions = instrumented_f(
        x_train,
        instrumented_theta,
    )

    assert bool(
        jnp.array_equal(
            plain_initial_predictions,
            instrumented_initial_predictions,
        )
    )

    (
        trained_plain_theta,
        plain_initial_loss,
        plain_final_loss,
    ) = fit_mlp_to_targets(
        plain_f,
        plain_theta,
        x_train,
        y_train,
        **fitting_kwargs,
    )

    (
        trained_instrumented_theta,
        instrumented_initial_loss,
        instrumented_final_loss,
    ) = fit_mlp_to_targets(
        instrumented_f,
        instrumented_theta,
        x_train,
        y_train,
        **fitting_kwargs,
    )

    plain_final_predictions = plain_f(
        x_train,
        trained_plain_theta,
    )
    instrumented_final_predictions = instrumented_f(
        x_train,
        trained_instrumented_theta,
    )

    assert bool(
        jnp.array_equal(
            plain_initial_loss,
            instrumented_initial_loss,
        )
    )
    assert bool(
        jnp.array_equal(
            plain_final_loss,
            instrumented_final_loss,
        )
    )
    assert bool(
        jnp.array_equal(
            plain_final_predictions,
            instrumented_final_predictions,
        )
    )

    assert call_log
