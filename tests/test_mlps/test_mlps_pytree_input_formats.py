import flax.nnx
import jax
import jax.numpy as jnp
import pytest

from causalprog._types import PyTree
from causalprog.mlps import FunctionalMLP, mlp


@pytest.mark.parametrize(
    (
        "input_fmt",
        "data_format",
        "ravel_is_identity_method",
    ),
    [
        pytest.param(1, jnp.array([1]), True, id="Scalar specified"),
        pytest.param(
            jnp.array(1), jnp.array([1]), True, id="Scalar specified by 0D-array"
        ),
        pytest.param(
            jnp.array([1]), jnp.array([1]), False, id="Scalar specified by 1D-array"
        ),
        pytest.param(
            jnp.array([2, 3]),
            jnp.array([2, 3]),
            False,
            id="Matrix-valued input format",
        ),
        pytest.param(
            {"a": 1, "b": jnp.array(1), "c": jnp.array([2, 3])},
            {"a": jnp.array([1]), "b": jnp.array([1]), "c": jnp.array([2, 3])},
            False,
            id="PyTree-style specification, upcast scalars to arrays",
        ),
    ],
)
def test_mlp_data_format(
    input_fmt: int | PyTree, data_format: PyTree, ravel_is_identity_method: bool
) -> None:
    """Check that `FunctionalMLP`s can track their input formats."""
    expected_ravel_method = (
        FunctionalMLP.identity
        if ravel_is_identity_method
        else FunctionalMLP.unravel_tree
    )

    f, _ = mlp(input_fmt, 1, hidden_dims=[2])

    assert jax.tree.all(
        jax.tree.map(lambda x, y: jnp.all(x == y), f.data_format, data_format)
    )
    # NOTE: We really do want to check via is, and on the private member here.
    assert f._data_to_column_vector is expected_ravel_method  # noqa: SLF001


def test_mlp_dict_to_col_consistency(build_mlp) -> None:
    """Sanity check that `_unravel_tree` provides a consistent ordering of the
    resulting column vector, in the event that dictionary keys are not created
    in the same order for otherwise identical inputs.
    """
    input_fmt = {"a": 1, "b": 2, "c": jnp.array([3, 4])}
    input_0 = {
        "a": 0.0,
        "b": jnp.arange(1, 3),
        "c": jnp.arange(3, 3 + 12).reshape(3, 4),
    }
    input_1 = {key: input_0[key] for key in reversed(input_0.keys())}

    f, _ = build_mlp(input_dim=input_fmt)

    input_0_col_vec = f.data_as_flat_array(input_0)
    input_1_col_vec = f.data_as_flat_array(input_1)

    assert jnp.allclose(input_0_col_vec, input_1_col_vec)


@pytest.mark.parametrize(
    "array_input",
    [
        pytest.param(jnp.array([1.0, -2.0, 0.5]), id="1D array input"),
        pytest.param(jnp.arange(12).reshape(3, 4), id="2D array input"),
    ],
)
def test_mlp_dict_and_array_input_consistency(
    build_mlp,
    array_input: jax.Array,
) -> None:
    """Sanity check that the unravelling of a PyTree with a single leaf
    is consistent with simply passing in the leaf as the direct input.
    """
    pytree_input_fmt = {"x": jnp.array(array_input.shape)}
    pytree_input = {"x": array_input}

    f_array, _ = build_mlp(input_dim=jnp.array(array_input.shape))
    f_pytree, _ = build_mlp(input_dim=pytree_input_fmt)

    array_col_vec = f_array.data_as_flat_array(array_input)
    pytree_col_vec = f_pytree.data_as_flat_array(pytree_input)

    assert jnp.allclose(array_col_vec, pytree_col_vec)


def test_mlp_pytree_input_matches_merged_model(build_mlp) -> None:
    """Check that `FunctionalMLP` correctly flattens PyTree inputs before evaluation."""
    input_fmt = {
        "a": 1,
        "b": jnp.array([2]),
        "c": jnp.array([2, 2]),
    }
    pytree_input = {
        "a": jnp.array(0.5),
        "b": jnp.array([1.0, -2.0]),
        "c": jnp.arange(4.0).reshape(2, 2),
    }

    f, theta = build_mlp(input_dim=input_fmt)
    merged_model = flax.nnx.merge(f.graphdef, theta)

    actual = f(pytree_input, theta)
    expected = merged_model(f.data_as_flat_array(pytree_input))

    assert actual.shape == expected.shape
    assert bool(jnp.allclose(actual, expected))
