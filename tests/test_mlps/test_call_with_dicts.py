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
    assert f._data_to_column_vector is expected_ravel_method
