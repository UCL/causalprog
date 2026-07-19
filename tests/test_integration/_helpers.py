from collections.abc import Iterable

import jax

from causalprog.graph.ricardo import MLPAlias


def vectorise_over_dict_args(f: MLPAlias, *dict_keys: Iterable[str]) -> MLPAlias:
    """Vectorise a pure function of dictionary arguments across the dictionary keys.

    This is essentially a wrapper around iterative applications of `jax.vmap` with the
    appropriate `in_axes` specified. The net effect is that if the input `f` was
    being called with a dictionary argument, whose keys were scalar-valued, the returned
    function can be called with the same dictionary argument whose keys are
    vector-valued, and returns a vector-valued output.

    Note that all vmap-ing is done along axis 0. If you want to pass in vector-values
    for some of the dictionary key inputs, ensure that they are aligned along the
    correct axis (each _row_ should be one value of the input).
    """
    vec_f = f
    all_keys = [key for key_list in dict_keys for key in key_list]
    for key in all_keys:
        vec_f = jax.vmap(
            vec_f,
            in_axes=tuple(
                {k: None if k != key else 0 for k in arg_keys} for arg_keys in dict_keys
            ),
        )
    return vec_f
