from collections.abc import Callable
from typing import TypeAlias, TypeVar

import jax
from jax.tree_util import PyTreeDef as PyTreeDefNative

PyTree = TypeVar("PyTree")
PyTreeDef: TypeAlias = PyTreeDefNative

ModelParam: TypeAlias = dict[str, PyTree]
MLPAlias: TypeAlias = Callable[[dict[str, jax.Array], ModelParam], jax.Array]
