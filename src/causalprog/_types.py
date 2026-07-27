from typing import TypeAlias, TypeVar

from jax.tree_util import PyTreeDef as PyTreeDefNative

PyTree = TypeVar("PyTree")
PyTreeDef: TypeAlias = PyTreeDefNative
