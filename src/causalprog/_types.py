from typing import TypeAlias, TypeVar

from jaxlib._jax.pytree import PyTreeDef as PyTreeDefNative

PyTree = TypeVar("PyTree")
PyTreeDef: TypeAlias = PyTreeDefNative
