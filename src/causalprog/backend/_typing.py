from inspect import _ParameterKind
from typing import Any, TypeAlias, TypeVar

ReturnType = TypeVar("ReturnType")
ParamNameMap: TypeAlias = dict[str, str]
ParamKind: TypeAlias = _ParameterKind
StaticValues: TypeAlias = dict[str, Any]
