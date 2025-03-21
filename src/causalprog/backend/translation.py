"""Data structure for storing information mapping backend to frontend."""

from dataclasses import dataclass, field

from ._typing import ParamNameMap, StaticValues


@dataclass
class Translation:
    """
    Helper class for mapping frontend signatures to backend signatures.

    Attributes:
        backend_name (str): Name of the backend method that is being translated into
            a frontend method.
        frontend_name (str): Name of the frontend method that the backend method will
            be used as.
        param_map (ParamNameMap): See ``old_to_new_names`` argument to
            ``causalprog.backend._convert_signature``.
        frozen_args (StaticValues): See ``give_static_value`` argument to
            ``causalprog.backend._convert_signature``.

    """

    backend_name: str
    frontend_name: str
    param_map: ParamNameMap
    frozen_args: StaticValues = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.backend_name, str):
            msg = f"backend_name '{self.backend_name}' is not a string."
            raise TypeError(msg)
        if not isinstance(self.frontend_name, str):
            msg = f"frontend_name '{self.frontend_name}' is not a string."
            raise TypeError(msg)

        self.frozen_args = dict(self.frozen_args)
        self.param_map = dict(self.param_map)

        if not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in self.param_map.items()
        ):
            msg = "Parameter map must map str -> str."
            raise TypeError(msg)
        if not all(isinstance(key, str) for key in self.frozen_args):
            msg = "Frozen args must be specified by name (str)."
            raise TypeError(msg)
