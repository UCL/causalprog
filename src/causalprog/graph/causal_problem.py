from typing import TYPE_CHECKING

from .graph import Graph

if TYPE_CHECKING:
    from .node import ParameterNode


class CausalProblem(Graph):
    """"""

    def __init__(self, label: str):
        super().__init__(label)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Clear all current values of parameter nodes."""
        self.set_parameters(**{node.label: None for node in self.parameter_nodes})

    def set_parameters(self, **parameter_values: float | None) -> None:
        """
        Set the current value of all parameter nodes to the new values.

        Parameter nodes are identified by variable name. Absent parameters retain their
        current value.
        """
        for name, new_value in parameter_values.items():
            node: ParameterNode = self.get_node(name)
            node.current_value = new_value
