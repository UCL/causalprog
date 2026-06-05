"""Regression functions (act as constraints for causal problems)."""

from numpy.typing import NDArray

from .graph import Graph


def build_regression_function(
    graph: Graph, trained_parameters: dict[str, NDArray], quadrature
) -> None:
    """"""
