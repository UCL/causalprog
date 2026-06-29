"""Functions to create example graphs."""

from collections.abc import Callable

from causalprog.graph import (
    ContinuousRandomVariableNode,
    DataNode,
    DiscreteRandomVariableNode,
    Graph,
)


def example_model(
    *,
    label: str = "G",
    l_len: int = 1,
    z_len: int = 1,
    k: int = 10,
    compute_u_x: Callable,
    compute_u_y: Callable,
    compute_phi_x: Callable,
    compute_x: Callable,
    compute_y: Callable,
) -> Graph:
    """
    Create a graph representing the example model.

    Args:
        label: The label of the graph.
        l_len: The number of entries in the vector data node L.
        z_len: The number of entries in the vector data node Z.
        k: The maximum value that could be taken by the mixture indicator C.
        compute_u_x: Compute UX given the value of C.
        compute_u_y: Compute UY given the value of C.
        compute_phi_x: Compute PhiX given the value of L.
        compute_x: Compute X given the values of Z, PhiX and UX.
        compute_y: Compute Y given the values of X and UY.

    Returns:
        A graph

    """
    graph = Graph(label=label)

    graph.add_node(DataNode(label="L", shape=(l_len,)))
    graph.add_node(DataNode(label="Z", shape=(z_len,)))
    graph.add_node(
        DiscreteRandomVariableNode(
            label="C", values=[float(i) for i in range(1, k + 1)]
        )
    )
    graph.add_node(
        ContinuousRandomVariableNode(label="UX", compute=compute_u_x, parents=["C"])
    )
    graph.add_node(
        ContinuousRandomVariableNode(label="UY", compute=compute_u_y, parents=["C"])
    )
    graph.add_node(
        ContinuousRandomVariableNode(label="PhiX", compute=compute_phi_x, parents=["L"])
    )
    graph.add_node(
        ContinuousRandomVariableNode(
            label="X", compute=compute_x, parents=["Z", "PhiX", "UX"]
        )
    )
    graph.add_node(
        ContinuousRandomVariableNode(label="Y", compute=compute_y, parents=["X", "UY"])
    )

    return graph
