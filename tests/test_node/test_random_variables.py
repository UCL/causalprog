import numpy as np

from causalprog.graph import (
    ContinuousRandomVariableNode,
    DiscreteRandomVariableNode,
    Graph,
)


def test_random_variable_node():
    node = ContinuousRandomVariableNode(label="X")
    assert np.isclose(node.evaluate(X=2.0), 2.0)


def test_missing_input(raises_context):
    node = ContinuousRandomVariableNode(label="X")

    with raises_context(ValueError("Missing input for node")):
        node.evaluate()


def test_invalid_discrete_node_value(raises_context):
    node = DiscreteRandomVariableNode(label="Y", values=[-0.5, 0.0, 0.5])
    with raises_context(ValueError("Invalid value for DiscreteRandomVariableNode")):
        node.evaluate(Y=-1.0)


def test_evaluate_down_graph():
    graph = Graph(label="G")

    graph.add_node(ContinuousRandomVariableNode(label="x"))
    graph.add_node(DiscreteRandomVariableNode(label="y", values=[-0.5, 0.0, 0.5]))
    graph.add_node(ContinuousRandomVariableNode(label="z", compute=lambda x, y: x + y))

    assert np.isclose(graph.get_node("z").evaluate(x=3.0, y=-0.5), 2.5)
