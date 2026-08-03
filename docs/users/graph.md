# Creating a using graphs in causalprog

In causalprog, directed acyclic graphs (DAGs) are used to represent causal problems.
This documentation page describes how these graphs can be created and used, and is aimed
at users of the library. Documentation for library developers can be found in [the documentation
for developers](../developers/graph.md).

# Creating a graph
A new (empty) graph can be created by directly calling the `Graph` class. There is one required
keyword argument: a label that is used to identify the graph. This label can be any string.

```python
from causalprog.graph import Graph

graph = Graph(label="I like graphs")
```

Nodes and edges can then be added to the graph using the method `add_node` and `add_edge`.
In this example, two data nodes that represent scalars are added with a directed edge pointing
from the first node to the second node.

```python
from causalprog.graph import DataNode

node1 = DataNode(label="first_node")
node2 = DataNode(label="second_node")

graph.add_node(node1)
graph.add_node(node2)
graph.add_edge(node1, node2)
```

The method `add_edge` can take either node labels or the nodes themselves as inputs, so the above
snippet could be rewritten more concisely as follows.

```python
from causalprog.graph import DataNode

graph.add_node(DataNode(label="first_node"))
graph.add_node(DataNode(label="second_node"))
graph.add_edge("first_node", "second_node")
```

If node objects are passed into `add_edge`, then `add_edge` will internally call `add_node` on
its inputs if they are not already nodes in the graph. Hence, the above snippets could be
written even more succinctly (but maybe less clearly) as follows.

```python
from causalprog.graph import DataNode

graph.add_edge(DataNode(label="first_node"), DataNode(label="second_node"))
```

When a node explicitly depends on other nodes, then edges will be automatically added to the
graph when the node is added. For example, in the following snippet a data node representing a
scalar and a continuous random variable node are added to the graph.

```
from causalprog.graph import DataNode, ContinuousRandomVariableNode

graph.add_node(DataNode(label="first_node"))
graph.add_node(ContinuousRandomVariableNode(label="X", parents=["first_node"]))
```

As `"first_node"` is a parent of the random variable node, the edge pointing from `"first_node"`
to `"X"` will automatically be added the graph in the final line of this snippet.

# Nodes

## `ConstantNode`

## `DataNode`

## Random variable nodes

`ContinuousRandomVariableNode` and `DiscreteRandomVariableNode`

## `DistributionNode`

# Using a graph

## Nodes and edges

## Root and leaf nodes

## Predecessors and successors

## Ordered nodes
`orders_nodes`, `roots_down_to_outcome`

## Sampling and evaluating nodes

# Graph algorithms

## `do`

## `evaluate` and `evaluate_down_to`

## `expectation` and `standard_deviation`
... and `moment`

## `replace_node`
