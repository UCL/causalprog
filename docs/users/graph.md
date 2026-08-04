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
This section of the documentation details the different types of graph node available in
causalprog.

## `ConstantNode`
A `ContantNode` is a node that represents a known constant value. These nodes have two required
keyword arguments that must be passed: a `label` for the node and the `value` that the node
represents:

```python
import jax.numpy as jnp
from causalprog.graph import ConstantNode

one = ConstantNode(label="one", value=1.0)
vector = ConstantNode(label="v", value=jnp.array([1.0, 1.0, 2.0]))
```

## `DataNode`
A `DataNode` is a node that represents a constant value that is not known when the node is created.
These nodes has one required keyword arguments that must be passed: a `label` for the node. They
can take the `shape` of the data that the node represents as an additional keyword argument, with
the default shape being `()` for a scalar value.

```python
import jax.numpy as jnp
from causalprog.graph import DataNode

scalar = DataNode(label="my_scalar")
vector = DataNode(label="my_vector", shape=(5, ))
matrix = DataNode(label="my_matrix", shape=(4, 2))
```

## Random variable nodes
Random variable node represent random variables (RVs) in a causal problem. There are two types of
random variable node in causalprog: `ContinuousRandomVariableNode` and `DiscreteRandomVariableNode`.
Both of these must be passed the a `label` for the node as a required keyword argument, and can
take a number of additional keyword arguments: the `shape` of the data that the RV outputs, a
function to `compute` the value of the RV from the values of its parents, and a list of `parents`
of the RV node. Discrete RV nodes must be passed an addition required keyword argument: a list of
possible `values` that the RV can output.

TODO: describe `compute`

```python
from causalprog.graph import ContinuousRandomVariableNode, DiscreteRandomVariableNode

TODO: example
```

## `DistributionNode`
Distrubution nodes represent the values of random variables (RVs) that can be sampled from.
These nodes formed part of an earlier experimental version of the library and are not used in the
current demonstation applications.

# Ricardo's graph
Many of the examples in causalprog use an example graph for a problem proposed by Ricardo Silva:

![Illustration of the continuous treatment model that we discuss.](../diagrams/continuous-treatment-model.svg)

causalprog provides a helper function to quickly generate this graph. This function must be passed
five required keyword arguments that tell the graph how to compute the nodes `"u_x"`, `"u_y"`,
`"phi_x"`, `"x"` and `"y"` from their parents.

```python
from causalprog.graph.ricardo import example_model

graph = example_model(
    compute_u_x=lambda values: values["c"][0] + 1.0,
    compute_u_y=lambda values: values["c"][1] * 2,
    compute_phi_x=lambda values: values["l"],
    compute_x=lambda values: values["z"] + values["phi_x"] - values["u_x"],
    compute_y=lambda values: values["x"] * values["u_y"],
)
```

# Using a graph
This section of the documentation demonstrated how graphs created using causalprog can be used.

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
