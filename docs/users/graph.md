# Creating a using graphs in causalprog

In causalprog, directed acyclic graphs (DAGs) are used to represent causal problems.
This documentation page describes how these graphs can be created and used, and is aimed
at users of the library. Documentation for library developers can be found in [the documentation
for developers](../developers/graph.md).

# Creating a graph

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
