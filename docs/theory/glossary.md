# Glossary

Definitions of terms and abbreviations that are used across the `causalprog` documentation and codebase.

## Glossary of Terms

### Causal Model

Let $X_1, X_2, ..., X_I$ ( $I\in\mathbb{N}$ ) be a collection of RVs.
For each $i$, let $V_i \subset {1, ..., i-1}$ be the (possibly empty) collection of the indices of the RVs that $X_i$ is dependant upon.
Note that we are assuming (WLOG) that the RVs are indexed somewhat sequentially in terms of causality / dependency.

The structure imposed by the $V_i$ allows for the relationships between the $X_i$ to be realised as a DAG, $G$.
The nodes represent the random variables $X_i$, and as such we use the notation $X_i$ interchangeably when referring to the RVs or nodes of $G$$.
An edge directed into $X_i$ from $X_k$ (where $(k < i)$) encodes that the distribution of $X_i$ depends on $X_k\$.

Let $D_i = \otimes_{k\in V_i} X_k$ and for each $X_i$.
Assume there exists a function $f_{X_i}$, deterministic in its arguments, and with $\mathrm{dom}(f_{X_i}) = D_i$, such that $X_i \sim f_{X_i}$.
That is to say, for each $i$ there is some deterministic function $f_{X_i}$ such that, given realisations of $X_k, k\in V_i$, $f_{X_i}$ fully describes the distribution of $X_i$.

The (parametrised) _causal model_ is then $\Theta := \left\{ f_{X_i} \right\}_{i\leq n}$.

### Model Parameter

See [Causal Model](#causal-model)

The collection of parameters $\Theta$ that fully parameterise a Causal Model are referred to as its model parameters.

### Structural Equation

See [Causal Model](#causal-model).

The structural equation of the RV $X_i$ is the deterministic function $f_{X_i}$.
Given realisations (or samples) of the RVs $X_k, k\in V_i$ that $X_i$ is dependent on, the structural equation fully describes the distribution of $X_i$.

You may also see the notation

$$ X_i \vert \{X_k\}_{k\in V_i} \sim f_{X_i} $$

used.

## Abbreviations

- DAG: [Directed Acyclic Graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph)
- RV(s): [Random Variable(s)](https://en.wikipedia.org/wiki/Random_variable)
- WLOG: [Without Loss Of Generality](https://en.wikipedia.org/wiki/Without_loss_of_generality)
