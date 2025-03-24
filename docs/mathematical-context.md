# Causal Problems and `causalprog`

INSERT FLAVOUR TEXT HERE ONCE YOURE DONE WILL

## Causal Problems

In full generality, we can describe a causal problem as follows.

Let $X_1, X_2, ..., X_n$ ( $n\in\mathbb{N}$ ) be a collection of random variables.
For each $i$, let $V_i \subset {1, ..., i-1}$ be the (possibly empty) collection of the indices of the random variables that $X_i$ is dependant upon.
Note that we are assuming (WLOG) that the random variables are indexed somewhat sequentially in terms of causality / dependency.

The structure imposed by the $V_i$ allows for the relationships between the $X_i$ to be realised as a DAG (Directed Acyclic Graph).
The nodes represent the random variables $X_i$, and an edge directed into $X_i$ from $X_k, (k < i)$ encodes that the distribution of $X_i$ depends on $X_k$.

Let $D_i = \otimes_{k\in V_i} X_k$ and for each $X_i$.
Assume there exists a function $f_{X_i}$, deterministic in its arguments, and with $\mathrm{dom}(f_{x_i}) = D_i$, such that $X_i \sim f_{X_i}$.
That is to say, for each $i$ there is some deterministic function $f_{X_i}$ such that, given realisations of $X_k, k\in V_i$, $f_{X_i}$ fully describes the distribution of $X_i$.
Define $\Theta := \left\{ f_{X_i} \right\}_{i\leq n}$, the _causal model_.

## Causal Estimands

Next, let $\sigma$ be a causal estimand of interest; that is to say, some quantity to be calculated from $\Theta$.
This could be something like the expectation or variance of one of the random variables $X_k$, for example.

One particular estimand of interest is the effect of _do_-ing something, given by the $\mathrm{do}$ operator.
The expected value of $X_k$ given that we "do" $X_l = x^*$ is written as $\mathbb{E}[ X_k \vert \mathrm{do}(X_l = x^*) ]$.
In general, this is different from $\mathbb{E}[ X_k \vert X_l = x^* ]$.
However, the $\mathrm{do}$ operator has a simple-to-explain effect if appealing to the causal problem $\Theta$; essentially replace (the function) $f_{X_l}$ in $\Theta$ with $x^*$ (or the appropriate mathematical object it defines).
