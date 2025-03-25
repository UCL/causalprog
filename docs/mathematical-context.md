# Causal Problems and `causalprog`

INSERT FLAVOUR TEXT HERE ONCE YOURE DONE WILL

## Causal Problems

In full generality, we can describe a causal problem as follows.

Let $X_1, X_2, ..., X_n$ ( $n\in\mathbb{N}$ ) be a collection of random variables.
For each $i$, let $V_i \subset {1, ..., i-1}$ be the (possibly empty) collection of the indices of the random variables that $X_i$ is dependant upon.
Note that we are assuming (WLOG) that the random variables are indexed somewhat sequentially in terms of causality / dependency.

The structure imposed by the $V_i$ allows for the relationships between the $X_i$ to be realised as a DAG (Directed Acyclic Graph).
The nodes represent the random variables $X_i$, and as such we use the notation $X_i$ interchangeably when referring to the random variables or nodes of the associated DAG.
An edge directed into $X_i$ from $X_k$ (where $(k < i)$) encodes that the distribution of $X_i$ depends on $X_k$.

Let $D_i = \otimes_{k\in V_i} X_k$ and for each $X_i$.
Assume there exists a function $f_{X_i}$, deterministic in its arguments, and with $\mathrm{dom}(f_{x_i}) = D_i$, such that $X_i \sim f_{X_i}$.
That is to say, for each $i$ there is some deterministic function $f_{X_i}$ such that, given realisations of $X_k, k\in V_i$, $f_{X_i}$ fully describes the distribution of $X_i$.
The _causal model_ is then $\Theta := \left\{ f_{X_i} \right\}_{i\leq n}$.

## Causal Estimands and the $\mathrm{do}$ Operator

Next, let $\sigma$ be a causal estimand of interest; that is to say, some quantity to be calculated from $\Theta$, so $\sigma = \sigma(\Theta)$.
This could be something like the expectation or variance of one of the random variables $X_k$, for example.

One particular estimand of interest is the effect of _do_-ing something, given by the $\mathrm{do}$ operator.
The expected value of $X_k$ given that we "do" $X_l = x^*$ is written as $\mathbb{E}[ X_k \vert \mathrm{do}(X_l = x^*) ]$.
In general, this is different from $\mathbb{E}[ X_k \vert X_l = x^* ]$.

However, the $\mathrm{do}$ operator has a relatively simple-to-explain effect on $\Theta$; essentially replace (the function) $f_{X_l}$ with the constant $x^*$ (or the appropriate mathematical object it defines).

### Functional Forms of Causal Estimands

For the time being, `causalprog` focuses on casual estimands that are predominantly integrals of some type.
In particular, the focus is on causal estimands that are the expected value of one of the random variables $X_k$ given some other conditions (such as the $\mathrm{do}$ operation being performed).
As such, computing the value of a causal estimand will be done largely through Monte Carlo sampling to approximate these integrands.
Since no assumption is made on the dimensionality of our random variables (and thus domains of the integrals), some of these integrals may require a large number of samples before giving a suitable approximation to the true value.

In the simple case where we have a random variable $Y$ which depends on $X$ and $U$, we have that

$$\mathbb{E}[ Y \vert \mathrm{do}(X = x^*) ] = \int f_{Y}(x^*, u) \mathrm{d}u \\ \approx \frac{1}{M} \sum_{i=1}^M f_Y(x^*, u^{(i)}),$$

with the approximation following from a Monte Carlo estimation of the integrand using samples $u^{(i)}$ drawn from $U$.

### Bounding Causal Estimands

Given a causal problem $\Theta$ and causal estimand $\sigma$, it is natural to ask whether we can obtain bounds for $\sigma$, given some empirical observations of (observable variables of) $\Theta$.
In general, this means that we are interested in solving the following the following optimization problem(s):
