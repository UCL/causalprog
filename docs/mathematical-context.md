# Causal Problems and `causalprog`

TL;DR, `causalprog` solves

$$ \max_{\Theta} / \min_{\Theta} \sigma(\Theta), \quad \text{subject to } \quad \mathrm{dist}(\phi_\mathrm{data}, \phi_\mathrm{model}(\Theta))\leq \epsilon, $$

for a given

- (Parametrised) causal model $\Theta$,
- Causal estimand $\sigma$,
- Matching constraints $\phi$, where;
  - $\phi_\mathrm{data}$ is empirically observed values of the constraining values,
  - $\phi_\mathrm{model}$ is the analytical estimate of the constraining values,
  - $\mathrm{dist}$ is a non-negative valued distance function (such as a suitable norm),
- Tolerance parameter $\epsilon$.

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
The (parametrised) _causal model_ is then $\Theta := \left\{ f_{X_i} \right\}_{i\leq n}$.

### Further Parametrisations of $\Theta$

f is a basis, and gives $\theta$ param vector.

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

$$\mathbb{E}[ Y \vert \mathrm{do}(X = x^*) ] = \int f_{Y}(x^*, u) \ \mathrm{d}u \\ \approx \frac{1}{M} \sum_{i=1}^M f_Y(x^*, u^{(i)}),$$

with the approximation following from a Monte Carlo estimation of the integrand using samples $u^{(i)}$ drawn from $U$.

### Bounds for Causal Estimands

Given a causal problem $\Theta$ and causal estimand $\sigma$, it is natural to ask whether we can obtain bounds for $\sigma$, given some empirical observations of (observable variables of) $\Theta$.
We denote such empirical observations as $\phi_\mathrm{data}$, and we denote the expected values of these observations (given $\Theta$) as $\phi_\mathrm{model} = \phi_\mathrm{model}(\Theta)$.

To obtain suitable bounds on $\sigma$, we must solving the following (pair of) optimization problem(s):

$$ \max_\Theta / \min_\Theta \sigma(\Theta), \quad \text{subject to } \phi_\mathrm{data} = \phi_\mathrm{model}(\Theta). $$

Solving for the minimum provides the lower bound for $\sigma$, and solving for the maximum the upper bound.
The corresponding argument-min $\Theta_{\mathrm{min}}$ (respectively argument-max $\Theta_{\mathrm{max}}$) are the realisable causal models (IE the causal models that are consistent with our empirical observations) that attain the extrema of $\sigma$.

In practice, the equality constraint forcing the matching of $\phi_\mathrm{data}$ to $\phi_\mathrm{model}$ is relaxed to force consistency to within some tolerance $\epsilon$.
Computationally, this means that we are interested in solving the problem of

$$ \max_\Theta / \min_\Theta \sigma(\Theta), \quad \text{subject to } \vert\vert \phi_\mathrm{data} - \phi_\mathrm{model}(\Theta) \vert\vert^2 \leq \epsilon, $$

for some appropriate norm $\vert\vert\cdot\vert\vert$, such as the $L^2$-norm.
Such problems can be tackled using approaches based on Lagrangian multipliers, for example, seeking the saddle points of the augmented lagrangian

$$ \mathcal{L}(\Theta, \lambda) := \sigma(\Theta) - \lambda \left( \vert\vert \phi_\mathrm{data} - \phi_\mathrm{model}(\Theta) \vert\vert^2 - \epsilon\right), $$

and then determining whether they are maxima or minima.
