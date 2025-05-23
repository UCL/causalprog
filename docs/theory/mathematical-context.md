# Causal Problems and `causalprog`

TL;DR, `causalprog` solves

$$ \max_{\Theta} / \min_{\Theta} \sigma(\Theta), \quad \text{subject to } \quad \mathrm{dist}(\phi_\mathrm{data}, \phi_\mathrm{model}(\Theta))\leq \epsilon, $$

given

- a (parametrised) causal model $\Theta$,
- a causal estimand $\sigma$,

and matching constraints $\phi = (\phi_j)$, where;

- $\phi_\mathrm{data}$ is empirically observed values of $\phi$,
- $\phi_\mathrm{model}$ is the analytical estimate of $\phi$ given $\Theta$,
- $\mathrm{dist}$ is a non-negative valued distance function (such as a suitable norm),
- a tolerance parameter $\epsilon$.

## Causal Problems

In full generality, we can describe a causal problem as follows.

Let $X_1, X_2, ..., X_I$ ( $I\in\mathbb{N}$ ) be a collection of random variables.
For each $i$, let $V_i \subset {1, ..., i-1}$ be the (possibly empty) collection of the indices of the random variables that $X_i$ is dependant upon.
Note that we are assuming (WLOG) that the random variables are indexed somewhat sequentially in terms of causality / dependency.

The structure imposed by the $V_i$ allows for the relationships between the $X_i$ to be realised as a DAG (Directed Acyclic Graph).
The nodes represent the random variables $X_i$, and as such we use the notation $X_i$ interchangeably when referring to the random variables or nodes of the associated DAG.
An edge directed into $X_i$ from $X_k$ (where $(k < i)$) encodes that the distribution of $X_i$ depends on $X_k$.

Let $D_i = \otimes_{k\in V_i} X_k$ and for each $X_i$.
Assume there exists a function $f_{X_i}$, deterministic in its arguments, and with $\mathrm{dom}(f_{x_i}) = D_i$, such that $X_i \sim f_{X_i}$.
That is to say, for each $i$ there is some deterministic function $f_{X_i}$ such that, given realisations of $X_k, k\in V_i$, $f_{X_i}$ fully describes the distribution of $X_i$.
We will refer to the $f_{X_i}$ as the _structural equation_ of $X_i$.
The (parametrised) _causal model_ is then $\Theta := \left\{ f_{X_i} \right\}_{i\leq n}$.

### Further Parametrisations of $\Theta$

A causal model $\Theta$ is parametrised by the structural equations, which themselves may be further parametrised.
In such a case, it is convenient to view the parametrisation of the structural equations as the parametrisation of $\Theta$.

For example, in equation (1), [Padh et. al.](https://arxiv.org/pdf/2202.10806), the structural equations are expressed as an expansion of (fixed) basis functions $\left\{\psi_{i, j}\right\}_{i\leq I, j\leq J}$, $J\in\mathbb{N}$:

$$ f_{X_i} = \sum_{j=1}^{J} \theta_{X_i}^{(j)}\psi_{i_j}. $$

Each $f_{X_i}$ is thus fully described in terms of their coefficients $\theta_{X_i} := (\theta_{X_i}^{(j)})_{j\leq J}$.
In such a case it is suitable to directly parametrise $\Theta = \left\{\theta_{X_i}\right\}_{i\leq I}$ rather than in terms of $f_{X_i}$.

## Causal Estimands

Next, let $\sigma$ be a causal estimand of interest; that is to say, some quantity to be calculated from $\Theta$, so $\sigma = \sigma(\Theta)$.
This could be something like the expectation or variance of one of the random variables $X_k$, for example.

For the time being, `causalprog` focuses on casual estimands that are predominantly integrals of some type.
In particular, the focus is on causal estimands that are the expectations (or possibly higher moments) of one of the random variables $X_k$ given some other conditions.
As such, computing the value of a causal estimand will be done largely through Monte Carlo sampling to approximate these integrands.
Since no assumption is made on the dimensionality of our random variables (and thus domains of the integrals), some of these integrals may require a large number of samples before giving a suitable approximation to the true value.

### The $\mathrm{do}$ Operator

One particular estimand of interest is the effect of _do_-ing something, described by the $\mathrm{do}$ operator.
The expected value of $X_k$ given that we "do" $X_l = x^*$ is written as $\mathbb{E}[ X_k \vert \mathrm{do}(X_l = x^*) ]$.
In general, this is different from $\mathbb{E}[ X_k \vert X_l = x^* ]$.

However, the $\mathrm{do}$ operator has a relatively simple-to-explain effect on $\Theta$; essentially replace (the function) $f_{X_l}$ with the constant $x^*$ (or the appropriate mathematical object it defines).

In the simple case where we have a random variable $Y$ which depends on $X$ (which we can control or fix) and $U$ (which we cannot control), we have that

$$\mathbb{E}[ Y \vert \mathrm{do}(X = x^*) ] = \int f_{Y}(x^*, u) \ \mathrm{d}u \\ \approx \frac{1}{M} \sum_{i=1}^M f_Y(x^*, u^{(i)}),$$

with the approximation following from a Monte Carlo estimation of the integrand using samples $u^{(i)}$ drawn from $U$.

## Matching Constraints

The matching constraints $\phi$ are observable quantities that ensure the theoretical model remains representative of our empirical observations.
When bounds for causal estimands are a concern, they serve to restrict the space of admissible causal models and thus tighten the obtainable bounds.

In general $\phi = (\phi_k)_{k\leq K}$, $K\in\mathbb{N}$ is a $K$-dimensional vector of matching constraints $\phi_k$.
The quantities described in each $\phi_k$ can be empirically observed to give some data vector $\phi_{\mathrm{data}}$, and estimated from a causal model $\Theta$ to give the theoretical values of these observables $\phi_{\mathrm{model}} = \phi_{\mathrm{model}}(\Theta)$.

A common set of $\phi_k$ are moment-matching constraints of the form

$$ \phi_{i, 0} = \mathbb{E}[X_i], \quad \phi_{i,j} = \mathbb{E}[X_i X_j], $$

for the _observable_ random variables $X_i$ (note that we can map the $(i,j)$-indexing to a single index $k$).

To attempt to infer the underlying causal model from observed data $\phi_{\mathrm{data}}$, one would have to examine the set of causal models for which $\phi_{\mathrm{data}} = \phi_{\mathrm{model}}$.
In practice, we are typically concerned with those causal models that are "close to" $\phi_{\mathrm{data}}$, rather than exactly equal, due to measurement inaccuracies or computational limitations.
As such, one may provide a suitable _distance function_ $\mathrm{dist}$ and tolerance parameter $\epsilon$.
$\mathrm{dist}(\phi_{\mathrm{data}}, \phi_{\mathrm{model}})$ is interpreted as a quantification of the difference between the observed and expected constraint values, with a value of 0 indicating equality.

$$\mathrm{dist}(\phi_{\mathrm{data}}, \phi_{\mathrm{model}}) = \vert\vert \phi_{\mathrm{data}} - \phi_{\mathrm{model}} \vert\vert^2_{L^2} $$

would be such an example.

## Bounds for Causal Estimands

Given a causal estimand $\sigma$, it is natural to ask whether we can provide bounds for $\sigma$ given some empirical observations of (observable variables of) $\Theta$.

To obtain suitable bounds on $\sigma$, and using the notation introduced in the sections above, we must solving the following (pair of) optimization problem(s):

$$ \max_\Theta / \min_\Theta \sigma(\Theta), \quad \text{subject to } \mathrm{dist}\left(\phi_\mathrm{data} - \phi_\mathrm{model}(\Theta) \right) \leq \epsilon, $$

Solving for the minimum provides the lower bound for $\sigma$, and solving for the maximum the upper bound.
The corresponding argument-min $\Theta_{\mathrm{min}}$ (respectively argument-max $\Theta_{\mathrm{max}}$) are the realisable causal models (IE the causal models that are consistent with our empirical observations up to the given tolerance) that attain the bounds of $\sigma$.

Such problems can be tackled using approaches based on Lagrangian multipliers, for example, seeking the saddle points of the augmented lagrangian

$$ \mathcal{L}(\Theta, \lambda) := \sigma(\Theta) - \lambda \left( \mathrm{dist}\left(\phi_\mathrm{data} - \phi_\mathrm{model}(\Theta) \right)- \epsilon\right), $$

and then determining whether they are maxima or minima.
