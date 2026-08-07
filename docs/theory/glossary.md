# Glossary

Definitions of terms and abbreviations that are used across the `causalprog` documentation and codebase.

## Glossary of Terms

### Causal Model

Let $X_1, X_2, ..., X_I$ ( $I\in\mathbb{N}$ ) be a collection of RVs.
For each $i$, let $V_i \subset {1, ..., i-1}$ be the (possibly empty) collection of the indices of the RVs that $X_i$ is dependant upon.
Note that we are assuming (WLOG) that the RVs are indexed somewhat sequentially in terms of causality / dependency.

The structure imposed by the $V_i$ allows for the relationships between the $X_i$ to be realised as a [DAG](#abbreviations), $\mathbb{G}$.
The nodes represent the random variables $X_i$, and as such we use the notation $X_i$ interchangeably when referring to the RVs or nodes of $\mathbb{G}$.
An edge directed into $X_i$ from $X_k$ (where $(k < i)$) encodes that the distribution of $X_i$ depends on $X_k\$.

Let $D_i = \otimes_{k\in V_i} X_k$ and for each $X_i$.
Assume there exists a function $f_{X_i}$, deterministic in its arguments, and with $\mathrm{dom}(f_{X_i}) = D_i$, such that $X_i \sim f_{X_i}(\{X_k\})$.
That is to say, for each $i$ there is some deterministic function $f_{X_i}$ such that, given realisations of $X_k, k\in V_i$, $f_{X_i}$ fully describes the distribution of $X_i$.

In practice these functions $f_{X_i}$ are typically further parametrised by some values $\theta_{X_i}$; for example when $f_{X_i}$ is some kind of neural network, $\theta_{X_i}$ would be the collection of weights and biases.

- We call $\theta = \bigcup_{X_i}\theta_{X_i}$ the model parameters.
- Given a value $\theta$ for the model parameters, we use $\mathbb{G}(\theta) := (\{X_i\}, \{V_i\}, \{\theta_{X_i}\}) $ to denote the causal model that it describes.
- The collection $\mathcal{S} = \bigcup_{\theta}\{ \mathbb{G}\theta \}$ is the set of all possible causal models that our description allows for.
- The collection of functions $\{f_{X_i}\}$ are referred to as the structural equations (of the causal models in $\mathcal{S}$).

It should be noted that the structural equations themselves could be used as the model parameters.
However the notation we have chosen here reflects that fact that the structural equations are typically themselves parameterised further, and also reflects the fact that directly optimising over an un-parametrised function space is not an easy task to be done programmatically.
Future development may allow for this to be done by considering, for example, the model parameters to be basis functions of the appropriate function spaces that the $f_{X_i}$ belong to.

### Causal Problem

Let

- $\mathcal{S}$ be a set of admissible causal models with model parameters $\theta$,
- $\mathcal{D}_{train}$ be the collection of all sets of training points for the models in $\mathcal{S}$,
- $B: \mathcal{D}_{train}\times\mathcal{S}\rightarrow [0, \infty)$ be a loss function,
- $d$ be a [causal response function](#causal-response),
- $\epsilon > 0$ be some [tolerance](#constraint-tolerance).

A causal problem is then an optimisation problem of the form

$$ \max_{\theta} / \min_{\theta} d(x; \theta), \quad \text{subject to } \quad \vert\vert B(\theta) \leq B(\theta^{\star}) + \epsilon, $$

where $x\in\mathcal{D}_{eval}$.

The "query bounds on the response function $d$" are the extreme values of $d$ that form the solution(s) to this problem.

### Causal Response

The causal response function is the objective function of a causal problem.
It typically represents some quantity of interest that we cannot directly measure nor obtain data for.

Formally, let $d: \mathcal{D}_{eval} \times \mathcal{S} \rightarrow \mathbb{R}$, where $\mathcal{S}$ is a set of [causal models](#causal-model) parametrised by $\theta$ and $\mathcal{D}_{eval}$ is the set of all evaluation points that models $\mathbb{G}(\theta)\in\mathcal{S}$ can be called at.
Then $d$ is a causal response function for the class of models $\mathcal{S}$.

Typically, the notation write $d(x; \theta) := d(x; \mathbb{G}(\theta))$ is used.
When $\theta$ is implicit, $d(x)$ may also be used.

Although the causal response function is formally defined as a function of the model parameters, in practice it is often defined implicitly in terms of (moments of) the RVs $X_i$ of the causal model.

### Constant Parameter

See [Structural Equation](#structural-equation).

"Constant Parameters" are a product of the computational implementation that `causalprog` uses to setup [Causal Problems](#causal-problem), and do not have a particular mathematical analogue.
In the `causalprog` codebase, constant parameters of a RV $X_i$ are used to "mask" arguments with constant values, that get passed to the computational functions that assemble the structural equations $f_{X_i}$.

When informing `causalprog` of the structural equation $f_{X_i}$ of a RV $X_i$, it is often convenient to use pre-existing functions from suitable libraries.
However, these functions may take more arguments, or be more general, than the description that $X_i$ needs.
Instead of wrapping such functions inside `lambda` expressions to mask the additional arguments, `causalprog` RVs have "constant parameters", which refer to the arguments of these functions which take a constant value and are not [derived parameters](#derived-parameter) nor [model parameters](#model-parameter).

To be explicit, suppose we have a collection of two RVs $X\sim f_{X} := \mathcal{N}(0, 1)$ and $Y\sim f_{Y}(X) := \mathcal{N}(X, 1)$.
The structural equations are $f_X = \mathcal{N}(0, 1)$ (essentially a constant) and (abusing notation slightly) $f_Y(x) = \mathcal{N}(x, 1)$.
Programmatically, we have a function `normal(mu, nu)` which evaluates to $\mathcal{\mu, \nu^2}$, that we want to use to describe $f_X$ and $f_Y$.

$f_X$ is a constant, but equates to evaluating `normal(0., 1.)`.
As such, we would call `mu` and `nu` "constant parameters" for the RV $X$, taking values 0 and 1 respectively.
$f_Y$ is non-constant, equating to evaluating `normal(X, 1.)`.
As such, we would refer to `nu` as a constant parameter for the RV $Y$, taking the value 1.
Note that `mu` is a derived parameter for $Y$.

### Loss Function

Let $\mathcal{S}$ be a collection of admissible models with model parameters $\theta$, and $\mathcal{D}_{train}$ be the collection of all sets of training points for the models in $\mathcal{S}$.
A loss function $B$ is then just a map $B: \mathcal{D}_{train}\times\mathcal{S}\rightarrow [0, \infty)$.

The value $B(\mathcal{D}; \theta)$ quantifies how well the model $\mathcal{G}(\theta)$ fits the observed dataset $\mathcal{D}$.
When the dataset is implicit, $B(\theta)$ is written instead.
By convention, lower values of $B$ indicate better fits.

Given a fixed dataset $\mathcal{D}$
$\theta^{\star} := \mathrm{argmin}_{\theta}

### Constraint Tolerance

See [Causal Problem](#causal-problem).

The values $\epsilon_k$ that appears in a causal problem is referred to as the tolerance (in the data corresponding to $\phi_k$).

## Abbreviations

- CE: [Causal Estimand](#causal-response), seen as a common abbreviation throughout the codebase.
- DAG: [Directed Acyclic Graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph)
- RV(s): [Random Variable(s)](https://en.wikipedia.org/wiki/Random_variable)
- WLOG: [Without Loss Of Generality](https://en.wikipedia.org/wiki/Without_loss_of_generality)
