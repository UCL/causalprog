# Glossary

Definitions of terms and abbreviations that are used across the `causalprog` documentation and codebase.

## Glossary of Terms

### Causal Estimand

See [Causal Problem](#causal-problem).

The objective function (typically denoted $\sigma$) of a causal problem is referred to as the causal estimand.
It typically represents some quantity of interest that we cannot directly measure nor obtain data for.

Although the causal estimand is formally defined as a function of the model parameters, in practice it is often defined implicitly in terms of (moments of) the RVs $X_i$ of the causal model.
As a simple example, if we have just a single RV $X\sim\mathcal{N}(\mu, \nu)$ in a causal model with two model parameters $\Theta = \{\mu, \nu\}$, then the causal estimand $\sigma$ is typically going to be specified by

$$ \sigma(\mu, \nu) = \mathbb{E}[X] $$

rather than

$$ \sigma(\mu, \nu) = \mu. $$

### Causal Model

Let $X_1, X_2, ..., X_I$ ( $I\in\mathbb{N}$ ) be a collection of RVs.
For each $i$, let $V_i \subset {1, ..., i-1}$ be the (possibly empty) collection of the indices of the RVs that $X_i$ is dependant upon.
Note that we are assuming (WLOG) that the RVs are indexed somewhat sequentially in terms of causality / dependency.

The structure imposed by the $V_i$ allows for the relationships between the $X_i$ to be realised as a DAG, $G$.
The nodes represent the random variables $X_i$, and as such we use the notation $X_i$ interchangeably when referring to the RVs or nodes of $G$$.
An edge directed into $X_i$ from $X_k$ (where $(k < i)$) encodes that the distribution of $X_i$ depends on $X_k\$.

Let $D_i = \otimes_{k\in V_i} X_k$ and for each $X_i$.
Assume there exists a function $f_{X_i}$, deterministic in its arguments, and with $\mathrm{dom}(f_{X_i}) = D_i$, such that $X_i \sim f_{X_i}(\{X_k\})$.
That is to say, for each $i$ there is some deterministic function $f_{X_i}$ such that, given realisations of $X_k, k\in V_i$, $f_{X_i}$ fully describes the distribution of $X_i$.

The (parametrised) _causal model_ is then $\Theta := \left\{ f_{X_i} \right\}_{i\leq n}$.

### Causal Problem

See [Causal Model](#causal-model).

Given a causal model $\Theta = \{\theta_j\}$ of RVs $X_i$, let $\sigma: \Theta \rightarrow \mathbb{R}$ and $\phi_k: \Theta\rightarrow\mathbb{R}$ be deterministic functions of the (model parameters describing) the RVs $X_i$.
Let $\phi_{\mathrm{data}, k}$ be observed, empirical data for the quantities $\phi_k$, and let $\eplsion_k > 0$ be the tolerance in the observed data.

A causal problem is then an optimisation problem of the form

$$ \max_{\Theta} / \min_{\Theta} \sigma(\Theta), \quad \text{subject to } \quad \vert\vert \phi_{\mathrm{data}, k} - \phi_k(\Theta) \vert\vert\leq \epsilon, \quad \forall k, $$

where a suitable norm ($\vert\vert\cdot\vert\vert$) is chosen for each constraint.

The solution to this problem is the maximum / minimum value of $\sigma$, and the corresponding collection of model parameters that give rise to these bounds.

The vectors $\phi_{\mathrm{data}} = (\phi_{\mathrm{data}, k})_k$, $\phi = (\phi_k)_k$, and $\epsilon = (\epsilon_k)$ may also be defined to write the constraints of this optimisation problem in vector form.

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

### Constraint (Function)

See [Causal Problem](#causal-problem).

The functions $\phi_k$ in a causal problem are referred to as the constraints (or constraint functions).
For given observed data $\phi_{\mathrm{data}, k}$ and tolerance in the data $\epsilon_k$, the constraint

$$ \vert\vert \phi_k - \phi_{\mathrm{data}, k} \vert\vert \leq \epsilon_k $$

appears in the corresponding causal problem.

The $phi_k$ represent quantities that can be estimated from a causal model, and which we have observed data for.

### Derived Parameter

See [Structural Equation](#structural-equation).

The arguments of the structural equations $f_{X_i}$ are referred to as derived parameters.
Note that this is in reference to the arguments themselves, not the (realisations of the) RVs that are passed into those arguments.

To be explicit, suppose we have a collection of two RVs $X\sim f_{X} := \mathcal{N}(0, 1)$ and $Y\sim f_{Y}(X) := \mathcal{N}(X, 1)$.
The structural equations are $f_X = \mathcal{N}(0, 1)$ (essentially a constant) and (abusing notation slightly) $f_Y(x) = \mathcal{N}(x, 1)$.

The argument $x$ to $f_Y$ is a derived parameter.

In the `causalprog` codebase, derived parameters of a RV $X_i$ are used to "mark" arguments (or parameters) of the structural equation $f_{X_i}$ that should be filled by realisations of a dependent variable $X_k$.

### Model Parameter

See [Causal Model](#causal-model)

Each member of the set $\Theta$ that fully parametrises a Causal Model are referred to as a model parameter, in the context above each $f_{X_i}$ would be seen as a model parameter.

However the structural equations $f_{X_i}$ can often themselves be further parametrised.
In such a case, the model parameters are those that fully parametrise the structural equations (and consequentially, $\Theta$).

For example, in equation (1), [Padh et. al.](https://arxiv.org/pdf/2202.10806), the structural equations are expressed as an expansion of (fixed) basis functions $\left\{\psi_{i, j}\right\}_{i\leq I, j\leq J}$, $J\in\mathbb{N}$:

$$ f_{X_i} = \sum_{j=1}^{J} \theta_{X_i}^{(j)}\psi_{i_j}. $$

Each $f_{X_i}$ is thus fully described in terms of their coefficients $\theta_{X_i} := (\theta_{X_i}^{(j)})_{j\leq J}$.
In such a case it is suitable to directly parametrise $\Theta = \left\{\theta_{X_i}\right\}_{i\leq I}$ rather than in terms of $f_{X_i}$, in which case each $\theta_{X_i}$ is a model parameter.

If the family of basis functions $\psi_{i,j}$ was not fixed, but also allowed to vary, the collection of model parameters would be

$$ \{ \theta_{X_i}, \psi_{i, j}\}. $$

### Structural Equation

See [Causal Model](#causal-model).

The structural equation of the RV $X_i$ is the deterministic function $f_{X_i}$.
Given realisations (or samples) of the RVs $X_k, k\in V_i$ that $X_i$ is dependent on, the structural equation fully describes the distribution of $X_i$.

You may also see the notation

$$ X_i \vert \{X_k\}_{k\in V_i} = f_{X_i}(\{X_k\}) $$

used.

### Tolerance (of a Constraint)

See [Causal Problem](#causal-problem).

The values $\epsilon_k$ that appears in a causal problem is referred to as the tolerance (in the data corresponding to $\phi_k$).

## Abbreviations

- DAG: [Directed Acyclic Graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph)
- RV(s): [Random Variable(s)](https://en.wikipedia.org/wiki/Random_variable)
- WLOG: [Without Loss Of Generality](https://en.wikipedia.org/wiki/Without_loss_of_generality)
