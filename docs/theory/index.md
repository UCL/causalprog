# Theory

On this page we attempt to give a brief introduction to the mathematical framework that `causalprog` operates in.

## Causal Models

Causal models are the central building blocks around which `causalprog` operates.
At the highest level, one can just think of _a causal model_ as some black-box object $G$ that takes some data point $x\in\mathcal{D}_{eval}\subset\mathbb{R}^N$ and predicts an outcome $G(x)\in\mathbb{R}^M$ based on $x$.

### Definition

Formally however, let $X_1, X_2, ..., X_I$ ( $I\in\mathbb{N}$ ) be a collection of RVs.
For each $i$, let $V_i \subset {1, ..., i-1}$ be the (possibly empty) collection of the indices of the RVs that $X_i$ is dependant upon.
Note that we are assuming (WLOG) that the RVs are indexed somewhat sequentially in terms of causality / dependency.

Let $D_i = \otimes_{k\in V_i} X_k$, and for each $X_i$ assume there exists a function $f_{X_i}$, deterministic in its arguments, with $\mathrm{dom}(f_{X_i}) = D_i$, such that $X_i \sim f_{X_i}(\{X_k\})$.
That is to say, for each $i$ there is some deterministic function $f_{X_i}$ such that

- Given realisations of $X_k, k\in V_i$, $f_{X_i}$ fully describes the distribution of $X_i$.
- As an alternative interpretation, given values $x_k$ taken by the random variables $X_k, k\in V_i$, the value $f_{X_i}(x_1, ..., x_k)$ is the model's prediction of the value of $X_i \vert X_k = x_k, k\in V_i$.

The collection $G = (\{X_i\}, \{V_i\}, \{f_{X_i}\})$ is a causal model.
Each $f_{X_i}$ is referred to as a "structural equation" of $G$.
In practice these functions $f_{X_i}$ are typically further parametrised by some values $\theta_{X_i}$; for example when $f_{X_i}$ is some kind of neural network, $\theta_{X_i}$ would be the collection of weights and biases.
As such we will typically work with the notation $G = (\{X_i\}, \{V_i\}, \theta)$.

For fixed $\{X_i\}, \{V_i\}$, we can define the set

$$\mathcal{G} = \bigcup_{\theta} (\{X_i\}, \{V_i\}, \{f_{X_i}\}), $$

where each element $G(\theta)\in\mathcal{G}$ is a causal model.
In this case; we refer to the collection $\theta$ as the model parameters.

### As Directed Acyclic Graphs

The structure imposed by the $V_i$ allows for the relationships between the $X_i$ to be realised as a [DAG](#abbreviations), $\mathbb{G}$.
The nodes represent the random variables $X_i$, and as such we use the notation $X_i$ interchangeably when referring to the RVs or nodes of $\mathbb{G}$.
An edge directed into $X_i$ from $X_k$ (where $(k < i)$) encodes that the distribution of $X_i$ depends on $X_k\$.

`causalprog` is designed such that all causal models can be specified by providing a suitable description of them, in terms of a DAG.

### Prediction and Training

Let $\mathcal{G}$ be a collection of causal models with model parameters $\theta$.

The vector of random variables $X_j$ for which $V_j = \emptyset$ are effectively the "inputs" to the models $G(\theta)\in\mathcal{G}$.
If we write

$$\mathcal{D}_{eval} := \otimes_{V_j = \emptyset}\mathrm{dom}(X_j), $$

then given any $x\in\mathcal{D}_{eval}$, $G(\theta)$ can pass the values in $x$ through each of the $f_{X_i}$ in sequence to obtain an estimate for any of the other $X_i$.

When one is conducting model training; we typically have a collection of observations of a subset $X_{i_k}$ of our RVs, and the corresponding "input" data $x\in\mathcal{D}_{eval}$.
This collection of input points and observed outcomes is the training dataset, denoted $\mathcal{D}_{train}$, and is an element of

$$
\mathcal{D} := \mathcal{P}(
  \mathcal{D}_{eval} \times
  \left(\otimes_{k}\mathrm{dom}(X_{i_k})\right)
).
$$

The problem of training a model is then finding the particular value $\theta^{\star}$ for $\theta$ such that the model $G(\theta^{star})$ is the model that best fits the training dataset.
The "best fit" (and consequentially $\theta^{\star}$) is typically determined by defining some kind of loss function

$$
\mathcal{B}:\mathcal{D}\times\mathcal{G}\rightarrow[0, \infty)
$$

and minimising

$$ B(theta) := \mathcal{B}(\mathcal{D}_{train}; \theta) $$

over $\theta$.
The arg-minimum of this problem being the value $\theta^{\star}$, interpreted as the parameter set that best describes whatever real-world process we have observed using $\mathcal{D}_{train}$ and believe is modelled by an element of $\mathcal{G}$.

### Causal Problems

In summary, one can think of a collection of (causal) models $\mathcal{G}$ as our model of how some real-world process works, up to some descriptors $\theta$.
We can think of $\mathcal{D}_{train}$ as some observations we have of this real-world process.
Typically the problem then is to determine $\theta^{\star}$ in the manner described above, extract the particular model $G(\theta^{\star})$, and then use $G(\theta^{\star})$ to make predictions about outcomes of unseen data points $x\in\mathcal{D}_{eval}$.

However; what if we were not necessarily concerned with making predictions, but rather quantifying how extreme the differences in the predictions could be if there was some margin for error in our observed data.
Or alternatively, we might be interested in quantifying possible extremes if "reality" was allowed to be a little bit different from what $\theta^_{\star}$ allows.

This is where we encounter the notation of a causal problem.
For a given $\theta$ and fixed training set $\mathcal{D}_{train}$, the value of $B(\theta) - B(\theta^{\star})$ is essentially a quantification of "a deviation from reality".
Now suppose we have a function

$$ d:\mathcal{D}_{eval}\times\mathcal{G}\rightarrow\mathbb{R}, $$

where $d(x; \theta) := d(x; G(\theta)) = G(\theta)(x)$ is the prediction that the model $G(\theta)$ makes of some quantity of interest to us, given input data $x$.
If we take some _tolerance_ $\epsilon > 0$ and a point $x\in\mathcal{D}_{eval}$, then we can pose the _causal problem_

$$
\max_{\theta} / \min_{\theta} d(x; \theta),
\quad \text{subject to } \quad
B(\theta) \leq B(\theta^{\star}) + \epsilon.
$$

The extreme values of $d$ are referred to as the query bounds of the causal estimand, and can be interpreted as "the best and worst that could happen to $d$ if we deviate from reality by an amount $\epsilon$".

As such, the objective in a causal problem not to find the set of parameter values $\theta$ that best describes reality, but rather the extreme values of some pertinent quantity $d$ if we suspect some error / deviation in our attempts to describe reality with our model $G(\theta^{\star})$.

## Notation and Abbreviations

### Abbreviations

- DAG: [Directed Acyclic Graph](https://en.wikipedia.org/wiki/Directed_acyclic_graph)
- RV(s): [Random Variable(s)](https://en.wikipedia.org/wiki/Random_variable)
- WLOG: [Without Loss Of Generality](https://en.wikipedia.org/wiki/Without_loss_of_generality)

### Notation

- $\mathcal{P}(\Omega)$ denotes the power set of the set $\Omega$.
