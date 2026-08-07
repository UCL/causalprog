# causalprog

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests status][tests-badge]][tests-link]
[![Linting status][linting-badge]][linting-link]
[![Documentation status][documentation-badge]][documentation-link]
[![License][license-badge]](http://github-pages.ucl.ac.uk/causalprog/LICENSE)

<!-- prettier-ignore-start -->
[tests-badge]:              https://github.com/UCL/causalprog/actions/workflows/tests.yml/badge.svg
[tests-link]:               https://github.com/UCL/causalprog/actions/workflows/tests.yml
[linting-badge]:            https://github.com/UCL/causalprog/actions/workflows/linting.yml/badge.svg
[linting-link]:             https://github.com/UCL/causalprog/actions/workflows/linting.yml
[documentation-badge]:      https://github.com/UCL/causalprog/actions/workflows/docs.yml/badge.svg
[documentation-link]:       https://github.com/UCL/causalprog/actions/workflows/docs.yml
[license-badge]:            https://img.shields.io/badge/License-MIT-yellow.svg
<!-- prettier-ignore-end -->

A Python package for causal modelling and inference with stochastic causal programming

This project is developed in collaboration with the
[Centre for Advanced Research Computing](https://ucl.ac.uk/arc), University
College London.

## Causal Problems and `causalprog`

### TLDR

Given a [causal model](docs/theory/index.md#causal-models) $\mathbb{G}(\theta)$ with model parameters $\theta$, `causalprog` is designed to help with the setup and solution of

$$ \max_{\theta} / \min_{\theta} d(\theta), \quad \text{subject to } \quad B(\theta) \leq B(\theta^{\star}) + \epsilon, $$

where;

- $d$ is a causal response function on the causal model,
- $B$ is a loss function,
- $\theta^{\star}$ is the parameter set that minimises the loss function $B$, $\theta^{\star} = \mathrm{argmin}_{\theta}B$,
- $\epsilon$ is a user-provided tolerance.

### The Longer Version

The typical situation that one is faced with when creating a probabilistic model of a real-world process is:

- Propose a probabilistic model of the process, parameterised by some quantities $\theta$.
  This effectively defines an "admissible class" of models $\mathcal{S}$, where each $\mathbb{G}(\theta)\in\mathcal{S}$ is one such possibility that can be used to predict the outcome of the real-world process we are modelling.
  Note that $\theta$ effectively parametrises $\mathcal{S}$, but we use the notation $\mathbb{G}(\theta)$ to distinguish between the models that live in $\mathcal{S}$ and the parameters that, well, parametrise them.
- Gather observable data $\mathcal{D}_{train}$ from the real-world process, that can be fitted to the random variables of our probabilistic models that live in $\mathcal{S}$.
- Use a suitable training technique, often boiling down to the optimisation of some loss function $B(\theta)$ defined on $\mathcal{S}$, to determine the set of parameters $\theta^{\star}$ that best describes the real-world process.
- The element $\mathbb{G}(\theta^{\star})\in\mathcal{S}$ is then interpreted as our understanding (or "best approximation") of the real-world process, and is then make predictions about the process at unseen data points $x_i$ drawn from some set of points $\mathcal{D}_{eval}$.

Put rather simply, the process described above can be thought of as:

> Given my understanding of how the world works ($\mathcal{S}$), and what I have observed about the world ($\mathcal{D}_{train}$), determine the best description of the world that my understanding can give ($\mathbb{G}(\theta^{\star})$).
> The "best description" is understood in terms of how the loss function $B$ is defined.

Keeping the above context; suppose that there is some quantity of interest to us, $d$, that each model in $\mathcal{S}$ can predict.
$d$ is referred to as the causal response function; with $d(x; \theta)$ being the prediction that $\mathbb{G}(\theta)$ makes for the value of the quantity $d$ given input data $x\in\mathcal{D}_{eval}$.
Furthermore, we can interpret the quantity $B(\theta) - B(\theta^{\star})$ (for any admissible $\theta$) as some kind of quantification of the "deviation from reality" of the model $\mathbb{G}(\theta)$ from $\mathbb{G}(\theta^{\star})$.
Now let $\epsilon > 0$ be some quantification we have for this "deviation from reality", or alternatively a "tolerance" we have in $B$'s ability to determine the optimal model parameters.
We can then ask the following question:

> What are the extreme values of our quantity of interest ($d$) if we are only partially confident ($\epsilon$) in $B$'s ability to determine the best model of the real world?
> Or alternatively, what are the extreme values of our quantity of interest ($d$) if we allow the world to deviate slightly ($\epsilon$) from reality?

It is this latter question that `causalprog` is concerned with.
Mathematically, this means we are looking to solve

$$ \max_{\theta} / \min_{\theta} d(x; \theta), \quad \text{subject to } \quad B(\theta) \leq B(\theta^{\star}) + \epsilon, $$

given $x\in\mathcal{D}_{eval}$ and $\epsilon > 0$.

The extreme (max and min values) of $d$ are referred to as the "query bounds on the causal response".
Problems of this type are what `causalprog` refers to as "causal problems".

`causalprog` provides utility for setting up causal problems [using DAGs](docs/theory/glossary.md#abbreviations), which can then be solved via your favourite stochastic optimiser and minimisation algorithm (though the package also provides a few solvers itself to help).

## Getting Started

See [our documentation for information on how to get started](http://github-pages.ucl.ac.uk/causalprog/getting-started) with `causalprog`.

## About

### Project team

- Ricardo Silva ([rbas-ucl](https://github.com/rbas-ucl))
- Jialin Yu ([jialin-yu](https://github.com/jialin-yu))
- Will Graham ([willGraham01](https://github.com/willGraham01))
- Matthew Scroggs ([mscroggs](https://github.com/mscroggs))
- Sam Molyneux ([sjmolyneux](https://github.com/samjmolyneux))

### Research software engineering contact

Centre for Advanced Research Computing, University College London
([arc.collaborations@ucl.ac.uk](mailto:arc.collaborations@ucl.ac.uk))

## Acknowledgements

This work was funded by Engineering and Physical Sciences Research Council (EPSRC).
