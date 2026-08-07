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

TL;DR, `causalprog` solves

$$ \max_{\Theta} / \min_{\Theta} \sigma(\Theta), \quad \text{subject to } \quad \vert\vert \phi_{\mathrm{data}, k} - \phi_k(\Theta) \vert\vert\leq \epsilon, \quad \forall k, $$

given

- a [model parameters](docs/theory/glossary.md#model-parameter) for a [causal model](docs/theory/glossary.md#causal-model) $\Theta$,
- a [causal estimand](docs/theory/glossary.md#causal-estimand) $\sigma$,

and [constraint functions](docs/theory/glossary.md#constraint-function) $\phi = (\phi_k)_k$, where;

- $\phi_\mathrm{data}$ is empirically observed values of $\phi$,
- $\phi = (\phi_k)_k$ is the analytical estimate of $\phi$ from the causal model, given $\Theta$,
- $\vert\vert\cdot\vert\vert$ is a non-negative valued distance function (such as a suitable norm),
- $\epsilon = (\epsilon_k)_k$ is the [tolerance in the observed data](docs/theory/glossary.md#tolerance-of-a-constraint).

The solution to a causal problem is;

- the maximum / minimum value of the causal estimand $\sigma$,
- and the corresponding set of model parameter values $\Theta$ that allows $\sigma$ to attain this extrema.

The causal estimand $\sigma$ is typically a quantity of interest, derived from our model, that we are unable to empirically observe (or is unfeasible for us to observe).
For the time being, `causalprog` focuses on casual estimands that are predominantly integrals of some type.
In particular, the focus is on causal estimands that are the expectations (or possibly higher moments) of one of the random variables $X_k$ given some other conditions.
As such, computing the value of a causal estimand will be done largely through Monte Carlo sampling to approximate these integrands.
Since no assumption is made on the dimensionality of our random variables (and thus domains of the integrals), some of these integrals may require a large number of samples before giving a suitable approximation to the true value.

The constraint functions $\phi_k$ represent quantities derived from our model that we can (and have) observed, and are used along with the tolerance values $\epsilon$ to limit the class of admissible models to those which it was feasible for us to be observing empirically.
Solving the resulting causal problem thus provides us with best / worst case estimates for $\sigma$, given what we know to be true about the real world.

`causalprog` provides utility for setting up causal problems [using DAGs](docs/theory/glossary.md#abbreviations), which can then be solved via your favourite stochastic optimiser and minimisation algorithm.
For example, one could seek the saddle points of the augmented lagrangian

$$ \mathcal{L}(\Theta, \lambda) := \sigma(\Theta) - \lambda \left( \vert\vert \phi_\mathrm{data} - \phi(\Theta) \vert\vert - \epsilon\right), $$

The package also provides some basic wrappers for these solvers, for the most common techniques / algorithms that are used to solve the optimisation problems that are encountered.

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
