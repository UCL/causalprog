# causalprog

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Tests status][tests-badge]][tests-link]
[![Linting status][linting-badge]][linting-link]
[![Documentation status][documentation-badge]][documentation-link]
[![License][license-badge]](./LICENSE.md)

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

## About

### Project team

- Ricardo Silva ([rbas2015](https://github.com/rbas2015))
- Jialin Yu ([jialin-yu](https://github.com/jialin-yu))
- Will Graham ([willGraham01](https://github.com/willGraham01))
- Matthew Scroggs ([mscroggs](https://github.com/mscroggs))
- Matt Graham ([matt-graham](https://github.com/matt-graham))

### Research software engineering contact

Centre for Advanced Research Computing, University College London
([arc.collaborations@ucl.ac.uk](mailto:arc.collaborations@ucl.ac.uk))

## Getting Started

### Prerequisites

<!-- Any tools or versions of languages needed to run code. For example specific Python or Node versions. Minimum hardware requirements also go here. -->

`causalprog` requires Python 3.11&ndash;3.13.

### Installation

<!-- How to build or install the application. -->

We recommend installing in a project specific virtual environment. To install the latest
development version of `causalprog` using `pip` in the currently active environment run

```sh
pip install git+https://github.com/UCL/causalprog.git
```

Alternatively create a local clone of the repository with

```sh
git clone https://github.com/UCL/causalprog.git
```

and then install in editable mode by running

```sh
pip install -e .
```

### Running tests

<!-- How to run tests on your local system. -->

Tests can be run across all compatible Python versions in isolated environments
using [`tox`](https://tox.wiki/en/latest/) by running

```sh
tox
```

To run tests manually in a Python environment with `pytest` installed run

```sh
pytest tests
```

again from the root of the repository.

### Building documentation

The MkDocs HTML documentation can be built locally by running

```sh
tox -e docs
```

from the root of the repository. The built documentation will be written to
`site`.

Alternatively to build and preview the documentation locally, in a Python
environment with the optional `docs` dependencies installed, run

```sh
mkdocs serve
```

## Acknowledgements

This work was funded by Engineering and Physical Sciences Research Council (EPSRC).
