# Getting Started

## Prerequisites

`causalprog` requires Python 3.11 - 3.14.
You may also want to setup a Python environment into which to install the package.

## User Installation

We recommend installing in a project specific virtual environment.
To install the latest development version of `causalprog` using `pip` in the currently active environment, run

```sh
pip install git+https://github.com/UCL/causalprog.git
```

Alternatively create a local clone of the repository with

```sh
git clone https://github.com/UCL/causalprog.git
```

and then install `causalprog` by running

```sh
pip install .
```

in the directory that you just cloned.

### Developer Installation

If you would like to contribute to `causalprog`, please see [our contributing page](./developers/contributing.md) for developer installation instructions, and developer conventions.

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
