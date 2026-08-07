# Contributing and Style Guide

## Getting Started: Developers

### Reporting bugs & suggesting enhancements

Bugs can be reported and enhancements can be suggested using the [issue tracker](https://github.com/UCL/causalprog/issues) on Github.
Further discussion about the bug or enhancement can take place in the Github issue.

### Contributing code

If you want to directly submit code to causalprog, you can do this by forking the causalprog repository, then submitting a pull request.
See the [developer installation instructions](#developer-installation) if you need help forking and installing the package in editable mode.

If you want to contribute, but are unsure where to start, have a look at the [issues labelled "good first issue"](https://github.com/UCL/causalprog/issues?q=is%3Aissue%20state%3Aopen%20label%3A%22good%20first%20issue%22).

On opening a pull request, various linting checks and automated tests will run.
You can click on these in the pull request to see where (if anywhere) there are issues that need fixing in your code.
You can find more information on the [testing suite](#testing-suite) and [documentation build](#building-the-documentation) below, should you find that these checks are failing.
You may also want to check our [style guide for docstrings](#docstring-style).

A member of the development team will review your pull request when you mark it as ready, providing either feedback or approval.
Once all discussion topics are resolved, and the automated tests pass, your pull request will be merged!

### Developer Installation

We recommend a slightly different installation method to that of a user if you plan to contribute to `causalprog`.

1. If you are not a member of the core development team: to ensure that you have write access to (a copy of) the `causalprog` repository, please create your [own personal fork](https://docs.github.com/en/pull-requests/how-tos/work-with-forks/fork-a-repo) of [the `causalprog` repository](github.com/UCL/causalprog).
   Core developer will have write access to the `causalprog` repository so can create branches directly inside it, but are welcome to use forks too if they so wish.
2. Use `git clone` to obtain a local copy of your fork.
3. Install `causalprog` in editable mode, along with it's optional dependencies, via

```sh
pip install -e .[dev,docs,test]
```

Don't forget to activate the Python environment you want to install `causalprog` into before running `pip install`.

### AI-assisted development

If any LLM or other AI tool is used, this must be declared in the text of the pull request.
Any AI-generated code must be thoroughly checked by the person opening the pull request before the PR is opened.

Please do not use a LLM or other tool to target issues labelled "good first issue" - these are intended to be good onboarding issues for human developers.

## Docstring Style

### Functions and Methods

`causalprog` uses [Google-style docstrings](https://mkdocstrings.github.io/python/usage/docstrings/google/), which should be formatted as

```python
def my_function(arg1, arg2):
  """
  Summary line.

  Further information in prose / paragraph format, mathematical notation is also supported here.
  If some of the function arguments require detailed explanation, this explanation should be placed here.

  Args:
    arg1: Description of the first argument
    arg2: Description of the second argument.

  Returns:
    Description of the object(s) that are returned by the method.

  Raises:
    ExceptionType: Conditions under which this is raised.
    ExceptionType: Conditions under which this is raised.

  """
```

`mkdocs` also supports the `Tip:` and `Note:` syntax within docstrings too, which should appear within the further information section of the docstring.

If a function's purpose, return type, and inputs are clear from it's definition and name, then the docstring may consist of a single summary line instead:

```python
def sum_items(item1, item2):
  """Return the sum of two items."""
  return item1 + item2
```

### Classes and Modules

Classes and modules should also obey Google-style docstring conventions where possible, but there is no need to provide an explicit listing of the methods (and / or attributes) that such objects provide in the docstrings themselves.
However, docstrings for classes and modules should still provide an adequate level of detail about what the module does / class represents, and the components that a user will typically be interacting with.

### Docstrings in the Tests and Examples

Outside the package source code, the docstring format is much more loose, though developers should try to stick to the Google-style when possible.

In the test suite; docstrings are typically used to describe the steps in longer, more involved tests, as well as the actual comparisons or `assert`ions that are made to ensure object being tested is functioning correctly.

In the examples; docstrings are typically provided in the summary format, relying on the surrounding prose to provide context for the reader.

## Testing Suite

`causalprog`'s test suite is written using [`pytest`](https://docs.pytest.org/en/stable/).
The package can be installed with its developer dependencies, including `pytest`, by specifying the `[dev]` optional dependency when installing the package.

### Running the tests

To run the test suite manually, run the following command inside your developer environment from the root of the repository:

```sh
(causalprog-environment) $ pytest tests/
```

Alternatively, tests can be run across all compatible Python versions in isolated environments using [`tox`](https://tox.wiki/en/latest/).
Running

```sh
(causalprog-environment) $ tox
```

in the repository root will do so.

### Organisation of the test suite

The test suite contains a `fixtures` subdirectory, which is loaded as a `pytest` plugin when the tests are run.
All `pytest.fixture` objects defined inside the `fixtures` subdirectory (and subdirectories therein) are discovered by `pytest`, and available for use by individual tests.

- Fixtures that are shared across multiple test files should be refactored into this folder, being placed into an appropriate file.
  If possible, include a docstring describing what the fixture does (if it is a method or function) or the instance it defines (if it defines an instance of a class, for example a fixed `Graph` used in multiple tests).
- Fixtures that are only used by tests within a single file, should be defined in that file rather than the `fixtures` directory.

We favour granularity for the files containing the tests themselves.
Unit tests are stored starting at the same level as the `fixtures` directory.
Our general guidelines for organising unit tests are:

- Use subdirectories to group files containing tests by module and class.
- For each method or function; write all its unit tests inside a single file.
- Keep to the limit of one method / function being tested per file, except in cases where it is sensible to include closely related methods / functions.

Any integration tests should be placed into the `test_integration` subfolder.
Again, this directory should contain a single file per integration test.

### Useful fixtures

Some useful fixtures that are included in the `fixtures` directory;

- `ssed` and `rng_key` (`fixtures/general.py`) - sets [the PRNG Key](https://docs.jax.dev/en/latest/_autosummary/jax.random.PRNGKey.html) that should be used across all tests, to ensure repeatability.
- `raises_context` (`fixtures/general.py`) - can be used to return a `pytest.raises` context that checks for a specific exception, including matching the error message.

## Building the Documentation

`causalprog` uses [MkDocs](https://www.mkdocs.org/) to build HTML documentation from the files in the `docs` directory.
Each markdown file will be rendered to a HTML document when the build runs, and any image files or other assets that are required for the build should be placed (in an appropriate location under the) `docs` directory accordingly.

The documentation can be built locally by running

```sh
(causalprog-environment) $ tox -e docs
```

from the root of the repository.
The built documentation will be written to `site`.

Alternatively to build and preview the documentation locally, in a Python
environment with the optional `docs` dependencies installed, run

```sh
(causalprog-environment) $ mkdocs serve
```
