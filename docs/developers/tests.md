# Testing Suite

`causalprog`'s test suite is written using [`pytest`](https://docs.pytest.org/en/stable/).
The package can be installed with its developer dependencies, including `pytest`, by specifying the `[dev]` optional dependency when installing the package.

## Running the tests

To run the test suite, you will need to clone the `causalprog` repository and then install `causalprog` into your developer environment with the `[dev]` optional dependencies.
We recommend specifying an editable installation if you intend to make contributions to the package.

```sh
(causalprog-environment) $ git clone git@github.com:UCL/causalprog.git
(causalprog-environment) $ cd causalprog
(causalprog-environment) $ pip install -e .[dev]
```

You can then run the tests manually inside your developer environment from the root of the repository,

```sh
(causalprog-environment) $ pytest tests/
```

Alternatively, tests can be run across all compatible Python versions in isolated environments using [`tox`](https://tox.wiki/en/latest/).
Running

```sh
(causalprog-environment) $ tox
```

in the repository root will do so.

## Organisation of the test suite

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

## Useful fixtures

Some useful fixtures that are included in the `fixtures` directory;

- `ssed` and `rng_key` (`fixtures/general.py`) - sets [the PRNG Key](https://docs.jax.dev/en/latest/_autosummary/jax.random.PRNGKey.html) that should be used across all tests, to ensure repeatability.
- `raises_context` (`fixtures/general.py`) - can be used to return a `pytest.raises` context that checks for a specific exception, including matching the error message.
