# Contributing and Style Guide

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

For information on setting up and running the tests, please see [the testing suite page](./tests.md).
