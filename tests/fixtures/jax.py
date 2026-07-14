import jax
import pytest


@pytest.fixture
def jax_enable_x64():
    """Enable x64 precision for a single test.

    Note that x64 precision can make a big difference to results, since numerical
    derivatives are sensitive to "non-ops". EG Calculations that analytically make no
    difference to the end result, but numerically _can_ affect the output value due to
    rounding etc.
    """
    setting = "jax_enable_x64"
    prev_setting_value = jax.config.read(setting)
    jax.config.update(setting, val=True)

    yield

    jax.config.update(setting, prev_setting_value)
