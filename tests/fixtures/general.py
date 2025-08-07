import jax.random as jrn
import pytest


@pytest.fixture
def seed() -> int:
    return 0


@pytest.fixture
def rng_key(seed: int):
    return jrn.key(seed)
