import re

import pytest

from causalprog._abc.backend_agnostic import BackendAgnostic


class OneMethodBackend:
    def method1(self) -> None:
        return


class TwoMethodBackend(OneMethodBackend):
    def method2(self) -> None:
        return


class ThreeMethodBackend(TwoMethodBackend):
    def method3(self) -> None:
        return


class BA(BackendAgnostic):
    """
    Designed to test the abstract ``BackendAgnostic`` class.

    Instances take ``*methods`` as an argument, which has the effect of setting
    ``self.method`` to be a function that returns ``True`` for each ``method`` in
    ``*methods``.
    """

    @property
    def _frontend_provides(self) -> tuple[str, ...]:
        return (
            "method1",
            "method2",
        )


@pytest.mark.parametrize(
    ("backend", "expected_missing"),
    [
        pytest.param(
            TwoMethodBackend(),
            set(),
            id="All methods defined.",
        ),
        pytest.param(
            ThreeMethodBackend(),
            set(),
            id="Additional methods defined.",
        ),
        pytest.param(
            OneMethodBackend(),
            {"method2"},
            id="Missing required method.",
        ),
    ],
)
def test_method_discovery(backend: object, expected_missing: set[str]) -> None:
    obj = BA(backend=backend)
    assert obj.get_backend() is backend

    assert obj._missing_methods == expected_missing  # noqa: SLF001
    if len(expected_missing) != 0:
        with pytest.raises(
            AttributeError,
            match=re.escape("Missing frontend methods: " + ", ".join(expected_missing)),
        ):
            obj.validate()
