import re
from collections.abc import Sequence

import pytest

from causalprog.backend.translation import Translation
from causalprog.backend.translator import Translator


class BackendObjNeedsNoTranslation:
    def frontend_method(self, denominator: float, numerator: float) -> float:
        return numerator / denominator


class BackendObjNameChangeOnly:
    def backend_method(self, denominator: float, numerator: float) -> float:
        return numerator / denominator


class BackendObjNeedsTranslation:
    def backend_method(self, num: float, denom: float) -> float:
        return num / denom


class BackendObjDropsArg:
    def backend_method(self, num: float, denom: float, constant: float) -> float:
        return num / denom + constant


class TranslatorForTesting(Translator):
    @property
    def _frontend_provides(self) -> tuple[str, ...]:
        return ("frontend_method",)

    def frontend_method(self, denominator: float, numerator: float) -> float:
        return self._call_backend_with("frontend_method", denominator, numerator)


@pytest.mark.parametrize(
    ("translations", "backend", "created_method_is_identity"),
    [
        pytest.param(
            (), BackendObjNeedsNoTranslation(), True, id="Backend needs no translation."
        ),
        pytest.param(
            (
                Translation(
                    backend_name="backend_method",
                    frontend_name="frontend_method",
                    param_map={},
                ),
            ),
            BackendObjNameChangeOnly(),
            False,
            id="Method name change, map is no longer identity.",
        ),
        pytest.param(
            (
                Translation(
                    backend_name="backend_method",
                    frontend_name="frontend_method",
                    param_map={"num": "numerator", "denom": "denominator"},
                ),
            ),
            BackendObjNeedsTranslation(),
            False,
            id="Full translation required.",
        ),
        pytest.param(
            (
                Translation(
                    backend_name="backend_method",
                    frontend_name="frontend_method",
                    param_map={"num": "numerator", "denom": "denominator"},
                    frozen_args={"constant": 0.0},
                ),
            ),
            BackendObjDropsArg(),
            False,
            id="Drop an argument.",
        ),
    ],
)
def test_creation_and_methods(
    translations: Sequence[Translation], backend, *, created_method_is_identity: bool
) -> None:
    translator = TranslatorForTesting(*translations, backend=backend)

    assert callable(translator.translations["frontend_method"])
    assert (
        translator.translations["frontend_method"] is translator.identity
    ) == created_method_is_identity

    input_denominator = 2.0
    input_numerator = 1.0
    assert translator.frontend_method(
        input_denominator, input_numerator
    ) == pytest.approx(input_numerator / input_denominator)


def test_must_provide_all_frontend_methods() -> None:
    # You cannot get away without defining one of the frontend methods
    with pytest.raises(
        AttributeError,
        match=re.escape(
            "No translation provided for frontend_method, which the backend is lacking."
        ),
    ):
        TranslatorForTesting(backend=BackendObjNameChangeOnly())
