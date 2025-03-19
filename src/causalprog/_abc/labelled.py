from abc import ABC


class Labelled(ABC):
    """
    ABC for objects that carry a label. This class can be used as a MixIn.

    Objects must be passed an explicit ``label`` parameter on instantiation,
    which provides a name for the object. This value is stored in the
    private ``_label`` attribute, and is only intended to be accessed via the
    ``label`` property of the class.
    """

    __slots__ = ("_label",)
    _label: str | None

    @property
    def label(self) -> str:
        """Label of this object."""
        if self._label is None:
            msg = "Node has no label."
            raise ValueError(msg)
        return self._label

    def __init__(self, *, label: str | None) -> None:
        self._label = label
