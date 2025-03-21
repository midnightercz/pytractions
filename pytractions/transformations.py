from typing import TypeVar, Generic
from .base import TList
from .traction import Traction

T = TypeVar("T")
X = TypeVar("X")


class Flatten(Traction, Generic[T]):
    """Flatten list of list of T to list of T."""

    i_complex: TList[TList[T]]
    o_flat: TList[T]

    def _run(self):
        for nested in self.i_complex:
            for item in nested:
                self.o_flat.append(item)


class FilterDuplicates(Traction, Generic[T]):
    """Remove duplicates from input list."""

    i_list: TList[T]
    o_list: TList[T]

    def _run(self):
        for item in self.i_list:
            if item not in self.o_list:
                self.o_list.append(item)


class Extractor(Traction, Generic[T, X]):
    """Extract field from input model as separated output."""

    a_field: str
    i_model: T
    o_model: X

    def _run(self):
        self.o_model = getattr(self.i_model, self.a_field)


class ListMultiplier(Traction, Generic[T, X]):
    """Multiply list by scalar."""

    i_list: TList[T]
    i_scalar: X
    o_list: TList[X]

    d_: str = """Takes lengh of input list and creates output list of the same length filled
with scalar value."""
    d_i_list: str = "Input list."
    d_i_scalar: str = "Scalar value."
    d_o_list: str = "Output list."

    def _run(self):
        for _ in range(len(self.i_list)):
            self.o_list.append(
                self._raw_i_scalar.content_from_json(self._raw_i_scalar.content_to_json()).data
            )
