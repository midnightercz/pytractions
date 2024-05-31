from typing import TypeVar, Generic
from .base import In, Out, Traction, OnUpdateCallable, TList, Arg

T = TypeVar("T")
X = TypeVar("X")


class Flatten(Traction, Generic[T]):
    """Flatten list of list of T to list of T."""

    i_complex: In[TList[TList[T]]]
    o_flat: Out[TList[T]]

    def _run(self, on_update: OnUpdateCallable):
        for nested in self.i_complex.data:
            for item in nested:
                self.o_flat.data.append(item)


class FilterDuplicates(Traction, Generic[T]):
    """Remove duplicates from input list."""

    i_list: In[TList[T]]
    o_list: Out[TList[T]]

    def _run(self, on_update: OnUpdateCallable):
        for item in self.i_list.data:
            if item not in self.o_list.data:
                self.o_list.data.append(item)


class Extractor(Traction, Generic[T, X]):
    """Extract field from input model as separated output."""

    a_field: Arg[str]
    i_model: In[T]
    o_model: Out[X]

    def _run(self, on_update: OnUpdateCallable):
        self.o_model.data = getattr(self.i_model.data, self.a_field.a)


class ListMultiplier(Traction, Generic[T, X]):
    """Multiply list by scalar."""

    i_list: In[TList[T]]
    i_scalar: In[X]
    o_list: Out[TList[X]]

    d_: str = """Takes lengh of input list and creates output list of the same length filled
with scalar value."""
    d_i_list: str = "Input list."
    d_i_scalar: str = "Scalar value."
    d_o_list: str = "Output list."

    def _run(self, on_update: OnUpdateCallable):
        for _ in range(len(self.i_list.data)):
            self.o_list.data.append(
                self.i_scalar.content_from_json(self.i_scalar.content_to_json()).data
            )
