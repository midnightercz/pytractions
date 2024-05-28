from typing import TypeVar, Generic
from .base import In, Out, Traction, OnUpdateCallable, TList, Arg, Base

T = TypeVar("T")
X = TypeVar("X")


class Flatten(Traction, Generic[T]):
    """Flatten list of list of T to list of T."""

    i_complex: In[TList[In[TList[In[T]]]]]
    o_flat: Out[TList[Out[T]]]

    def _run(self, on_update: OnUpdateCallable):
        i = 0
        for nested in self.i_complex.data:
            for item in nested.data:
                self.o_flat.data.append(Out[self._params[0]](data=item.data))
                i += 1


class FilterDuplicates(Traction, Generic[T]):
    """Remove duplicates from input list."""

    i_list: In[TList[In[T]]]
    o_list: Out[TList[Out[T]]]

    def _run(self, on_update: OnUpdateCallable):
        for item in self.i_list.data:
            if Out[self._params[0]](data=item.data) not in self.o_list.data:
                self.o_list.data.append(Out[self._params[0]](data=item.data))


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
            if isinstance(self.i_scalar.data, Base):
                self.o_list.data.append(
                    self.i_scalar.data.content_from_json(self.i_scalar.data.content_to_json())
                )
            else:
                self.o_list.data.append(self.i_scalar.data)
