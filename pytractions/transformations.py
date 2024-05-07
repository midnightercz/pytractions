
from typing import TypeVar, Generic
from .base import Base, In, Out, Traction, OnUpdateCallable, TList, Arg

T = TypeVar("T")
X = TypeVar("X")


class Flatten(Traction, Generic[T]):
    i_complex: In[TList[In[TList[In[T]]]]]
    o_flat: Out[TList[Out[T]]]

    def _run(self, on_update: OnUpdateCallable):
        i = 0
        for nested in self.i_complex.data:
            for item in nested.data:
                self.o_flat.data.append(Out[self._params[0]](data=item.data))
                i += 1


class FilterDuplicates(Traction, Generic[T]):
    i_list: In[TList[In[T]]]
    o_list: Out[TList[Out[T]]]

    def _run(self, on_update: OnUpdateCallable):
        i = 0
        for item in self.i_list.data:
            if Out[self._params[0]](data=item.data) not in self.o_list.data:
                self.o_list.data.append(Out[self._params[0]](data=item.data))



class CartesianProduct(Traction, Generic[T, X]):
    i_outer: In[TList[In[T]]]
    i_inner: In[TList[In[X]]]
    
    o_outer: Out[TList[Out[TList[Out[T]]]]]
    o_inner: Out[TList[Out[TList[Out[X]]]]]

    def _run(self, on_update: OnUpdateCallable):
        for outer in self.i_outer.data:
            for inner in self.i_inner.data:
                self.o_outer.data.append(outer)
                self.o_inner.data.append(inner)


class Repeat(Traction, Generic[T, X]):
    i_element: In[T]
    i_times: In[TList[In[X]]]
    
    o_repeated: Out[TList[Out[T]]]

    def _run(self, on_update: OnUpdateCallable):
        for _ in self.i_times.data:
            self.o_repeated.data.append(self.i_element.data)


class Repeat2D(Traction, Generic[T, X]):
    i_list: In[TList[In[T]]]
    i_complex_list: In[TList[In[TList[In[X]]]]]
    
    o_repeated: Out[TList[Out[TList[Out[T]]]]]

    def _run(self, on_update: OnUpdateCallable):
        for elem, nested in zip(self.i_list.data, self.i_complex_list.data):
            self.o_repeated.data.append(Out[TList[Out[T]]](data=TList[Out[T]]()))
            for nested_elem in nested.data:
                self.o_repeated.data[-1].data.append(Out[T](data=elem.data))


class Explode(Traction):
    i_model: In[Base]

    def _run(self, on_update: OnUpdateCallable):
        for f in self._fields:
            if f.startswith("o_"):
                getattr(self, f).data = getattr(self.i_model.data, f.replace("o_","", 1))

class Extractor(Traction, Generic[T, X]):
    a_field: Arg[str]
    i_model: In[T]
    o_model: Out[X]

    def _run(self, on_update: OnUpdateCallable):
        self.o_model.data = getattr(self.i_model.data, self.a_field.a)


