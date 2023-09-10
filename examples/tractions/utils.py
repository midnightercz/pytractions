from typing import Generic, TypeVar

from pytraction.base import Traction, TList, In, Out

T = TypeVar('T')

class ListMultiplier(Traction):
    i_input: In[T]
    i_in_list: In[TList[T]]
    o_output: Out[TList[T]]

    def run(self):
        for x in range(0, len(self.i_in_list.data)):
            self.o_output.data.append(self.i_input.data)
