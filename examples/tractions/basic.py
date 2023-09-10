import json

from ..resources.rng import RNG

from pytraction.base import Traction, In, Out, Res


class T1(Traction):
    i_in1: In[int]
    o_out1: Out[int]

    def _run(self, on_update) -> None:
        self.o_out1.data = self.i_in1.data


class T2(Traction):
    i_in1: In[int]
    o_out1: Out[int]
    r_rng_gen: Res[RNG]

    def _run(self, on_update) -> None:
        self.o_out1.data = \
            self.i_in1.data * self.r_rng_gen.r.generate()


r = RNG()
t1 = T1(uid='T1-example', i_in1=In[int](data=10))
t2 = T2(uid='T2-example', i_in1=t1.o_out1, r_rng_gen=Res[RNG](r=r))
t2.run()

