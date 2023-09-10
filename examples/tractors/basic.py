import json

from pytraction.base import In, Res, Out
from pytraction.tractor import Tractor

from ..tractions.basic import T1, T2
from ..resources.rng import RNG


class TT(Tractor):
    i_in1: In[int] = In[int](data=10)
    r_rng: Res[RNG] = Res[RNG](r=RNG())

    t_t1: T1 = T1(uid='T1-example', i_in1=i_in1)
    t_t2: T2 = T2(uid='T2-example',
            i_in1=t_t1.o_out1,
            r_rng_gen=r_rng)

    o_out1: Out[int] = t_t2.o_out1

tt = TT(uid='tt-example', i_in1=In[int](data=10), r_rng=Res[RNG](r=RNG()))
tt.run()
print(json.dumps(tt.to_json()))
