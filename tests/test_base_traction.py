from typing import List, Dict, Union, Optional, TypeVar, Generic

import pytest

from pytraction.base import Traction, JSONIncompatibleError, TList, TDict, Out, In, Arg, Res, ANY, Base

# Jsonable test cases

T = TypeVar("T")


def test_traction_ok_args_1():
    o = Out[int]
    o2 = Out[str]
    o3 = Out[int]
    o4 = Out[ANY]
    print(dir(o))

    class TTest(Traction):
        i_in1: In[int]
        o_out1: Out[int]
        r_res1: Res[int]
        a_arg1: Arg[int]


def test_traction_wrong_args_1():
    with pytest.raises(TypeError):
        class TTest(Traction):
            i_in1: Out[int]


def test_traction_wrong_args_2():
    with pytest.raises(TypeError):
        class TTest(Traction):
            o_out1: In[int]


def test_traction_wrong_args_3():
    with pytest.raises(TypeError):
        class TTest(Traction):
            a_arg1: In[int]


def test_traction_wrong_args_4():
    with pytest.raises(TypeError):
        class TTest(Traction):
            r_res1: In[int]


def test_traction_inputs_1():
    class TTest(Base):
        #i_in1: In[int]
        pass

    t = TTest()#uid=10)


    o: Out[int] = Out[int](data=10)

    #t = TTest(uid="1", i_in1=o)
