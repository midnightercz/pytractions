from typing import List, Dict, Union, Optional, TypeVar, Generic

import pytest

from pytraction.base import Traction, JSONIncompatibleError, TList, TDict, Out, In, Arg, Res

# Jsonable test cases

T = TypeVar("T")


def test_traction_ok_args_1():
    with pytest.raises(TypeError):
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
            r_res1: Res[int]
