from typing import List, Dict, Union, Optional, TypeVar, Generic

import pytest

from pytraction.base import (
    Traction,
    JSONIncompatibleError,
    TList,
    TDict,
    Out,
    In,
    Arg,
    Res,
    ANY,
    Base,
    NoData,
    TypeNode,
)

# Jsonable test cases

T = TypeVar("T")

from dataclasses import field


def test_traction_ok_args_1():
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
    class TTest(Traction):
        i_in1: In[int] = In[int]()

    o: Out[int] = Out[int](data=10)
    t = TTest(uid="1", i_in1=o)


def test_traction_inputs_read():
    class TTest(Traction):
        i_in1: In[int] = field(default=In[int]())

    o: Out[int] = Out[int](data=10)
    t = TTest(uid="1", i_in1=o)
    assert id(t.i_in1.data) == id(o.data)


def test_traction_inputs_read_unset():
    class TTest(Traction):
        i_in1: In[int]

    t = TTest(uid="1")
    assert TypeNode.from_type(type(t.i_in1)) == TypeNode.from_type(NoData[int])


def test_traction_inputs_read_set():
    class TTest(Traction):
        i_in1: In[int]

    o: Out[int] = Out[int](data=10)
    t = TTest(uid="1", i_in1=o)
    assert TypeNode.from_type(type(t.i_in1)) == TypeNode.from_type(Out[int])
    assert t.i_in1.data == 10


def test_traction_to_json():
    class TTest(Traction):
        i_in1: In[int] = In[int]()

    o: Out[int] = Out[int](data=10)
    t = TTest(uid="1", i_in1=o)
    assert t.to_json() == {
        "details": {},
        "errors": {},
        "i_in1": {"data": 10},
        "skip": False,
        "skip_reason": "",
        "state": "ready",
        "stats": {"finished": "", "skipped": False, "started": ""},
        "uid": '1',
    }


def test_traction_outputs_no_init():
    class TTest(Traction):
        o_out1: Out[int]

    t = TTest(uid="1")
    assert t.o_out1 == Out[int]()


def test_traction_outputs_no_init_custom_default():
    class TTest(Traction):
        o_out1: Out[int] = Out[int](data=10)

    t = TTest(uid="1")
    assert t.o_out1 == Out[int](data=10)



def test_traction_outputs_no_init_custom_default():
    class TTest(Traction):
        o_out1: Out[int] = Out[int](data=10)

    t = TTest(uid="1")
    assert t.o_out1 == Out[int](data=10)


def test_traction_chain():
    class TTest1(Traction):
        o_out1: Out[int]

        def _run(self, on_update) -> None:  # pragma: no cover
            self.o_out1.data = 20

    class TTest2(Traction):
        i_in1: In[int]
        o_out1: Out[int]

        def _run(self, on_update) -> None:  # pragma: no cover
            self.o_out1.data = self.i_in1.data + 10

    t1 = TTest1(uid="1")
    t2 = TTest2(uid="1", i_in1=t1.o_out1)

    t1.run()
    t2.run()
    assert t2.o_out1.data == 30


def test_traction_chain_in_to_out():
    class TTest1(Traction):
        o_out1: Out[int]

        def _run(self, on_update) -> None:  # pragma: no cover
            self.o_out1.data = 20

    class TTest2(Traction):
        i_in1: In[int]
        o_out1: Out[int]

        def _run(self, on_update) -> None:  # pragma: no cover
            self.o_out1 = self.i_in1

    t1 = TTest1(uid="1")
    t2 = TTest2(uid="1", i_in1=t1.o_out1)

    t1.run()
    t2.run()
    assert t2.o_out1.data == 20
    t1.o_out1.data = 30

    assert t2.i_in1.data == 30

