from typing import List, Dict, Union, Optional, TypeVar, Generic, Type

import pytest

from pytractions.base import (
    Traction,
    STMD,
    TList,
    Out,
    In,
    Arg,
    Res,
    Base,
    NoData,
    TypeNode,
    STMDExecutorType,
    STMD,
    STMDSingleIn,
)


from pytractions.tractor import Tractor

class NOPResource(Base):
    pass


class NOPResource2(Base):
    pass


class EmptyTraction(Traction):
    i_input: In[int]
    o_output: Out[int]
    a_arg: Arg[int]
    r_res: Res[NOPResource]

    def _run(self, on_update) -> None:  # pragma: no cover
        if not self.i_input.data:
            print(self.uid, "DATA", self.i_input.data)
        if not self.i_coeficient.data:
            print(self.uid, "COEF", self.i_coeficient.data)
        self.o_output.data = self.i_input.data * 2 / self.i_coeficient.data


class Double(Traction):
    i_input: In[int]
    i_coeficient: In[float]
    o_output: Out[int]

    def _run(self, on_update) -> None:  # pragma: no cover
        if not self.i_input.data:
            print(self.uid, "DATA", self.i_input.data)
        if not self.i_coeficient.data:
            print(self.uid, "COEF", self.i_coeficient.data)
        self.o_output.data = self.i_input.data * 2 / self.i_coeficient.data


class STMDDouble(STMD):
    _traction: Type[Traction] = Double

    i_input: In[TList[In[int]]]
    i_coeficient: STMDSingleIn[float]

    o_output: Out[TList[Out[int]]]


class Half(Traction):
    i_input: In[int]
    i_coeficient: In[float]
    o_output: Out[int]

    def _run(self, on_update) -> None:  # pragma: no cover
        if not self.i_input.data:
            print(self.uid, "DATA", self.i_input.data)
        if not self.i_coeficient.data:
            print(self.uid, "COEF", self.i_coeficient.data)
        self.o_output.data = self.i_input.data / 2  * self.i_coeficient.data


class STMDHalf(STMD):
    _traction: Type[Traction] = Half

    i_input: In[TList[In[int]]]
    i_coeficient: STMDSingleIn[float]
    o_output: Out[TList[Out[int]]]


class Calculator(Tractor):
    i_inputs: In[TList[In[int]]] = In[TList[In[int]]]()
    i_coeficient: STMDSingleIn[float] = STMDSingleIn[float]()
    a_pool_size: Arg[int] = Arg[int](a=30)

    t_double: STMDDouble = STMDDouble(
        uid='double',
        i_input=i_inputs,
        i_coeficient=i_coeficient,
        a_executor_type=Arg[STMDExecutorType](a=STMDExecutorType.THREAD),
        a_pool_size=a_pool_size)

    t_half: STMDHalf = STMDHalf(uid='half',
        i_input=t_double.o_output,
        i_coeficient=i_coeficient,
        a_executor_type=Arg[STMDExecutorType](a=STMDExecutorType.THREAD),
        a_pool_size=a_pool_size)

    o_output: Out[TList[Out[int]]] = t_half.o_output


def test_stmd_attr_validation():
    # wrong input
    with pytest.raises(TypeError):
        class TestSTMD(STMD):
            _traction: Type[Traction] = EmptyTraction
            i_input: In[int]
            o_output: Out[TList[Out[int]]]
            r_res: Res[NOPResource]
            a_arg: Arg[int]

    # wrong output
    with pytest.raises(TypeError):
        class TestSTMD(STMD):
            _traction: Type[Traction] = EmptyTraction
            i_input: In[TList[In[int]]]
            o_output: TList[Out[int]]
            r_res: Res[NOPResource]
            a_arg: Arg[int]

    # wrong resource
    with pytest.raises(TypeError):
        class TestSTMD(STMD):
            _traction: Type[Traction] = EmptyTraction
            i_input: In[TList[In[int]]]
            o_output: Out[TList[Out[int]]]
            r_res: Arg[NOPResource]
            a_arg: Arg[int]

    # wrong resource
    with pytest.raises(TypeError):
        class TestSTMD(STMD):
            _traction: Type[Traction] = EmptyTraction
            i_input: In[TList[In[int]]]
            o_output: Out[TList[Out[int]]]
            r_res: Res[NOPResource]
            a_arg: In[str]

    # wrong doc arg
    with pytest.raises(TypeError):
        class TestSTMD(STMD):
            _traction: Type[Traction] = EmptyTraction
            i_input: In[TList[In[int]]]
            o_output: Out[TList[Out[int]]]
            r_res: Res[NOPResource]
            a_arg: Arg[str]
            d_: int

    # wrong attr doc arg
    with pytest.raises(TypeError):
        class TestSTMD(STMD):
            _traction: Type[Traction] = EmptyTraction
            i_input: In[TList[In[int]]]
            o_output: Out[TList[Out[int]]]
            r_res: Res[NOPResource]
            a_arg: Arg[str]
            d_a_arg: int = 0

    # uknown doc attr
    with pytest.raises(TypeError):
        class TestSTMD(STMD):
            _traction: Type[Traction] = EmptyTraction
            i_input: In[TList[In[int]]]
            o_output: Out[TList[Out[int]]]
            r_res: Res[NOPResource]
            a_arg: Arg[str]
            d_a_uknown: str = ""

    # custom attr
    with pytest.raises(TypeError):
        class TestSTMD(STMD):
            _traction: Type[Traction] = EmptyTraction
            i_input: In[TList[In[int]]]
            o_output: Out[TList[Out[int]]]
            r_res: Res[NOPResource]
            a_arg: Arg[str]
            custom: str

def test_stmd():
    c = Calculator(
        uid='calculator',
        i_inputs=In[TList[In[int]]](data=TList[In[int]]([In[int](data=x) for x in range(1,10)])),
        i_coeficient=STMDSingleIn[float](data=0.5)
    )
    c.run()


