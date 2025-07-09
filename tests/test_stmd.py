from typing import Type, Union

import pytest

from pytractions.base import (
    TList,
    Port,
    Base,
    STMDSingleIn,
)
from pytractions.traction import Traction
from pytractions.stmd import STMD

from pytractions.executor import ThreadPoolExecutor, ProcessPoolExecutor, LoopExecutor, RayExecutor

from pytractions.tractor import Tractor


class NOPResource(Base):
    """No operation resource."""

    pass


class NOPResource2(Base):
    """No operation resource2."""

    pass


class EmptyTraction(Traction):
    """Empty traction."""

    i_input: Port[int]
    o_output: Port[int]
    a_arg: Port[int]
    r_res: Port[NOPResource]

    def _run(self) -> None:  # pragma: no cover
        if not self.i_input.data:
            print(self.uid, "DATA", self.i_input.data)
        if not self.i_coeficient.data:
            print(self.uid, "COEF", self.i_coeficient.data)
        self.o_output.data = self.i_input.data * 2 / self.i_coeficient.data


class Double(Traction):
    """Double the input and divide by coeficient."""

    i_input: Port[float]
    i_coeficient: Port[float]
    o_output: Port[float]

    def _run(self) -> None:  # pragma: no cover
        if not self.i_input:
            print(self.uid, "DATA", self.i_input)
        if not self.i_coeficient:
            print(self.uid, "COEF", self.i_coeficient)
        self.o_output = self.i_input * 2 / self.i_coeficient


class STMDDouble(STMD):
    """Double the inputs and divide by coeficient."""

    _traction: Type[Traction] = Double

    i_input: Port[TList[float]]
    i_coeficient: STMDSingleIn[float]

    o_output: Port[TList[float]]


class Half(Traction):
    """Half the input and multiply by coeficient."""

    i_input: Port[float]
    i_coeficient: Port[float]
    o_output: Port[float]

    def _run(self) -> None:  # pragma: no cover
        if not self.i_input:
            print(self.uid, "DATA", self.i_input)
        if not self.i_coeficient:
            print(self.uid, "COEF", self.i_coeficient)
        self.o_output = self.i_input / 2 * self.i_coeficient


class STMDHalf(STMD):
    """Half the inputs and multiply by coeficient."""

    _traction: Type[Traction] = Half

    i_input: Port[TList[float]]
    i_coeficient: STMDSingleIn[float]
    o_output: Port[TList[float]]


class Calculator(Tractor):
    """Calculator."""

    i_inputs: Port[TList[float]] = Port[TList[float]]()
    i_coeficient: STMDSingleIn[float] = STMDSingleIn[float]()
    a_executor: Port[Union[ProcessPoolExecutor, ThreadPoolExecutor, LoopExecutor]] = Port[
        Union[ProcessPoolExecutor, ThreadPoolExecutor, LoopExecutor]
    ](data=ThreadPoolExecutor(pool_size=30, executor_type="thread_pool_executor"))

    t_double: STMDDouble = STMDDouble(
        uid="double",
        i_input=i_inputs,
        i_coeficient=i_coeficient,
        a_executor=a_executor,
    )
    print("RAW", t_double.i_input, id(t_double._raw_o_output))

    t_half: STMDHalf = STMDHalf(
        uid="half",
        i_input=t_double._raw_o_output,
        i_coeficient=i_coeficient,
        a_executor=a_executor,
    )

    o_output: Port[TList[float]] = t_half._raw_o_output


def test_stmd_attr_validation():
    # wrong doc arg
    with pytest.raises(TypeError):

        class TestSTMD5(STMD):
            _traction: Type[Traction] = EmptyTraction
            i_input: Port[TList[Port[int]]]
            o_output: Port[TList[Port[int]]]
            r_res: Port[NOPResource]
            a_arg: Port[str]
            d_: int

    # wrong attr doc arg
    with pytest.raises(TypeError):

        class TestSTMD6(STMD):
            _traction: Type[Traction] = EmptyTraction
            i_input: Port[TList[Port[int]]]
            o_output: Port[TList[Port[int]]]
            r_res: Port[NOPResource]
            a_arg: Port[str]
            d_a_arg: int = 0

    # uknown doc attr
    with pytest.raises(TypeError):

        class TestSTMD7(STMD):
            _traction: Type[Traction] = EmptyTraction
            i_input: Port[TList[Port[int]]]
            o_output: Port[TList[Port[int]]]
            r_res: Port[NOPResource]
            a_arg: Port[str]
            d_a_uknown: str = ""

    # custom attr
    with pytest.raises(TypeError):

        class TestSTMD(STMD):
            _traction: Type[Traction] = EmptyTraction
            i_input: Port[TList[Port[int]]]
            o_output: Port[TList[Port[int]]]
            r_res: Port[NOPResource]
            a_arg: Port[str]
            custom: str

    # wrong inner type
    with pytest.raises(TypeError):

        class TestSTMD8(STMD):
            _traction: Type[Traction] = EmptyTraction
            i_input: Port[TList[str]]
            o_output: Port[TList[Port[int]]]
            r_res: Port[NOPResource]
            a_arg: Port[str]
            custom: str


def test_stmd_calculator():
    c = Calculator(
        uid="calculator",
        i_inputs=Port[TList[float]](data=TList[float]([float(x) for x in range(1, 10)])),
        i_coeficient=Port[float](data=0.5),
    )
    c.run()


class G_TTest1(Traction):
    """Test traction."""

    o_out1: Port[float]
    i_in1: Port[Union[float, int]]
    a_multiplier: Port[float]

    def _run(self) -> None:  # pragma: no cover
        self.o_out1 = self.i_in1 * self.a_multiplier


class Complex(Base):
    """Test class."""

    real: float = 0.0
    imag: float = 0.0


class G_TTest2(Traction):
    """Test traction."""

    o_out1: Port[Complex]
    i_in1: Port[Complex]
    a_multiplier: Port[float]

    def _run(self) -> None:  # pragma: no cover
        self.o_out1 = self.i_in1


class G_TestTractor(Tractor):
    """Test tractor."""

    i_in1: Port[Union[float, int]] = Port[Union[float, int]](data=20.0)
    a_multiplier: Port[float] = Port[float](data=1.0)

    t_traction_1: G_TTest1 = G_TTest1(uid="1", i_in1=i_in1, a_multiplier=a_multiplier)
    t_traction_2: G_TTest1 = G_TTest1(uid="2", i_in1=t_traction_1.o_out1, a_multiplier=a_multiplier)

    o_out1: Port[float] = t_traction_2._raw_o_out1


def test_stmd_tractor(fixture_isodate_now) -> None:
    class TestSTMD(STMD):
        a_multiplier: Port[float] = Port[float](data=0.0)
        i_in1: Port[TList[Union[float, int]]] = Port[TList[Union[float, int]]](
            data=TList[Union[float, int]]([])
        )

        _traction: Type[Traction] = G_TestTractor

        o_out1: Port[TList[float]] = Port[TList[float]](data=TList[float]([]))

    stmd_in1 = Port[TList[float]](data=TList[float]([1.0, 2.0, 3.0, 4.0, 5.0]))
    thread_pool_executor = ThreadPoolExecutor(pool_size=1, executor_type="thread_pool_executor")

    stmd1 = TestSTMD(
        uid="tt1",
        a_multiplier=Port[float](data=10.0),
        i_in1=stmd_in1,
        a_executor=Port[Union[ProcessPoolExecutor, ThreadPoolExecutor, LoopExecutor]](
            data=thread_pool_executor
        ),
    )
    stmd1.run()
    assert stmd1.o_out1[0] == 100.0
    assert stmd1.o_out1[1] == 200.0
    assert stmd1.o_out1[2] == 300.0
    assert stmd1.o_out1[3] == 400.0
    assert stmd1.o_out1[4] == 500.0


def test_stmd_local(fixture_isodate_now) -> None:

    class TestSTMD(STMD):
        a_multiplier: Port[float] = Port[float](data=0.0)

        _traction: Type[Traction] = G_TTest1

        o_out1: Port[TList[float]] = Port[TList[float]](data=TList[float]([]))
        i_in1: Port[TList[float]]

    stmd_in1 = Port[TList[float]](data=TList[float]([1.0, 2.0, 3.0, 4.0, 5.0]))
    stmd1 = TestSTMD(uid="tt1", a_multiplier=Port[float](data=10.0), i_in1=stmd_in1)
    stmd1.run()
    assert stmd1.o_out1[0] == 10.0
    assert stmd1.o_out1[1] == 20.0
    assert stmd1.o_out1[2] == 30.0
    assert stmd1.o_out1[3] == 40.0
    assert stmd1.o_out1[4] == 50.0


def test_stmd_threads(fixture_isodate_now) -> None:

    class TestSTMD(STMD):
        a_multiplier: Port[float] = Port[float](data=0.0)

        _traction: Type[Traction] = G_TTest2

        o_out1: Port[TList[Complex]] = Port[TList[Complex]](data=TList[Complex]([]))
        i_in1: Port[TList[Complex]]

    stmd_in1 = Port[TList[Complex]](
        data=TList[Complex](
            [
                Complex(real=1.0, imag=1.0),
                Complex(real=2.0, imag=1.0),
                Complex(real=3.0, imag=1.0),
                Complex(real=4.0, imag=1.0),
                Complex(real=5.0, imag=1.0),
            ]
        )
    )
    thread_pool_executor = ThreadPoolExecutor(pool_size=1, executor_type="thread_pool_executor")
    stmd1 = TestSTMD(
        uid="tt1",
        a_executor=Port[Union[ProcessPoolExecutor, ThreadPoolExecutor, LoopExecutor]](
            data=thread_pool_executor
        ),
        a_multiplier=Port[float](data=10.0),
        i_in1=stmd_in1,
    )
    stmd1.run()
    assert stmd1.o_out1[0] == Complex(real=1.0, imag=1.0)
    assert stmd1.o_out1[1] == Complex(real=2.0, imag=1.0)
    assert stmd1.o_out1[2] == Complex(real=3.0, imag=1.0)
    assert stmd1.o_out1[3] == Complex(real=4.0, imag=1.0)
    assert stmd1.o_out1[4] == Complex(real=5.0, imag=1.0)


class GTestSTMD(STMD):
    """Test STMD."""

    a_multiplier: Port[float] = Port[float](data=0.0)

    _traction: Type[Traction] = G_TTest1

    o_out1: Port[TList[float]] = Port[TList[float]](data=TList[float]([]))
    i_in1: Port[TList[float]]


def test_stmd_processes(fixture_isodate_now) -> None:

    process_pool_executor = ProcessPoolExecutor(pool_size=1, executor_type="process_pool_executor")
    stmd_in1 = Port[TList[float]](
        data=TList[float](
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ]
        )
    )
    stmd1 = GTestSTMD(
        uid="tt1",
        a_executor=Port[Union[ProcessPoolExecutor, ThreadPoolExecutor, LoopExecutor]](
            data=process_pool_executor
        ),
        a_multiplier=Port[float](data=10.0),
        i_in1=stmd_in1,
    )
    stmd1.run()
    assert stmd1.o_out1[0] == 10.0
    assert stmd1.o_out1[1] == 20.0
    assert stmd1.o_out1[2] == 30.0
    assert stmd1.o_out1[3] == 40.0
    assert stmd1.o_out1[4] == 50.0


def test_wrap_stmd_threadpool(fixture_isodate_now) -> None:

    stmd_in1 = Port[TList[float]](
        data=TList[float](
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ]
        )
    )
    thread_pool_executor = ThreadPoolExecutor(pool_size=1, executor_type="thread_pool_executor")

    stmd1 = STMD.wrap(G_TTest1)(
        uid="tt1",
        a_executor=Port[Union[ProcessPoolExecutor, ThreadPoolExecutor, LoopExecutor]](
            data=thread_pool_executor
        ),
        a_multiplier=Port[float](data=10.0),
        i_in1=stmd_in1,
    )
    stmd1.run()
    assert stmd1.o_out1[0] == 10.0
    assert stmd1.o_out1[1] == 20.0
    assert stmd1.o_out1[2] == 30.0
    assert stmd1.o_out1[3] == 40.0
    assert stmd1.o_out1[4] == 50.0


def test_wrap_stmd_processpool(fixture_isodate_now) -> None:

    stmd_in1 = Port[TList[float]](
        data=TList[float](
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ]
        )
    )
    process_pool_executor = ProcessPoolExecutor(pool_size=1, executor_type="process_pool_executor")

    stmd1 = STMD.wrap(G_TTest1)(
        uid="tt1",
        a_executor=Port[Union[ProcessPoolExecutor, ThreadPoolExecutor, LoopExecutor]](
            data=process_pool_executor
        ),
        a_multiplier=Port[float](data=10.0),
        i_in1=stmd_in1,
    )
    stmd1.run()
    assert stmd1.o_out1[0] == 10.0
    assert stmd1.o_out1[1] == 20.0
    assert stmd1.o_out1[2] == 30.0
    assert stmd1.o_out1[3] == 40.0
    assert stmd1.o_out1[4] == 50.0


def test_wrap_stmd_ray(fixture_isodate_now) -> None:

    stmd_in1 = Port[TList[float]](
        data=TList[float](
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ]
        )
    )
    ray_executor = RayExecutor(pool_size=1)

    stmd1 = STMD.wrap(G_TTest1)(
        uid="tt1",
        a_executor=Port[Union[ProcessPoolExecutor, ThreadPoolExecutor, LoopExecutor, RayExecutor]](
            data=ray_executor
        ),
        a_multiplier=Port[float](data=10.0),
        i_in1=stmd_in1,
    )
    stmd1.run()
    assert stmd1.o_out1[0] == 10.0
    assert stmd1.o_out1[1] == 20.0
    assert stmd1.o_out1[2] == 30.0
    assert stmd1.o_out1[3] == 40.0
    assert stmd1.o_out1[4] == 50.0
