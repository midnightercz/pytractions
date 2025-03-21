from typing import Union, Type

from pytractions.base import (
    TList,
    NullPort,
    Port,
    Base,
)
from pytractions.traction import Traction
from pytractions.tractor import Tractor
from pytractions.stmd import STMD


def test_traction_ok_args_1():
    class TTest(Traction):
        i_in1: int
        o_out1: int
        r_res1: int
        a_arg1: int


def test_traction_simple_io():
    class TTest1(Traction):
        i_in: int
        o_out: int

        def _run(self) -> None:  # pragma: no cover
            self.o_out = self.i_in + 10

    t = TTest1(uid="1", i_in=10)
    t.run()
    assert t.o_out == 20


def test_traction_outputs_no_init():
    class TTest(Traction):
        o_out1: int

    t = TTest(uid="1")
    assert t.o_out1 == 0


def test_traction_outputs_no_init_custom_default():
    class TTest(Traction):
        o_out1: int = 10

    t = TTest(uid="1")
    assert t.o_out1 == 10


def test_traction_chain():
    class TTest1(Traction):
        o_out1: int

        def _run(self) -> None:  # pragma: no cover
            self.o_out1 = 20

    class TTest2(Traction):
        i_in1: int
        o_out1: int

        def _run(self) -> None:  # pragma: no cover
            print("IN", self.i_in1)
            self.o_out1 = self.i_in1 + 10

    t1 = TTest1(uid="1")
    t2 = TTest2(uid="1", i_in1=t1._raw_o_out1)

    t1.run()
    t2.run()
    assert t2.o_out1 == 30


def test_traction_chain_in_to_out():
    class TTest1(Traction):
        o_out1: int

        def _run(self) -> None:  # pragma: no cover
            self.o_out1 = 20

    class TTest2(Traction):
        i_in1: int
        o_out1: int

        def _run(self) -> None:  # pragma: no cover
            self.o_out1 = self.i_in1

    t1 = TTest1(uid="1")
    t2 = TTest2(uid="1", i_in1=t1._raw_o_out1)

    t1.run()
    t2.run()
    assert t2.o_out1 == 20
    t1.o_out1 = 30

    assert t2.i_in1 == 30


def test_tractor_run() -> None:
    class TTest2(Traction):
        i_in1: float
        o_out1: float
        a_reducer: float

        def _run(self) -> None:  # pragma: no cover
            print("I", self.i_in1, "/", "A", self.a_reducer)
            self.o_out1 = (self.i_in1 + 1) / float(self.a_reducer)

    class TestTractor(Tractor):
        a_multiplier: Port[float] = Port[float](data=0.0)
        a_reducer: Port[float] = Port[float](data=0.0)

        i_in1: Port[float] = Port[float](data=1.0)

        t_ttest1: TTest2 = TTest2(uid="1", a_reducer=a_reducer)
        t_ttest2: TTest2 = TTest2(uid="2", a_reducer=a_reducer)
        t_ttest3: TTest2 = TTest2(uid="3", a_reducer=a_reducer)
        t_ttest4: TTest2 = TTest2(uid="4", a_reducer=a_reducer)

        t_ttest1.i_in1 = i_in1

        t_ttest2.i_in1 = t_ttest1.o_out1
        t_ttest3.i_in1 = t_ttest2.o_out1
        t_ttest4.i_in1 = t_ttest3.o_out1

        o_out1: float = t_ttest4.o_out1

    tt = TestTractor(
        uid="tt1",
        a_multiplier=10.0,
        a_reducer=2.0,
        i_in1=10.0,
    )
    tt.run()
    assert tt.o_out1 == 1.5625


class Seq(Base):
    """Sequence resource."""

    val: int = 0

    def inc(self):
        """Return incremented value."""
        self.val += 1
        return self.val


class TestTraction(Traction):
    """Test Traction."""

    i_input: int
    o_output: int
    r_seq: Seq

    def _run(self) -> None:
        self.o_output = self.i_input + self.r_seq.inc()


class TestTractorX(Tractor):
    """Test Tractor."""

    i_in1: int = NullPort[int]()
    r_seq: Seq = NullPort[Seq]()
    t_t1: TestTraction = TestTraction(uid="1", i_input=i_in1, r_seq=r_seq)
    o_out1: int = t_t1.o_output


class TestTractor2(Tractor):
    """Test Tractor2."""

    i_in1: int = NullPort[int]()
    r_seq: Seq = NullPort[Seq]()
    t_tractor1: TestTractorX = TestTractorX(uid="1", i_in1=i_in1, r_seq=r_seq)
    o_out1: int = t_tractor1.o_out1


def test_tractor_nested():
    seq = Seq(val=10)
    t = TestTractor2(uid="1", i_in1=Port[int](data=1), r_seq=Port[Seq](data=seq))
    t.run()
    assert t.o_out1 == 12
    assert t.tractions["t_tractor1"].o_out1 == 12
    assert t.tractions["t_tractor1"].i_in1 == 1


class G_TTest1(Traction):
    """Test traction."""

    o_out1: float
    i_in1: Union[float, int]
    a_multiplier: float

    def _run(self) -> None:  # pragma: no cover
        print("I", self.i_in1, "*", "A", self.a_multiplier)
        self.o_out1 = self.i_in1 * self.a_multiplier


def test_stmd_local(fixture_isodate_now) -> None:

    class TestSTMD(STMD):
        a_multiplier: Port[float]

        _traction: Type[Traction] = G_TTest1

        o_out1: TList[float]
        i_in1: TList[float]

    stmd_in1 = Port[TList[float]](data=TList[float]([1.0, 2.0, 3.0, 4.0, 5.0]))
    stmd1 = TestSTMD(uid="tt1", a_multiplier=Port[float](data=10.0), i_in1=stmd_in1)
    stmd1.run()
    assert stmd1.o_out1[0] == 10.0
    assert stmd1.o_out1[1] == 20.0
    assert stmd1.o_out1[2] == 30.0
    assert stmd1.o_out1[3] == 40.0
    assert stmd1.o_out1[4] == 50.0
