import pytest

from pytractions.base import Base, Port, NullPort
from pytractions.traction import Traction
from pytractions.tractor import Tractor
from pytractions.exc import WrongInputMappingError, WrongArgMappingError


def test_tractor_attr():
    with pytest.raises(TypeError):

        class TT5(Tractor):
            t_traction: int

    with pytest.raises(TypeError):

        class TT6(Tractor):
            custom_attribute: int


class Seq(Base):
    """Sequence resource."""

    val: int = 0

    def inc(self):
        """Return incremented value."""
        self.val += 1
        return self.val


class TestTraction(Traction):
    """Test Traction."""

    i_input: Port[float]
    o_output: Port[int]
    r_seq: Port[Seq]

    def _run(self) -> None:
        print("----------- RUN ----------")
        print("IN", self.i_input)
        self.o_output = int(self.i_input) + self.r_seq.inc()


class TestTractor(Tractor):
    """Test Tractor."""

    i_in1: Port[float] = NullPort[float]()
    r_seq: Port[Seq] = NullPort[Seq]()
    t_t1: TestTraction = TestTraction(uid="traction-1", i_input=i_in1, r_seq=r_seq)
    o_out1: Port[int] = t_t1._raw_o_output


class TestTractor2(Tractor):
    """Test Tractor2."""

    i_in1: Port[float] = NullPort()
    r_seq: Port[Seq] = NullPort[Seq]()
    t_tractor1: TestTractor = TestTractor(uid="tractor-1", i_in1=i_in1, r_seq=r_seq)
    o_out1: Port[int] = t_tractor1._raw_o_out1


def test_tractor_nested():
    seq = Seq(val=10)
    t = TestTractor2(uid="1", i_in1=Port[float](data=1.0), r_seq=Port[Seq](data=seq))
    t.run()
    print(
        "TRACTOR-1 RAW_OUT",
        id(t.tractions["t_tractor1"]._raw_o_out1),
        t.tractions["t_tractor1"]._raw_o_out1,
    )
    print("TRACTOR-1 OUT", id(t.tractions["t_tractor1"].o_out1), t.tractions["t_tractor1"].o_out1)
    print("TRACTOR-2 RAW OOT", id(t._raw_o_out1), t._raw_o_out1)
    print("TRACTOR-2 OUT", id(t.o_out1), t.o_out1)
    assert t.tractions["t_tractor1"].o_out1 == 12
    assert t.tractions["t_tractor1"].i_in1 == 1
    assert t.o_out1 == 12


class Complex(Base):
    """Complex number."""

    real: float = 0.0
    imaginary: float = 0.0


class TestTractionComplex(Traction):
    """Test Traction."""

    i_real: Port[float]
    i_imaginary: Port[float]
    o_output: Port[Complex]

    def _run(self) -> None:
        print("REAL", self.i_real, "IMAGINARY", self.i_imaginary)
        self.o_output.real = self.i_real
        self.o_output.imaginary = self.i_imaginary


class TestTractorNestedAttrs(Tractor):
    """Test Tractor."""

    i_in1: Port[float] = NullPort[float]()
    i_in2: Port[float] = NullPort[float]()
    r_seq: Port[Seq] = NullPort[Seq]()
    t_t1: TestTractionComplex = TestTractionComplex(uid="1", i_real=i_in1, i_imaginary=i_in2)
    t_t2: TestTraction = TestTraction(uid="2", i_input=t_t1.o_output.real, r_seq=r_seq)

    o_out1: Port[int] = t_t2.o_output


def test_tractor_nested_attrs():
    print(TestTractorNestedAttrs._io_map)
    print(TestTractorNestedAttrs._outputs_map)
    t = TestTractorNestedAttrs(
        uid="1",
        i_in1=Port[float](data=33.0),
        i_in2=Port[float](data=22.0),
        r_seq=Port[Seq](data=Seq(val=10)),
    )

    t.run()
    print(t.o_out1)
    assert t.o_out1 == 44


def test_tractor_input_to_arg():
    """Test traction arg a_arg is set to tractor input i_in1. Which should fail."""

    class TestTractionArg(Traction):
        a_arg: float
        o_output: float

        def _run(self) -> None:
            self.o_output = self.i_arg

    with pytest.raises(WrongArgMappingError):

        class TestTractorInputToArg(Tractor):
            i_in1: Port[float] = NullPort[float]()
            t_t1: TestTractionArg = TestTractionArg(uid="1", a_arg=i_in1)
            o_out1: Port[float] = t_t1.o_output


def test_tractor_arg_to_input():
    """Test traction input i_int is set to tractor arg a_arg1. Which should fail."""

    class TestTractionInt(Traction):
        i_int: float
        o_output: float

        def _run(self) -> None:
            self.o_output = self.i_int

    with pytest.raises(WrongInputMappingError):

        class TestTractorArgToInput(Tractor):
            a_arg1: Port[float] = NullPort[float]()
            t_t1: TestTractionInt = TestTractionInt(uid="1", i_int=a_arg1)
            o_out1: Port[float] = t_t1.o_output


def test_tractor_default_input():

    class TestTractionInt(Traction):
        i_int: float = 213.0
        o_output: float

        def _run(self) -> None:
            self.o_output = self.i_int

    class TestTractorArgToInput(Tractor):
        t_t1: TestTractionInt = TestTractionInt(uid="1")
        o_out1: Port[float] = t_t1.o_output

    t = TestTractorArgToInput(uid="1")
    t.run()
    assert t.o_out1 == 213.0


def test_tractor_default_input_overwrite():

    class TestTractionInt(Traction):
        i_int: float = 213.0
        o_output: float

        def _run(self) -> None:
            self.o_output = self.i_int

    class TestTractorArgToInput(Tractor):
        i_in1: Port[float] = NullPort[float]()
        t_t1: TestTractionInt = TestTractionInt(uid="1", i_int=i_in1)
        o_out1: Port[float] = t_t1.o_output

    t = TestTractorArgToInput(uid="1", i_in1=Port[float](data=123.0))
    t.run()
    assert t.o_out1 == 123.0


def test_tractor_none_input():

    class TestTractionInt(Traction):
        i_int: float = 213.0
        o_output: float

        def _run(self) -> None:
            self.o_output = self.i_int

    class TestTractorArgToInput(Tractor):
        i_in1: Port[float] = NullPort[float]()
        t_t1: TestTractionInt = TestTractionInt(uid="1", i_int=i_in1)
        o_out1: Port[float] = t_t1.o_output

    with pytest.raises(TypeError):
        TestTractorArgToInput(uid="1", i_in1=None)
