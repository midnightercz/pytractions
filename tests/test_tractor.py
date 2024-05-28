import pytest

from pytractions.base import Traction, TIn, In, Out, OnUpdateCallable, Res, TRes, Base
from pytractions.tractor import Tractor


def test_tractor_attr():
    with pytest.raises(TypeError):

        class TT1(Tractor):
            i_in1: int

    with pytest.raises(TypeError):

        class TT2(Tractor):
            o_out1: int

    with pytest.raises(TypeError):

        class TT3(Tractor):
            a_arg1: int

    with pytest.raises(TypeError):

        class TT4(Tractor):
            r_res1: int

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

    i_input: In[int]
    o_output: Out[int]
    r_seq: Res[Seq]

    def _run(self, on_update: OnUpdateCallable) -> None:
        self.o_output.data = self.i_input.data + self.r_seq.r.inc()


class TestTractor(Tractor):
    """Test Tractor."""

    i_in1: In[int] = TIn[int]()
    r_seq: Res[Seq] = TRes[Seq]()

    t_t1: TestTraction = TestTraction(uid="1", i_input=i_in1, r_seq=r_seq)

    o_out1: Out[int] = t_t1.o_output


class TestTractor2(Tractor):
    """Test Tractor2."""

    i_in1: In[int] = TIn()
    r_seq: Res[Seq] = TRes[Seq]()

    t_tractor1: TestTractor = TestTractor(uid="1", i_in1=i_in1, r_seq=r_seq)

    o_out1: Out[int] = t_tractor1.o_out1


def test_tractor_nested():
    seq = Seq(val=10)
    t = TestTractor2(uid="1", i_in1=In[int](data=1), r_seq=Res[Seq](r=seq))
    t.run()
    assert t.o_out1.data == 12
    assert t.tractions["t_tractor1"].o_out1.data == 12
    assert t.tractions["t_tractor1"].i_in1.data == 1
