import pytest

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
