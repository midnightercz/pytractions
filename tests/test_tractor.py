
from typing import List, Dict, Union, Optional, TypeVar, Generic

import pytest

from pytractions.base import Base, JSONIncompatibleError, TList, TDict, JSON_COMPATIBLE, TypeNode, NoAnnotationError
from pytractions.tractor import Tractor

def test_tractor_attr():
    with pytest.raises(TypeError):
        class TT(Tractor):
            i_in1: int

    with pytest.raises(TypeError):
        class TT(Tractor):
            o_out1: int

    with pytest.raises(TypeError):
        class TT(Tractor):
            a_arg1: int

    with pytest.raises(TypeError):
        class TT(Tractor):
            r_res1: int

    with pytest.raises(TypeError):
        class TT(Tractor):
            t_traction: int

    with pytest.raises(TypeError):
        class TT(Tractor):
            custom_attribute: int
