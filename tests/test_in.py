import pytest

from pytractions.base import In, Base, Traction, OnUpdateCallable, TList


def test_input_type_check():
    with pytest.raises(TypeError):
        In[str](data=1)
