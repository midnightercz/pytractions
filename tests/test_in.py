import pytest

from pytractions.base import Port


def test_input_type_check():
    with pytest.raises(TypeError):
        Port[str](data=1)
