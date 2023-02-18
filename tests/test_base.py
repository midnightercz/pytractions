from typing import List, Dict, Union, Optional, TypeVar, Generic

import pytest

from pytraction.base import Base, JSONIncompatibleError

# Jsonable test cases

def test_base_jsonable_basic_ok():
    class TestC(Base):
        i: int
        s: str


def test_base_jsonable_basic_union_ok():
    class TestC(Base):
        i: Union[int, str]


def test_base_jsonable_basic_optional():
    class TestC(Base):
        i: Optional[int]


def test_base_jsonable_basic_structured_ok():
    class TestC(Base):
        l: List[int]
        d: Dict[str, int]


def test_base_jsonable_basic_structured_optional_ok():
    class TestC(Base):
        l: Optional[List[int]]
        d: Dict[str, Optional[int]]


def test_base_jsonable_basic_structured_union_ok():
    class TestC(Base):
        l: List[int]
        d: Dict[str, Union[int, str]]


def test_base_jsonable_basic_nested_structured_ok():
    class TestC(Base):
        l: List[Dict[str, int]]


def test_base_jsonable_basic_nested_structured_optional_ok():
    class TestC(Base):
        l: List[Optional[Dict[str, Optional[int]]]]


def test_base_jsonable_basic_nested_structured_union_ok():
    class TestC(Base):
        l: List[Union[Dict[str, int], Dict[str, str]]]


def test_base_jsonable_basic_clasess_ok():
    class TestC1(Base):
        i: int

    class TestC(Base):
        c: TestC1


def test_base_jsonable_basic_clasess_optional_ok():
    class TestC1(Base):
        i: int

    class TestC(Base):
        c: Optional[TestC1]


def test_base_jsonable_basic_clasess_union_ok():
    class TestC1(Base):
        i: int

    class TestC(Base):
        c: Dict[str, Union[TestC1, int]]


def test_generic_jsonable_ok():
    T = TypeVar("T")

    class TestC1(Base, Generic[T]):
        x: T


def test_generic_jsonable_concrete_ok():
    T = TypeVar("T")

    class TestC1(Base, Generic[T]):
        x: T

    TestC1[int]

# Jsonable expected to fail cases


def test_base_jsonable_basic_fail():
    with pytest.raises(JSONIncompatibleError):
        class TestC(Base):
            i: tuple
            s: str


def test_base_jsonable_basic_union_fail():
    with pytest.raises(JSONIncompatibleError):
        class TestC(Base):
            i: Union[int, tuple]


def test_base_jsonable_basic_structured_fail():
    with pytest.raises(JSONIncompatibleError):
        class TestC(Base):
            l: List[int]
            d: Dict[str, tuple]


def test_base_jsonable_basic_structured_union_fail():
    with pytest.raises(JSONIncompatibleError):
        class TestC(Base):
            l: List[int]
            d: Dict[str, Union[tuple, str]]


def test_base_jsonable_basic_nested_structured_fail():
    with pytest.raises(JSONIncompatibleError):
        class TestC(Base):
            l: List[Dict[str, tuple]]


def test_base_jsonable_basic_nested_structured_union_fail():
    with pytest.raises(JSONIncompatibleError):
        class TestC(Base):
            l: List[Union[Dict[str, tuple], Dict[str, str]]]


def test_base_jsonable_basic_clasess_fail():
    class TestC1:
        i: int

    with pytest.raises(JSONIncompatibleError):
        class TestC(Base):
            c: TestC1


def test_base_jsonable_basic_clasess_union_fail():
    class TestC1:
        i: int

    with pytest.raises(JSONIncompatibleError):
        class TestC(Base):
            c: Dict[str, Union[TestC1, int]]


def test_generic_jsonable_concrete_fail():
    T = TypeVar("T")

    class X:
        pass

    class TestC1(Base, Generic[T]):
        x: T

    TestC1[X]
    assert False

# setattr validation test cases


def test_base_setattr_ok():
    class TestC(Base):
        i: int
        s: str

    t = TestC(i=10, s="a")
    t.i = 100
    t.s = "a"

def test_base_setattr_optional_ok():
    class TestC(Base):
        s: Optional[str]

    t = TestC(s="a")
    t.s = None


def test_base_setattr_union_ok():
    class TestC(Base):
        s: Union[str, int]

    t = TestC(s="a")
    t.s = 10


def test_base_setattr_union_ok():
    class TestC(Base):
        s: Union[str, int]

    t = TestC(s="a")
    t.s = 10


def test_base_setattr_complex_ok():
    class TestC(Base):
        s: List[int]

    t = TestC(s=["a"])
    t.s = [10]


def test_base_setattr_complex1_ok():
    class TestC(Base):
        s: List[int]

    l: List[int] = [10]
    t = TestC(s=["a"])
    t.s = l


def test_base_setattr_complex1_ok():

    T = TypeVar("T")

    class TestList(Base, Generic[T]):
        l: List[T]

    class TestC(Base):
        tl: TestList[int]
    
    l: List[str]  = ["a"] 

    tl2: TestList[int] = TestList[int](l=l)

    t = TestC(tl=TestList[int](l=[20]))
    t.tl = tl2
