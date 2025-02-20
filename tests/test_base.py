from typing import List, Dict, Union, Optional, TypeVar, Generic

import pytest

from pytractions.base import (
    Base,
    JSONIncompatibleError,
    TList,
    TDict,
    NoAnnotationError,
)

# Jsonable test cases

T = TypeVar("T")
T2 = TypeVar("T")


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
        l: TList[int]
        d: TDict[str, int]


def test_base_jsonable_basic_structured_optional_ok():
    class TestC(Base):
        l: Optional[TList[int]]
        d: TDict[str, Optional[int]]


def test_base_jsonable_basic_structured_union_ok():
    class TestC(Base):
        l: TList[int]
        d: TDict[str, Union[int, str]]


def test_base_jsonable_basic_nested_structured_ok():
    class TestC(Base):
        l: TList[TDict[str, int]]


def test_base_jsonable_basic_nested_structured_optional_ok():
    class TestC(Base):
        l: TList[Optional[TDict[str, Optional[int]]]]


def test_base_jsonable_basic_nested_structured_union_ok():
    class TestC(Base):
        l: TList[Union[TDict[str, int], TDict[str, str]]]


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
        c: TDict[str, Union[TestC1, int]]


def test_generic_jsonable_ok():
    class TestC1(Base, Generic[T]):
        x: T


def test_generic_jsonable_concrete_ok():
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
    class X:
        pass

    class TestC1(Base, Generic[T]):
        x: T

    with pytest.raises(JSONIncompatibleError):
        TestC1[X]


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


def test_base_setattr_complex_ok():
    class TestC(Base):
        s: TList[int]

    t = TestC(s=TList[int]([5]))
    t.s = TList[int]([10])


def test_base_setattr_complex2_ok():
    class TestC(Base, Generic[T]):
        td: TDict[str, T]

    d: TDict[str, int] = TDict[str, int]({"a": 10})

    t = TestC[int](td=TDict[str, int]({"b": 30}))
    t.td = d


# setattr validation test cases - fail


def test_base_setattr_fail():
    class TestC(Base):
        i: int
        s: str

    t = TestC(i=10, s="a")
    with pytest.raises(TypeError):
        t.i = "a"
    with pytest.raises(TypeError):
        t.s = 100


def test_base_setattr_missing_annotation_fail():
    with pytest.raises(NoAnnotationError):

        class TestC(Base):
            i = 10


def test_base_setattr_no_attr_fail():
    class TestC(Base):
        i: int
        s: str

    t = TestC(i=10, s="a")
    with pytest.raises(AttributeError):
        t.a = "a"


def test_base_setattr_optional_fail():
    class TestC(Base):
        s: Optional[str]

    t = TestC(s="a")
    print("---")
    t.s = None
    print("---")
    with pytest.raises(TypeError):
        t.s = 100


def test_base_setattr_union_fail():
    class TestC(Base):
        s: Union[str, int]

    t = TestC(s=10)
    with pytest.raises(TypeError):
        t.s = {}


def test_base_setattr_complex_fail():
    class TestC(Base):
        s: TList[str]

    t = TestC(s=TList[str](["a"]))
    with pytest.raises(TypeError):
        t.s = [10]


def test_base_setattr_complex1_fail():
    class TestC(Base):
        s: TList[int]

    l: TList[str] = TList[str](["a"])
    t = TestC(s=TList[int]([10]))
    with pytest.raises(TypeError):
        t.s = l


def test_base_setattr_complex2_fail():
    class TestList(Base, Generic[T]):
        l: TList[T]

    class TestC(Base):
        tl: TestList[int]

    _list = TList[str](["a"])

    tl2: TestList[str] = TestList[str](l=_list)

    t = TestC(tl=TestList[int](l=TList[int]([20])))
    with pytest.raises(TypeError):
        t.tl = tl2


def test_base_setattr_complex3_ok():

    class TestDict(Base, Generic[T]):
        d: TDict[str, T]

    class TestC(Base):
        td: TestDict[int]

    d: Dict[str, int] = {"a": 10}

    with pytest.raises(TypeError):
        TestDict[int](d=d)

    d2: TDict[str, int] = TDict[str, int]({"a": 10})
    d3: TDict[str, str] = TDict[str, str]({"a": "a"})

    td: TestDict[int] = TestDict[int](d=d2)
    t = TestC(td=td)
    td2: TestDict[str] = TestDict[str](d=d3)
    with pytest.raises(TypeError):
        t.td = td2


def test_base_generic_nested():
    class TestA(Base):
        pass

    class TestC(Base, Generic[T]):
        a: T

    TestC[TestA](a=TestA())


# Test Generics
def test_generic_wrong_param_count():
    class TestC(Base, Generic[T, T2]):
        a: T
        b: T2

    with pytest.raises(TypeError):
        TestC[int]
