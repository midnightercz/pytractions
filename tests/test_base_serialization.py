import pytest

from pytractions.base import Base, TList, TDict

from typing import Optional, Union
from enum import Enum


class TestC(Base):
    """Test class for Base."""

    foo: int
    bar: str


class TestC2(Base):
    """Test class for Base."""

    attr1: str
    attr2: int


class TestC3(Base):
    """Test class for Base."""

    c2: TestC2
    foo: int
    bar: str
    intlist: TList[int]
    complex_list: TList[TestC2]


class TestEnum(str, Enum):
    """Test enum."""

    A = "A"
    B = "B"
    C = "C"


class TestC4(TestC3):
    """Test class for Base."""

    complex_dict: TDict[str, TestC2]
    optional_str: Optional[str]
    union_arg: Union[int, str]
    c: Union[TestC, TestC2]
    e: TestEnum
    x: str


def test_base_to_json_simple():
    tc = TestC(foo=10, bar="bar")
    assert tc.to_json() == {
        "$data": {"foo": 10, "bar": "bar"},
        "$type": {
            "args": [],
            "type": "TestC",
            "module": "tests.test_base_serialization",
        },
    }


def test_base_from_json_simple_no_type():
    json_data = {
        "$data": {"foo": 10, "bar": "bar"},
        "$type": {
            "args": [],
            "type": "TestC",
            "module": "tests.test_base_serialization",
        },
    }
    tc = Base.from_json(json_data)
    assert tc.foo == 10
    assert tc.bar == "bar"
    assert tc.__class__ == TestC


def test_base_from_json_simple():
    class TestC(Base):
        foo: int
        bar: str

    tc = TestC.from_json({"foo": 10, "bar": "bar"})
    assert tc.foo == 10
    assert tc.bar == "bar"


def test_base_from_json_complex():
    class TestC2(Base):
        attr1: str
        attr2: int

    class TestC(Base):
        foo: int
        bar: str
        c2: TestC2

    tc = TestC.from_json({"foo": 10, "bar": "bar", "c2": {"attr1": "a", "attr2": 20}})
    assert tc.foo == 10
    assert tc.bar == "bar"
    assert tc.c2.attr1 == "a"
    assert tc.c2.attr2 == 20


def test_base_from_json_complex_no_type():
    jdata = {
        "$data": {
            "foo": 10,
            "bar": "bar",
            "c2": {
                "$data": {"attr1": "a", "attr2": 10},
                "$type": {"args": [], "type": "TestC2", "module": "tests.test_base_serialization"},
            },
            "intlist": {
                "$data": [20, 40],
                "$type": {
                    "args": [{"args": [], "type": "int", "module": "builtins"}],
                    "type": "TList",
                    "module": "pytractions.base",
                },
            },
            "complex_list": {
                "$data": [],
                "$type": {
                    "args": [
                        {"args": [], "type": "TestC2", "module": "tests.test_base_serialization"}
                    ],
                    "type": "TList",
                    "module": "pytractions.base",
                },
            },
        },
        "$type": {
            "args": [],
            "type": "TestC3",
            "module": "tests.test_base_serialization",
        },
    }
    tc = Base.from_json(jdata)
    assert tc.__class__ == TestC3
    assert tc.foo == 10
    assert tc.bar == "bar"
    assert tc.c2.attr1 == "a"
    assert tc.c2.attr2 == 10
    assert tc.intlist == TList[int]([20, 40])
    assert tc.complex_list == TList[TestC2]([])


def test_base_from_json_simple_fail_wrong_arg_type():
    class TestC(Base):
        foo: int
        bar: str

    with pytest.raises(TypeError):
        TestC.from_json({"foo": "a", "bar": "bar"})


def test_base_from_json_simple_fail_extra_arg():
    class TestC(Base):
        foo: int
        bar: str

    with pytest.raises(ValueError):
        tc = TestC.from_json({"foo": 10, "bar": "bar", "extra": "arg"})
        print(tc)


def test_base_to_json_complex():
    tc = TestC3(
        foo=10,
        bar="bar",
        c2=TestC2(attr1="a", attr2=10),
        intlist=TList[int]([20, 40]),
        complex_list=TList[TestC2]([]),
    )
    assert tc.to_json() == {
        "$data": {
            "foo": 10,
            "bar": "bar",
            "c2": {
                "$data": {"attr1": "a", "attr2": 10},
                "$type": {"args": [], "type": "TestC2", "module": "tests.test_base_serialization"},
            },
            "intlist": {
                "$data": [20, 40],
                "$type": {
                    "args": [{"args": [], "type": "int", "module": "builtins"}],
                    "type": "TList",
                    "module": "pytractions.base",
                },
            },
            "complex_list": {
                "$data": [],
                "$type": {
                    "args": [
                        {"args": [], "type": "TestC2", "module": "tests.test_base_serialization"}
                    ],
                    "type": "TList",
                    "module": "pytractions.base",
                },
            },
        },
        "$type": {
            "args": [],
            "type": "TestC3",
            "module": "tests.test_base_serialization",
        },
    }


def test_base_to_from_json_complex():
    tc = TestC3(
        foo=10,
        bar="bar",
        c2=TestC2(attr1="a", attr2=10),
        intlist=TList[int]([20, 40]),
        complex_list=TList[TestC2]([]),
    )
    tc2 = TestC.from_json(tc.to_json())
    assert tc == tc2


def test_base_content_to_json():
    tc2_1 = TestC2(attr1="tc2-str1-", attr2=10)
    tc2_2 = TestC2(attr1="tc2-str1-", attr2=10)
    tc = TestC4(
        foo=10,
        bar="bar",
        c2=TestC2(attr1="a", attr2=10),
        intlist=TList[int]([20, 40]),
        complex_list=TList[TestC2]([tc2_1, tc2_2]),
        complex_dict=TDict[str, TestC2](dict(a=tc2_1, b=tc2_2)),
        optional_str=None,
        union_arg=10,
        c=TestC(foo=10, bar="bar"),
        e=TestEnum.A,
        x="a",
    )
    tc_content = TestC4.content_to_json(tc)
    print(tc_content)
    tc2 = TestC4.content_from_json(tc_content)
    assert tc == tc2
