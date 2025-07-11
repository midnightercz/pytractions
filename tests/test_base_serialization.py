import pytest

from pytractions.base import Base, TList, TDict

from typing import Optional, Union, Literal
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


def test_tdict_complex():
    t: TDict[str, TDict[str, TList[str]]] = TDict[str, TDict[str, TList[str]]](
        {
            "a": TDict[str, TList[str]]({"b": TList[str](["c", "d"])}),
            "e": TDict[str, TList[str]]({"f": TList[str](["g", "h"])}),
        }
    )
    assert t.content_to_json() == {"a": {"b": ["c", "d"]}, "e": {"f": ["g", "h"]}}


def test_tdict_complex_from_json():
    t = TDict[str, TDict[str, TList[str]]].content_from_json(
        {"a": {"b": ["c", "d"]}, "e": {"f": ["g", "h"]}}
    )
    assert t == TDict[str, TDict[str, TList[str]]](
        {
            "a": TDict[str, TList[str]]({"b": TList[str](["c", "d"])}),
            "e": TDict[str, TList[str]]({"f": TList[str](["g", "h"])}),
        }
    )


def test_tdict_complex_from_json2():
    class TestModel(Base):
        foo: int
        bar: str

    assert TDict[str, TDict[str, TList[TestModel]]].content_from_json(
        {
            "identity": {
                "digest": [
                    {"foo": 1, "bar": "2"},
                ],
                "digest2": [
                    {"foo": 3, "bar": "3"},
                    {"foo": 4, "bar": "4"},
                ],
            }
        }
    ) == TDict[str, TDict[str, TList[TestModel]]](
        {
            "identity": TDict[str, TList[TestModel]](
                {
                    "digest": TList[TestModel]([TestModel(foo=1, bar="2")]),
                    "digest2": TList[TestModel](
                        [TestModel(foo=3, bar="3"), TestModel(foo=4, bar="4")]
                    ),
                }
            )
        }
    )


class GenericClass(Base):
    """Test class."""

    a: int


class SubclassClass1(GenericClass):
    """Test class."""

    cls_type: Literal["sub1"]
    b: str


class SubclassClass2(GenericClass):
    """Test class."""

    cls_type: Literal["sub2"]
    c: str


def test_subclasses_deserialization():
    subcls_i = SubclassClass1(a=1, b="test1", cls_type="sub1")
    content = subcls_i.content_to_json()

    new_subcls_i = GenericClass.content_from_json(content)

    assert subcls_i == new_subcls_i

    subcls_i = SubclassClass2(a=1, c="test2", cls_type="sub2")
    content = subcls_i.content_to_json()

    new_subcls_i = GenericClass.content_from_json(content)

    assert subcls_i == new_subcls_i


def test_json_schema():
    assert TestC.to_json_schema() == {
        "type": "object",
        "properties": {
            "foo": {"type": "integer", "title": "foo"},
            "bar": {"type": "string", "title": "bar"},
        },
        "required": ["foo", "bar"],
        "title": "TestC",
    }


def test_json_schema_complex():
    assert TestC3.to_json_schema() == {
        "type": "object",
        "properties": {
            "c2": {
                "type": "object",
                "properties": {
                    "attr1": {"type": "string", "title": "attr1"},
                    "attr2": {"type": "integer", "title": "attr2"},
                },
                "required": ["attr1", "attr2"],
                "title": "c2",
            },
            "foo": {"type": "integer", "title": "foo"},
            "complex_list": {
                "items": {
                    "properties": {
                        "attr1": {"type": "string", "title": "attr1"},
                        "attr2": {"type": "integer", "title": "attr2"},
                    },
                    "required": ["attr1", "attr2"],
                    "type": "object",
                    "title": "list item",
                },
                "type": "array",
                "title": "complex_list",
            },
            "bar": {"type": "string", "title": "bar"},
            "intlist": {
                "items": {"type": "integer", "title": "list item"},
                "type": "array",
                "title": "intlist",
            },
        },
        "required": ["c2", "foo", "bar", "intlist", "complex_list"],
        "title": "TestC3",
    }


def test_json_schema_complex2():
    assert TestC4.to_json_schema() == {
        "type": "object",
        "title": "TestC4",
        "properties": {
            "c2": {
                "type": "object",
                "properties": {
                    "attr1": {"type": "string", "title": "attr1"},
                    "attr2": {"type": "integer", "title": "attr2"},
                },
                "required": ["attr1", "attr2"],
                "title": "c2",
            },
            "foo": {"type": "integer", "title": "foo"},
            "complex_list": {
                "items": {
                    "properties": {
                        "attr1": {"type": "string", "title": "attr1"},
                        "attr2": {"type": "integer", "title": "attr2"},
                    },
                    "required": ["attr1", "attr2"],
                    "title": "list item",
                    "type": "object",
                },
                "type": "array",
                "title": "complex_list",
            },
            "complex_dict": {
                "items": {
                    "properties": {
                        "name": {"type": "string"},
                        "value": {
                            "properties": {
                                "attr1": {"type": "string", "title": "attr1"},
                                "attr2": {"type": "integer", "title": "attr2"},
                            },
                            "required": ["attr1", "attr2"],
                            "title": "None",
                            "type": "object",
                        },
                    },
                    "type": "object",
                },
                "type": "array",
                "title": "complex_dict",
            },
            "bar": {"type": "string", "title": "bar"},
            "c": {
                "oneOf": [
                    {
                        "properties": {
                            "bar": {"type": "string", "title": "bar"},
                            "foo": {"type": "integer", "title": "foo"},
                        },
                        "type": "object",
                        "required": ["foo", "bar"],
                        "title": "TestC",
                    },
                    {
                        "properties": {
                            "attr1": {"type": "string", "title": "attr1"},
                            "attr2": {"type": "integer", "title": "attr2"},
                        },
                        "type": "object",
                        "title": "TestC2",
                        "required": ["attr1", "attr2"],
                    },
                ],
                "title": "c",
                "type": "object",
            },
            "e": "string",
            "intlist": {
                "items": {"title": "list item", "type": "integer"},
                "title": "intlist",
                "type": "array",
            },
            "optional_str": {
                "title": "optional_str",
                "type": "string",
            },
            "union_arg": {
                "oneOf": [
                    {"type": "integer", "title": "<class 'int'>"},
                    {"type": "string", "title": "<class 'str'>"},
                ],
                "title": "union_arg",
                "type": "object",
            },
            "x": {"title": "x", "type": "string"},
        },
        "required": [
            "c2",
            "foo",
            "bar",
            "intlist",
            "complex_list",
            "complex_dict",
            "union_arg",
            "c",
            "e",
            "x",
        ],
    }
