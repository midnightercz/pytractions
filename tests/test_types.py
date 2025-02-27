try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from typing import Union, Optional, TypeVar, Generic


from pytractions.base import (
    Base,
    TList,
    JSON_COMPATIBLE,
    TypeNode,
)
from pytractions.utils import ANY

# Jsonable test cases

T = TypeVar("T")
T2 = TypeVar("T")

# Type testing


def test_type_json():
    t = TypeNode.from_type(TList[JSON_COMPATIBLE])
    tjson = t.to_json()
    print(tjson)
    t2 = TypeNode.from_json(tjson)
    print(t2.to_json())
    assert t == t2


def test_base_type_to_json():
    class TestC(Base):
        i: int
        s: str

    assert TestC.type_to_json() == {
        "$type": {
            "args": [],
            "module": "tests.test_types",
            "type": "test_base_type_to_json.<locals>.TestC",
        },
        "i": {
            "$type": {
                "args": [],
                "module": "builtins",
                "type": "int",
            },
            "default": None,
        },
        "s": {
            "$type": {
                "args": [],
                "module": "builtins",
                "type": "str",
            },
            "default": None,
        },
    }


def test_base_type_nested_to_json():
    class TestA(Base):
        a: str

    class TestC(Base):
        i: int
        s: str
        ta: TestA

    assert TestC.type_to_json() == {
        "$type": {
            "args": [],
            "module": "tests.test_types",
            "type": "test_base_type_nested_to_json.<locals>.TestC",
        },
        "i": {
            "$type": {
                "args": [],
                "module": "builtins",
                "type": "int",
            },
            "default": None,
        },
        "s": {
            "$type": {
                "args": [],
                "module": "builtins",
                "type": "str",
            },
            "default": None,
        },
        "ta": {
            "$type": {
                "args": [],
                "module": "tests.test_types",
                "type": "test_base_type_nested_to_json.<locals>.TestA",
            },
            "a": {
                "$type": {
                    "args": [],
                    "module": "builtins",
                    "type": "str",
                },
                "default": None,
            },
        },
    }


def test_type_union():
    t1 = TypeNode.from_type(Union[int, str])
    t2 = TypeNode.from_type(Union[str, int])
    assert t1 == t2


def test_type_union_subclass():
    class Ref(Base, Generic[T]):
        ref: T

    class TestA(Base):
        a: str

    class TestB(TestA):
        b: int

    t1 = TypeNode.from_type(Union[TestB, Ref[TestB]])
    t2 = TypeNode.from_type(Union[TestA, Ref[ANY]])
    assert t1 == t2


def test_type_union_optional():
    t1 = TypeNode.from_type(Optional[int])
    t2 = TypeNode.from_type(Union[int, str, bool, float, type(None)])
    assert t1 == t2


def test_self():
    class TestC(Base):
        prev: Optional[Self]

    tc1 = TestC(prev=None)
    TestC(prev=tc1)
