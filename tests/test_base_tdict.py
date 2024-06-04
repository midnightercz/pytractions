from typing import TypeVar, Generic, Union

import pytest

from pytractions.base import Base, TList, TDict

# Jsonable test cases

T = TypeVar("T")


def test_base_dict_new_ok():
    d: TDict[int, int] = TDict[int, int]({10: 10})
    assert d._dict == {10: 10}


def test_base_dict_new_ok_complex():
    d: TDict[int, TList[int]] = TDict[int, TList[int]]({10: TList[int]([10])})
    assert d._dict == {10: TList[int]([10])}


def test_base_dict_new_ok_generic():
    class TestC(Base, Generic[T]):
        d: TDict[int, T]

    tc = TestC[int](d=TDict[int, int]({10: 10}))
    assert tc.d._dict == {10: 10}


def test_base_dict_get_ok():
    d: TDict[int, int] = TDict[int, int]({10: 10})

    assert d._dict.get(10) == 10
    assert d._dict.get(20) is None


def test_base_dict_update_ok():
    d: TDict[int, int] = TDict[int, int]({10: 10})
    d.update(TDict[int, int]({20: 20}))
    assert d._dict == {10: 10, 20: 20}


def test_base_dict_fromkeys_ok():
    d: TDict[int, int] = TDict[int, int]({10: 10})
    d2 = d.fromkeys([10, 20], 10)
    assert d2._dict == {10: 10, 20: 10}
    assert d2 != d


def test_base_dict_contains_ok():
    d: TDict[int, int] = TDict[int, int]({10: 10})
    assert 10 in d


def test_base_dict_getitem_ok():
    d: TDict[int, int] = TDict[int, int]({10: 10})
    assert d[10] == 10


def test_base_dict_setitem_ok():
    d: TDict[int, int] = TDict[int, int]({10: 10})
    d[10] = 20
    assert d._dict == {10: 20}


def test_base_dict_delitem_ok():
    d: TDict[int, int] = TDict[int, int]({10: 10, 20: 20})
    del d[20]
    assert d._dict == {10: 10}


def test_base_dict_pop_ok():
    d: TDict[int, int] = TDict[int, int]({10: 10, 20: 20})
    d.pop(10) == 10
    assert d._dict == {20: 20}


def test_base_dict_setdefault_ok():
    d: TDict[int, int] = TDict[int, int]({10: 10, 20: 20})
    d.setdefault(30, 10)
    assert d[30] == 10


def test_base_dict_new_fail():
    with pytest.raises(TypeError):
        TDict[int, int]({10: "a"})


def test_base_dict_new_fail_complex():
    with pytest.raises(TypeError):
        TDict[int, TList[int]]({10: TList[str](["a"])})


def test_base_dict_new_fail_generic():
    class TestC(Base, Generic[T]):
        d: TDict[int, T]

    with pytest.raises(TypeError):
        TestC[int](d=TDict[int, int]({10: "a"}))


def test_base_dict_get_fail():
    d: TDict[int, int] = TDict[int, int]({10: 10})

    with pytest.raises(TypeError):
        d.get("a")


def test_base_dict_update_fail():
    d: TDict[int, int] = TDict[int, int]({10: 10})
    with pytest.raises(TypeError):
        d.update(TDict[int, str]({20: "a"}))


def test_base_dict_fromkeys_fail():
    d: TDict[int, int] = TDict[int, int]({10: 10})
    with pytest.raises(TypeError):
        d.fromkeys([10, "a"], 10)

    with pytest.raises(TypeError):
        d.fromkeys([10, 20], "a")


def test_base_dict_contains_fail():
    d: TDict[int, int] = TDict[int, int]({10: 10})
    with pytest.raises(TypeError):
        assert "a" in d


def test_base_dict_getitem_fail():
    d: TDict[int, int] = TDict[int, int]({10: 10})
    with pytest.raises(TypeError):
        assert d["a"] == 10


def test_base_dict_setitem_fail():
    d: TDict[int, int] = TDict[int, int]({10: 10})
    with pytest.raises(TypeError):
        d["a"] = 10
    with pytest.raises(TypeError):
        d[10] = "a"


def test_base_dict_delitem_fail():
    d: TDict[int, int] = TDict[int, int]({10: 10, 20: 20})
    with pytest.raises(TypeError):
        del d["a"]


def test_base_dict_pop_fail():
    d: TDict[int, int] = TDict[int, int]({10: 10, 20: 20})
    with pytest.raises(TypeError):
        d.pop("a")


def test_base_dict_setdefault_fail():
    d: TDict[int, int] = TDict[int, int]({10: 10, 20: 20})
    with pytest.raises(TypeError):
        d.setdefault("a", 10)


def test_base_dict_len():
    d: TDict[int, int] = TDict[int, int]({10: 10, 20: 20})
    assert len(d) == 2


def test_base_dict_iter():
    d: TDict[int, int] = TDict[int, int]({10: 10, 20: 20})
    assert [i for i in d] == [10, 20]


def test_base_dict_clear():
    d: TDict[int, int] = TDict[int, int]({10: 10, 20: 20})
    d.clear()
    assert d._dict == {}


def test_base_dict_keys():
    d: TDict[int, int] = TDict[int, int]({10: 10, 20: 20})
    assert list(d.keys()) == [10, 20]


def test_base_dict_values():
    d: TDict[int, int] = TDict[int, int]({10: 10, 20: 20})
    assert list(d.values()) == [10, 20]


def test_base_dict_items():
    d: TDict[int, int] = TDict[int, int]({10: 10, 20: 20})
    assert list(d.items()) == [(10, 10), (20, 20)]


def test_base_dict_popitem():
    d: TDict[int, int] = TDict[int, int]({10: 10, 20: 20})
    assert d.popitem() == (20, 20)


def test_base_dict_to_json_simple():
    d: TDict[int, int] = TDict[int, int]({10: 10, 20: 20})
    assert d.to_json() == {
        "$type": {
            "args": [
                {
                    "args": [],
                    "module": "builtins",
                    "type": "int",
                },
                {
                    "args": [],
                    "module": "builtins",
                    "type": "int",
                },
            ],
            "module": "pytractions.base",
            "type": "TDict",
        },
        "$data": {10: 10, 20: 20},
    }


def test_base_dict_to_json_complex():
    d: TDict[int, int] = TDict[int, TDict[str, int]](
        {10: TDict[str, int]({"a": 10}), 20: TDict[str, int]({"b": 20})}
    )
    assert d.to_json() == {
        "$type": {
            "args": [
                {
                    "args": [],
                    "module": "builtins",
                    "type": "int",
                },
                {
                    "args": [
                        {
                            "args": [],
                            "module": "builtins",
                            "type": "str",
                        },
                        {
                            "args": [],
                            "module": "builtins",
                            "type": "int",
                        },
                    ],
                    "module": "pytractions.base",
                    "type": "TDict",
                },
            ],
            "module": "pytractions.base",
            "type": "TDict",
        },
        "$data": {
            10: {
                "$data": {
                    "a": 10,
                },
                "$type": {
                    "args": [
                        {
                            "args": [],
                            "module": "builtins",
                            "type": "str",
                        },
                        {
                            "args": [],
                            "module": "builtins",
                            "type": "int",
                        },
                    ],
                    "module": "pytractions.base",
                    "type": "TDict",
                },
            },
            20: {
                "$data": {
                    "b": 20,
                },
                "$type": {
                    "args": [
                        {
                            "args": [],
                            "module": "builtins",
                            "type": "str",
                        },
                        {
                            "args": [],
                            "module": "builtins",
                            "type": "int",
                        },
                    ],
                    "module": "pytractions.base",
                    "type": "TDict",
                },
            },
        },
    }


def test_base_dict_to_json_complex_key():
    key1: TList[str] = TList[str](["a", "b", "c"])
    key2: TList[str] = TList[str](["d", "e", "f"])
    d: TDict[int, int] = TDict[TList[str], int]({key1: 10, key2: 20})
    assert d.to_json() == {
        "$data": {
            '{"$data": ["a", "b", "c"], "$type": {"args": [{"args": [], "module": "builtins", '
            '"type": "str"}], "module": "pytractions.base", "type": "TList"}}': 10,
            '{"$data": ["d", "e", "f"], "$type": {"args": [{"args": [], "module": "builtins", '
            '"type": "str"}], "module": "pytractions.base", "type": "TList"}}': 20,
        },
        "$type": {
            "args": [
                {
                    "args": [
                        {
                            "args": [],
                            "module": "builtins",
                            "type": "str",
                        },
                    ],
                    "module": "pytractions.base",
                    "type": "TList",
                },
                {
                    "args": [],
                    "module": "builtins",
                    "type": "int",
                },
            ],
            "module": "pytractions.base",
            "type": "TDict",
        },
    }


def test_base_dict_from_json_simple():
    json_dict = {
        "$type": {
            "args": [
                {
                    "args": [],
                    "module": "builtins",
                    "type": "int",
                },
                {
                    "args": [],
                    "module": "builtins",
                    "type": "int",
                },
            ],
            "module": "pytractions.base",
            "type": "TDict",
        },
        "$data": {10: 10, 20: 20},
    }
    d: TDict[int, int] = TDict[int, int].from_json(json_dict)
    assert d == TDict[int, int]({10: 10, 20: 20})


def test_base_dict_content_from_json():
    d = TDict[str, Union[str, TList[str]]].content_from_json(
        {
            "config_file": "test",
            "digest": ["sha256:6ef06d8c90c863ba4eb4297f1073ba8cb28c1f6570e2206cdaad2084e2a4715d"],
            "reference": ["quay.io/namespace/image:1"],
            "signing_key": "signing_key",
        }
    )
    assert d == TDict[str, Union[str, TList[str]]](
        {
            "config_file": "test",
            "digest": TList[str](
                ["sha256:6ef06d8c90c863ba4eb4297f1073ba8cb28c1f6570e2206cdaad2084e2a4715d"]
            ),
            "reference": TList[str](["quay.io/namespace/image:1"]),
            "signing_key": "signing_key",
        }
    )
