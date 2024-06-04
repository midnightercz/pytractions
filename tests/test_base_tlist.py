from typing import TypeVar, Generic, Union

import pytest

from pytractions.base import Base, TList, TDict

# Jsonable test cases

T = TypeVar("T")


def test_base_list_new_ok():
    l: TList[int] = TList[int]([10])
    assert l._list == [10]


def test_base_list_new_ok_complex():
    l: TList[TDict[int, str]] = TList[TDict[int, str]]([TDict[int, str]({10: "a"})])
    assert l._list == [TDict[int, str]({10: "a"})]


def test_base_list_new_ok_generic():
    class TestC(Base, Generic[T]):
        l: TList[T]

    tc = TestC[int](l=TList[int]([10]))
    assert tc.l._list == [10]


def test_base_list_append_ok():
    l: TList[int] = TList[int]([10])
    l.append(20)

    assert l._list == [10, 20]


def test_base_list_extend_ok():
    l: TList[int] = TList[int]([10])
    l.extend(TList[int]([30, 40]))

    assert l._list == [10, 30, 40]


def test_base_list_add_ok():
    l: TList[int] = TList[int]([10])
    l2 = l + TList[int]([30, 40])
    assert l2._list == [10, 30, 40]


def test_base_list_insert_ok():
    l: TList[int] = TList[int]([10])
    l.insert(0, 5)

    assert l._list == [5, 10]


def test_base_list_setitem_ok():
    l: TList[int] = TList[int]([10])
    l[0] = 5

    assert l._list == [5]


def test_contains_ok():
    l: TList[int] = TList[int]([10])
    assert 10 in l


def test_base_list_new_fail():
    with pytest.raises(TypeError):
        TList[int](["a"])


def test_base_list_new_complex_fail():
    with pytest.raises(TypeError):
        TList[TDict[int, str]]([TDict[int, int]({10: 10})])


def test_base_list_new_generic_fail():
    class TestC(Base, Generic[T]):
        l: TList[T]

    with pytest.raises(TypeError):
        TestC[int](l=TList[int](["a"]))


def test_base_list_new_generic_no_type_fail():
    with pytest.raises(TypeError):
        TList([1, 2, 3])


def test_base_list_append_fail():
    l: TList[int] = TList[int]([10])
    with pytest.raises(TypeError):
        l.append("a")


def test_base_list_extend_fail():
    l: TList[int] = TList[int]([10])
    with pytest.raises(TypeError):
        l.extend(TList[str](["a", "a"]))


def test_base_list_insert_fail():
    l: TList[int] = TList[int]([10])
    with pytest.raises(TypeError):
        l.insert(0, "a")


def test_base_list_setitem_fail():
    l: TList[int] = TList[int]([10])
    with pytest.raises(TypeError):
        l[0] = "a"


def test_contains_fail():
    l: TList[int] = TList[int]([10])
    with pytest.raises(TypeError):
        assert "a" in l


def test_base_list_add_fail():
    l: TList[int] = TList[int]([10])
    with pytest.raises(TypeError):
        l + TList[str](["a", "a"])


def test_base_list_len():
    l: TList[int] = TList[int]([10, 20, 30])

    assert len(l) == 3


def test_del_item():
    l: TList[int] = TList[int]([10, 20, 30])
    del l[0]
    assert l._list == [20, 30]


def test_iter():
    l: TList[int] = TList[int]([10, 20, 30])
    ret = []
    for x in l:
        ret.append(x)
    assert ret == [10, 20, 30]


def test_clear():
    l: TList[int] = TList[int]([10, 20, 30])
    l.clear()
    assert l._list == []


def test_count():
    l: TList[int] = TList[int]([10, 20, 30])
    assert l.count(10) == 1


def test_index():
    l: TList[int] = TList[int]([10, 20, 30])
    assert l.index(20) == 1


class TestC(Base):
    """Test class for complex test cases."""

    x: int
    y: TDict[str, Union[str, TList[str]]]


def test_index_complex():
    l: TList[TestC] = TList[TestC](
        [
            TestC(
                x=10,
                y=TDict[str, Union[str, TList[str]]](
                    {"test": "test1", "foo": TList[str](["a", "b"])}
                ),
            ),
            TestC(
                x=20,
                y=TDict[str, Union[str, TList[str]]](
                    {"test": "test2", "foo": TList[str](["c", "d"])}
                ),
            ),
            TestC(
                x=30,
                y=TDict[str, Union[str, TList[str]]](
                    {"test": "test3", "foo": TList[str](["e", "f"])}
                ),
            ),
        ]
    )
    print(l)
    assert (
        l.index(
            TestC(
                x=20,
                y=TDict[str, Union[str, TList[str]]](
                    {"test": "test2", "foo": TList[str](["c", "d"])}
                ),
            )
        )
        == 1
    )


def test_insert():
    l: TList[int] = TList[int]([10, 20, 30])
    l.insert(2, 25)
    assert l._list == [10, 20, 25, 30]


def test_pop():
    l: TList[int] = TList[int]([10, 20, 25, 30])
    assert l.pop(2) == 25
    assert l._list == [10, 20, 30]


def test_remove():
    l: TList[int] = TList[int]([10, 20, 25, 30])
    l.remove(25)
    assert l._list == [10, 20, 30]


def test_reverse():
    l: TList[int] = TList[int]([10, 20, 30])
    l.reverse()
    assert l._list == [30, 20, 10]


def test_sort():
    l: TList[int] = TList[int]([10, 1, 20, 5, 30])
    l.sort()
    assert l._list == [1, 5, 10, 20, 30]
