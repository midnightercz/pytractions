from typing import Union, TypeVar

import pytest

from pytractions.base import (
    TList,
    Port,
    Base,
    NullPort,
    TypeNode,
)

from pytractions.traction import Traction
from pytractions.tractor import Tractor
from pytractions.exc import WrongInputMappingError


T = TypeVar("T")


def test_traction_ok_args_1():
    class TTest(Traction):
        i_in1: Port[int]
        o_out1: Port[int]
        r_res1: Port[int]
        a_arg1: Port[int]


def test_traction_ok_args_description_1():
    class TTest(Traction):
        i_in1: Port[int]
        o_out1: Port[int]
        r_res1: Port[int]
        a_arg1: Port[int]
        d_i_in1: str = "Description of i_in1"


# def test_traction_wrong_args_1():
#     with pytest.raises(TypeError):
#
#         class TTest(Traction):
#             i_in1: Port[int]
#
#
# def test_traction_wrong_args_2():
#     with pytest.raises(TypeError):
#
#         class TTest(Traction):
#             o_out1: Port[int]
#
#
# def test_traction_wrong_args_3():
#     with pytest.raises(TypeError):
#
#         class TTest(Traction):
#             a_arg1: Port[int]
#
#
# def test_traction_wrong_args_4():
#     with pytest.raises(TypeError):
#
#         class TTest(Traction):
#             r_res1: Port[int]


def test_traction_inputs_1():
    class TTest(Traction):
        i_in1: Port[int] = Port[int]()

    o: Port[int] = Port[int](data=10)
    TTest(uid="1", i_in1=o)


def test_traction_inputs_read_unset():
    class TTest(Traction):
        i_in1: Port[int]

    t = TTest(uid="1")
    assert TypeNode.from_type(type(t._raw_i_in1)) == TypeNode.from_type(NullPort[int])


def test_traction_inputs_read_set():
    class TTest(Traction):
        i_in1: Port[int]

    o: Port[int] = Port[int](data=10)
    t = TTest(uid="1", i_in1=o)
    assert TypeNode.from_type(type(t._raw_i_in1)) == TypeNode.from_type(Port[int])
    assert t.i_in1 == 10


def test_traction_to_json():
    class TTest(Traction):
        i_in1: Port[int] = Port[int]()
        d_i_in1: str = "description of i_in1"

    o: Port[int] = Port[int](data=10)
    t = TTest(uid="1", i_in1=o)
    assert t.to_json() == {
        "$data": {
            "d_i_in1": "description of i_in1",
            "details": {
                "$data": {},
                "$type": {
                    "args": [
                        {"args": [], "type": "str", "module": "builtins"},
                        {"args": [], "type": "str", "module": "builtins"},
                    ],
                    "type": "TDict",
                    "module": "pytractions.base",
                },
            },
            "errors": {
                "$data": [],
                "$type": {
                    "args": [{"args": [], "type": "str", "module": "builtins"}],
                    "type": "TList",
                    "module": "pytractions.base",
                },
            },
            "i_in1": {
                "$data": {"data": 10},
                "$type": {
                    "args": [{"args": [], "type": "int", "module": "builtins"}],
                    "type": "Port",
                    "module": "pytractions.base",
                },
            },
            "skip": False,
            "skip_reason": "",
            "state": "ready",
            "stats": {
                "$data": {"finished": "", "skipped": False, "started": ""},
                "$type": {"args": [], "type": "TractionStats", "module": "pytractions.traction"},
            },
            "uid": "1",
        },
        "$type": {
            "args": [],
            "module": "tests.test_base_traction",
            "type": "test_traction_to_json.<locals>.TTest",
        },
    }


def test_traction_inlist_to_json():
    class TTest(Traction):
        i_in1: Port[TList[int]]

    o: Port[TList[int]] = Port[TList[int]](data=TList[int]([10]))
    t = TTest(uid="1", i_in1=o)
    assert t.to_json() == {
        "$data": {
            "details": {
                "$data": {},
                "$type": {
                    "args": [
                        {"args": [], "type": "str", "module": "builtins"},
                        {"args": [], "type": "str", "module": "builtins"},
                    ],
                    "type": "TDict",
                    "module": "pytractions.base",
                },
            },
            "errors": {
                "$data": [],
                "$type": {
                    "args": [{"args": [], "type": "str", "module": "builtins"}],
                    "type": "TList",
                    "module": "pytractions.base",
                },
            },
            "i_in1": {
                "$data": {
                    "data": {
                        "$data": [10],
                        "$type": {
                            "args": [{"args": [], "module": "builtins", "type": "int"}],
                            "module": "pytractions.base",
                            "type": "TList",
                        },
                    }
                },
                "$type": {
                    "args": [
                        {
                            "args": [{"args": [], "module": "builtins", "type": "int"}],
                            "module": "pytractions.base",
                            "type": "TList",
                        }
                    ],
                    "module": "pytractions.base",
                    "type": "Port",
                },
            },
            "skip": False,
            "skip_reason": "",
            "state": "ready",
            "stats": {
                "$data": {"finished": "", "skipped": False, "started": ""},
                "$type": {"args": [], "type": "TractionStats", "module": "pytractions.traction"},
            },
            "uid": "1",
        },
        "$type": {
            "args": [],
            "module": "tests.test_base_traction",
            "type": "test_traction_inlist_to_json.<locals>.TTest",
        },
    }


def test_traction_to_from_json():
    class TTest(Traction):
        i_in1: Port[int]  # = In[int]()

    o: Port[int] = Port[int](data=10)
    t = TTest(uid="1", i_in1=o)
    t2 = TTest.from_json(t.to_json(), _locals=locals())
    assert t == t2


def test_traction_outputs_no_init():
    class TTest(Traction):
        o_out1: Port[int]

    t = TTest(uid="1")
    assert t.o_out1 == 0


def test_traction_outputs_no_init_custom_default():
    class TTest(Traction):
        o_out1: Port[int] = Port[int](data=10)

    t = TTest(uid="1")
    assert t.o_out1 == 10


def test_traction_chain():
    class TTest1(Traction):
        o_out1: Port[int]

        def _run(self) -> None:  # pragma: no cover
            self.o_out1 = 20

    class TTest2(Traction):
        i_in1: Port[int]
        o_out1: Port[int]

        def _run(self) -> None:  # pragma: no cover
            print("IN", self.i_in1)
            self.o_out1 = self.i_in1 + 10

    t1 = TTest1(uid="1")
    t2 = TTest2(uid="1", i_in1=t1._raw_o_out1)

    t1.run()
    t2.run()
    print(t2._raw_o_out1)
    assert t2.o_out1 == 30


def test_traction_chain_in_to_out():
    class TTest1(Traction):
        o_out1: Port[int]

        def _run(self) -> None:  # pragma: no cover
            self.o_out1 = 20

    class TTest2(Traction):
        i_in1: Port[int]
        o_out1: Port[int]

        def _run(self) -> None:  # pragma: no cover
            self.o_out1 = self.i_in1

    t1 = TTest1(uid="1")
    t2 = TTest2(uid="1", i_in1=t1._raw_o_out1)

    t1.run()
    t2.run()
    assert t2.o_out1 == 20
    t1.o_out1 = 30

    assert t2.i_in1 == 30


def test_traction_json(fixture_isodate_now):
    class TTest1(Traction):
        o_out1: Port[int]

        def _run(self) -> None:  # pragma: no cover
            self.o_out1 = 20

    class TTest2(Traction):
        i_in1: Port[Union[int, float]]
        o_out1: Port[int]

        def _run(self) -> None:  # pragma: no cover
            self.o_out1 = self.i_in1 + 10

    t1 = TTest1(uid="1")
    t2 = TTest2(uid="2", i_in1=t1._raw_o_out1)

    t1.run()
    t2.run()
    assert t1.to_json() == {
        "$data": {
            "details": {
                "$data": {},
                "$type": {
                    "args": [
                        {"args": [], "type": "str", "module": "builtins"},
                        {"args": [], "type": "str", "module": "builtins"},
                    ],
                    "type": "TDict",
                    "module": "pytractions.base",
                },
            },
            "errors": {
                "$data": [],
                "$type": {
                    "args": [{"args": [], "type": "str", "module": "builtins"}],
                    "type": "TList",
                    "module": "pytractions.base",
                },
            },
            "o_out1": {
                "$data": {"data": 20},
                "$type": {
                    "args": [{"args": [], "type": "int", "module": "builtins"}],
                    "type": "Port",
                    "module": "pytractions.base",
                },
            },
            "skip": False,
            "skip_reason": "",
            "state": "finished",
            "stats": {
                "$data": {
                    "finished": "1990-01-01T00:00:01.00000Z",
                    "skipped": False,
                    "started": "1990-01-01T00:00:00.00000Z",
                },
                "$type": {"args": [], "module": "pytractions.traction", "type": "TractionStats"},
            },
            "uid": "1",
        },
        "$type": {
            "args": [],
            "module": "tests.test_base_traction",
            "type": "test_traction_json.<locals>.TTest1",
        },
    }

    assert t2.to_json() == {
        "$data": {
            "details": {
                "$data": {},
                "$type": {
                    "args": [
                        {"args": [], "type": "str", "module": "builtins"},
                        {"args": [], "type": "str", "module": "builtins"},
                    ],
                    "type": "TDict",
                    "module": "pytractions.base",
                },
            },
            "errors": {
                "$data": [],
                "$type": {
                    "args": [{"args": [], "type": "str", "module": "builtins"}],
                    "type": "TList",
                    "module": "pytractions.base",
                },
            },
            "i_in1": "TTest1[1]#o_out1",
            "o_out1": {
                "$data": {"data": 30},
                "$type": {
                    "args": [{"args": [], "type": "int", "module": "builtins"}],
                    "type": "Port",
                    "module": "pytractions.base",
                },
            },
            "skip": False,
            "skip_reason": "",
            "state": "finished",
            "stats": {
                "$data": {
                    "finished": "1990-01-01T00:00:03.00000Z",
                    "skipped": False,
                    "started": "1990-01-01T00:00:02.00000Z",
                },
                "$type": {"args": [], "module": "pytractions.traction", "type": "TractionStats"},
            },
            "uid": "2",
        },
        "$type": {
            "args": [],
            "module": "tests.test_base_traction",
            "type": "test_traction_json.<locals>.TTest2",
        },
    }


def test_tractor_members_order() -> None:
    class TTest1(Traction):
        o_out1: Port[float]
        a_multiplier: Port[float]

        def _run(self) -> None:  # pragma: no cover
            self.o_out1.data = 20 * self.a_multiplier.a

    class TTest2(Traction):
        i_in1: Port[float]
        o_out1: Port[float]
        a_reducer: Port[float]

        def _run(self) -> None:  # pragma: no cover
            self.o_out1.data = (self.i_in1.data + 10) / float(self.a_reducer.a)

    class TestTractor(Tractor):
        a_multiplier: Port[float] = Port[float](data=0.0)
        a_reducer: Port[float] = Port[float](data=0.0)

        t_ttest1: TTest1 = TTest1(uid="1", a_multiplier=a_multiplier)
        t_ttest4: TTest2 = TTest2(uid="4", a_reducer=a_reducer)
        t_ttest3: TTest2 = TTest2(uid="3", a_reducer=a_reducer)
        t_ttest2: TTest2 = TTest2(uid="2", a_reducer=a_reducer)

        t_ttest2.i_in1 = t_ttest1.o_out1
        t_ttest3.i_in1 = t_ttest4.o_out1
        t_ttest4.i_in1 = t_ttest1.o_out1

        o_out1: Port[float] = t_ttest4.o_out1

    ttrac = TestTractor(uid="t1")

    tractions = []
    for k, v in ttrac.__dict__.items():
        if k.startswith("t_"):
            tractions.append(k)

    assert tractions == ["t_ttest1", "t_ttest4", "t_ttest3", "t_ttest2"]


def test_tractor_members_invalid_order() -> None:
    class TTest1(Traction):
        o_out1: Port[float]
        a_multiplier: Port[float]

        def _run(self) -> None:  # pragma: no cover
            self.o_out1.data = 20 * self.a_multiplier.data

    class TTest2(Traction):
        i_in1: Port[float]
        o_out1: Port[float]
        a_reducer: Port[float]

        def _run(self) -> None:  # pragma: no cover
            self.o_out1.data = (self.i_in1.data + 10) / float(self.a_reducer.data)

    with pytest.raises(WrongInputMappingError):

        class TestTractor(Tractor):
            a_multiplier: Port[float] = Port[float](data=0.0)
            a_reducer: Port[float] = Port[float](data=0.0)

            t_ttest1: TTest1 = TTest1(uid="1", a_multiplier=a_multiplier)
            t_ttest2: TTest2 = TTest2(uid="4", a_reducer=a_reducer)
            t_ttest4: TTest2 = TTest2(uid="3", a_reducer=a_reducer)
            t_ttest3: TTest2 = TTest2(uid="2", a_reducer=a_reducer)

            t_ttest2.i_in1 = t_ttest1.o_out1
            t_ttest4.i_in1 = t_ttest3.o_out1

            o_out1: Port[float] = t_ttest4.o_out1


def test_tractor_run() -> None:
    class TTest2(Traction):
        i_in1: Port[float]
        o_out1: Port[float]
        a_reducer: Port[float]

        def _run(self) -> None:  # pragma: no cover
            print(self.uid)
            print("I", self.i_in1, "/", "A", self.a_reducer)
            print("ID RAW BEFORE RUN", id(self._raw_o_out1))
            print("ID BEFORE RUN", id(self.o_out1))
            self.o_out1 = (self.i_in1 + 1) / float(self.a_reducer)
            print("ID RAW AFTER RUN", id(self._raw_o_out1))
            print("ID AFTER RUN", id(self.o_out1))

    class TestTractor(Tractor):
        a_multiplier: Port[float] = Port[float](data=0.0)
        a_reducer: Port[float] = Port[float](data=0.0)

        i_in1: Port[float] = Port[float](data=1.0)

        t_ttest1: TTest2 = TTest2(uid="1", a_reducer=a_reducer)
        t_ttest2: TTest2 = TTest2(uid="2", a_reducer=a_reducer)
        t_ttest3: TTest2 = TTest2(uid="3", a_reducer=a_reducer)
        t_ttest4: TTest2 = TTest2(uid="4", a_reducer=a_reducer)

        t_ttest1.i_in1 = i_in1

        t_ttest2.i_in1 = t_ttest1.o_out1
        t_ttest3.i_in1 = t_ttest2.o_out1
        t_ttest4.i_in1 = t_ttest3.o_out1

        o_out1: Port[float] = t_ttest4.o_out1

    tt = TestTractor(
        uid="tt1",
        a_multiplier=Port[float](data=10.0),
        a_reducer=Port[float](data=2.0),
        i_in1=Port[float](data=10.0),
    )
    tt.run()
    print("FINISHED")
    print(id(tt._raw_o_out1))
    print(id(tt.o_out1))
    assert tt.o_out1 == 1.5625


class UselessResource(Base):
    """Testing resource."""

    values_stack: TList[int]

    def get_some_value(self) -> int:
        """Get first value from the init list."""
        return self.values_stack.pop(0)


def test_tractor_run_resources() -> None:

    class TTest2(Traction):
        i_in1: Port[float]
        o_out1: Port[float]
        a_reducer: Port[float]
        r_useless_res: Port[UselessResource]

        def _run(self) -> None:  # pragma: no cover
            useless_value = self.r_useless_res.get_some_value()
            self.o_out1 = (self.i_in1 + useless_value) / float(self.a_reducer)
            print("IN", self.i_in1, "/", "A", self.a_reducer, "/", "U", useless_value)

    class TestTractor(Tractor):
        a_multiplier: Port[float] = Port[float](data=0.0)
        a_reducer: Port[float] = Port[float](data=0.0)

        r_useless: Port[UselessResource] = Port[UselessResource](
            data=UselessResource(values_stack=TList[int](TList[int]([])))
        )

        i_in1: Port[float] = Port[float](data=1.0)

        t_ttest1: TTest2 = TTest2(uid="1", a_reducer=a_reducer, r_useless_res=r_useless)
        t_ttest2: TTest2 = TTest2(uid="2", a_reducer=a_reducer, r_useless_res=r_useless)
        t_ttest3: TTest2 = TTest2(uid="3", a_reducer=a_reducer, r_useless_res=r_useless)
        t_ttest4: TTest2 = TTest2(uid="4", a_reducer=a_reducer, r_useless_res=r_useless)

        t_ttest1.i_in1 = i_in1
        t_ttest2.i_in1 = t_ttest1.o_out1
        t_ttest3.i_in1 = t_ttest2.o_out1
        t_ttest4.i_in1 = t_ttest3.o_out1

        o_out1: Port[float] = t_ttest4.o_out1

    tt = TestTractor(
        uid="tt1",
        a_multiplier=Port[float](data=10.0),
        a_reducer=Port[float](data=2.0),
        i_in1=Port[float](data=10.0),
        r_useless=Port[UselessResource](
            data=UselessResource(values_stack=TList[int]([2, 5, 1, 7, 2]))
        ),
    )

    tt.run()
    assert tt.o_out1 == 5.125


def test_tractor_to_json(fixture_isodate_now) -> None:
    class TTest1(Traction):
        o_out1: Port[float]
        a_multiplier: Port[float]

        def _run(self) -> None:  # pragma: no cover
            self.o_out1 = 1 * self.a_multiplier

    class TTest2(Traction):
        i_in1: Port[float]
        o_out1: Port[float]
        a_reducer: Port[float]

        def _run(self) -> None:  # pragma: no cover
            self.o_out1 = (self.i_in1 + 1) / float(self.a_reducer)

    class TestTractor(Tractor):
        a_multiplier: Port[float] = Port[float](data=0.0)
        a_reducer: Port[float] = Port[float](data=0.0)

        t_ttest1: TTest1 = TTest1(uid="1", a_multiplier=a_multiplier)
        t_ttest2: TTest2 = TTest2(uid="2", a_reducer=a_reducer)
        t_ttest3: TTest2 = TTest2(uid="3", a_reducer=a_reducer)
        t_ttest4: TTest2 = TTest2(uid="4", a_reducer=a_reducer)

        t_ttest2.i_in1 = t_ttest1.o_out1
        t_ttest3.i_in1 = t_ttest2.o_out1
        t_ttest4.i_in1 = t_ttest3.o_out1

        o_out1: Port[float] = t_ttest4.o_out1

    tt = TestTractor(
        uid="tt1", a_multiplier=Port[float](data=10.0), a_reducer=Port[float](data=2.0)
    )

    tt.run()

    assert tt.to_json() == {
        "$data": {
            "a_multiplier": {
                "$data": {"data": 10.0},
                "$type": {
                    "args": [{"args": [], "type": "float", "module": "builtins"}],
                    "type": "Port",
                    "module": "pytractions.base",
                },
            },
            "a_reducer": {
                "$data": {"data": 2.0},
                "$type": {
                    "args": [{"args": [], "type": "float", "module": "builtins"}],
                    "type": "Port",
                    "module": "pytractions.base",
                },
            },
            "details": {
                "$data": {},
                "$type": {
                    "args": [
                        {"args": [], "type": "str", "module": "builtins"},
                        {"args": [], "type": "str", "module": "builtins"},
                    ],
                    "type": "TDict",
                    "module": "pytractions.base",
                },
            },
            "errors": {
                "$data": [],
                "$type": {
                    "args": [{"args": [], "type": "str", "module": "builtins"}],
                    "type": "TList",
                    "module": "pytractions.base",
                },
            },
            "o_out1": {
                "$data": {"data": 2.125},
                "$type": {
                    "args": [{"args": [], "type": "float", "module": "builtins"}],
                    "type": "Port",
                    "module": "pytractions.base",
                },
            },
            "skip": False,
            "skip_reason": "",
            "state": "finished",
            "stats": {
                "$data": {
                    "finished": "1990-01-01T00:00:08.00000Z",
                    "skipped": False,
                    "started": "1990-01-01T00:00:00.00000Z",
                },
                "$type": {"args": [], "module": "pytractions.traction", "type": "TractionStats"},
            },
            "t_ttest1": {
                "$data": {
                    "a_multiplier": {
                        "$data": {"data": 0.0},
                        "$type": {
                            "args": [{"args": [], "type": "float", "module": "builtins"}],
                            "type": "Port",
                            "module": "pytractions.base",
                        },
                    },
                    "details": {
                        "$data": {},
                        "$type": {
                            "args": [
                                {"args": [], "type": "str", "module": "builtins"},
                                {"args": [], "type": "str", "module": "builtins"},
                            ],
                            "type": "TDict",
                            "module": "pytractions.base",
                        },
                    },
                    "errors": {
                        "$data": [],
                        "$type": {
                            "args": [{"args": [], "type": "str", "module": "builtins"}],
                            "type": "TList",
                            "module": "pytractions.base",
                        },
                    },
                    "o_out1": {
                        "$data": {"data": 0.0},
                        "$type": {
                            "args": [{"args": [], "type": "float", "module": "builtins"}],
                            "type": "Port",
                            "module": "pytractions.base",
                        },
                    },
                    "skip": False,
                    "skip_reason": "",
                    "state": "ready",
                    "stats": {
                        "$data": {"finished": "", "skipped": False, "started": ""},
                        "$type": {
                            "args": [],
                            "module": "pytractions.traction",
                            "type": "TractionStats",
                        },
                    },
                    "uid": "1",
                },
                "$type": {
                    "args": [],
                    "module": "tests.test_base_traction",
                    "type": "test_tractor_to_json.<locals>.TTest1",
                },
            },
            "t_ttest2": {
                "$data": {
                    "a_reducer": {
                        "$data": {"data": 0.0},
                        "$type": {
                            "args": [{"args": [], "type": "float", "module": "builtins"}],
                            "type": "Port",
                            "module": "pytractions.base",
                        },
                    },
                    "details": {
                        "$data": {},
                        "$type": {
                            "args": [
                                {"args": [], "type": "str", "module": "builtins"},
                                {"args": [], "type": "str", "module": "builtins"},
                            ],
                            "type": "TDict",
                            "module": "pytractions.base",
                        },
                    },
                    "errors": {
                        "$data": [],
                        "$type": {
                            "args": [{"args": [], "type": "str", "module": "builtins"}],
                            "type": "TList",
                            "module": "pytractions.base",
                        },
                    },
                    "i_in1": {
                        "$data": {"data": 0.0},
                        "$type": {
                            "args": [
                                {
                                    "args": [],
                                    "module": "builtins",
                                    "type": "float",
                                }
                            ],
                            "module": "pytractions.base",
                            "type": "Port",
                        },
                    },
                    "o_out1": {
                        "$data": {"data": 0.0},
                        "$type": {
                            "args": [{"args": [], "type": "float", "module": "builtins"}],
                            "type": "Port",
                            "module": "pytractions.base",
                        },
                    },
                    "skip": False,
                    "skip_reason": "",
                    "state": "ready",
                    "stats": {
                        "$data": {"finished": "", "skipped": False, "started": ""},
                        "$type": {
                            "args": [],
                            "module": "pytractions.traction",
                            "type": "TractionStats",
                        },
                    },
                    "uid": "2",
                },
                "$type": {
                    "args": [],
                    "module": "tests.test_base_traction",
                    "type": "test_tractor_to_json.<locals>.TTest2",
                },
            },
            "t_ttest3": {
                "$data": {
                    "a_reducer": "TTest2[2]#a_reducer",
                    "details": {
                        "$data": {},
                        "$type": {
                            "args": [
                                {"args": [], "type": "str", "module": "builtins"},
                                {"args": [], "type": "str", "module": "builtins"},
                            ],
                            "type": "TDict",
                            "module": "pytractions.base",
                        },
                    },
                    "errors": {
                        "$data": [],
                        "$type": {
                            "args": [{"args": [], "type": "str", "module": "builtins"}],
                            "type": "TList",
                            "module": "pytractions.base",
                        },
                    },
                    "i_in1": {
                        "$data": {"data": 0.0},
                        "$type": {
                            "args": [
                                {
                                    "args": [],
                                    "module": "builtins",
                                    "type": "float",
                                }
                            ],
                            "module": "pytractions.base",
                            "type": "Port",
                        },
                    },
                    "o_out1": {
                        "$data": {"data": 0.0},
                        "$type": {
                            "args": [{"args": [], "type": "float", "module": "builtins"}],
                            "type": "Port",
                            "module": "pytractions.base",
                        },
                    },
                    "skip": False,
                    "skip_reason": "",
                    "state": "ready",
                    "stats": {
                        "$data": {"finished": "", "skipped": False, "started": ""},
                        "$type": {
                            "args": [],
                            "module": "pytractions.traction",
                            "type": "TractionStats",
                        },
                    },
                    "uid": "3",
                },
                "$type": {
                    "args": [],
                    "module": "tests.test_base_traction",
                    "type": "test_tractor_to_json.<locals>.TTest2",
                },
            },
            "t_ttest4": {
                "$data": {
                    "a_reducer": "TTest2[2]#a_reducer",
                    "details": {
                        "$data": {},
                        "$type": {
                            "args": [
                                {"args": [], "type": "str", "module": "builtins"},
                                {"args": [], "type": "str", "module": "builtins"},
                            ],
                            "type": "TDict",
                            "module": "pytractions.base",
                        },
                    },
                    "errors": {
                        "$data": [],
                        "$type": {
                            "args": [{"args": [], "type": "str", "module": "builtins"}],
                            "type": "TList",
                            "module": "pytractions.base",
                        },
                    },
                    "i_in1": {
                        "$data": {"data": 0.0},
                        "$type": {
                            "args": [
                                {
                                    "args": [],
                                    "module": "builtins",
                                    "type": "float",
                                }
                            ],
                            "module": "pytractions.base",
                            "type": "Port",
                        },
                    },
                    "o_out1": {
                        "$data": {"data": 0.0},
                        "$type": {
                            "args": [{"args": [], "type": "float", "module": "builtins"}],
                            "type": "Port",
                            "module": "pytractions.base",
                        },
                    },
                    "skip": False,
                    "skip_reason": "",
                    "state": "ready",
                    "stats": {
                        "$data": {"finished": "", "skipped": False, "started": ""},
                        "$type": {
                            "args": [],
                            "module": "pytractions.traction",
                            "type": "TractionStats",
                        },
                    },
                    "uid": "4",
                },
                "$type": {
                    "args": [],
                    "module": "tests.test_base_traction",
                    "type": "test_tractor_to_json.<locals>.TTest2",
                },
            },
            "tractions": {
                "$data": {
                    "t_ttest1": {
                        "$data": {
                            "a_multiplier": "TestTractor[tt1]#a_multiplier",
                            "details": {
                                "$data": {},
                                "$type": {
                                    "args": [
                                        {"args": [], "type": "str", "module": "builtins"},
                                        {"args": [], "type": "str", "module": "builtins"},
                                    ],
                                    "type": "TDict",
                                    "module": "pytractions.base",
                                },
                            },
                            "errors": {
                                "$data": [],
                                "$type": {
                                    "args": [{"args": [], "type": "str", "module": "builtins"}],
                                    "type": "TList",
                                    "module": "pytractions.base",
                                },
                            },
                            "o_out1": {
                                "$data": {"data": 10.0},
                                "$type": {
                                    "args": [{"args": [], "type": "float", "module": "builtins"}],
                                    "type": "Port",
                                    "module": "pytractions.base",
                                },
                            },
                            "skip": False,
                            "skip_reason": "",
                            "state": "finished",
                            "stats": {
                                "$data": {
                                    "finished": "1990-01-01T00:00:01.00000Z",
                                    "skipped": False,
                                    "started": "1990-01-01T00:00:00.00000Z",
                                },
                                "$type": {
                                    "args": [],
                                    "module": "pytractions.traction",
                                    "type": "TractionStats",
                                },
                            },
                            "uid": "tt1::1",
                        },
                        "$type": {
                            "args": [],
                            "module": "tests.test_base_traction",
                            "type": "test_tractor_to_json.<locals>.TTest1",
                        },
                    },
                    "t_ttest2": {
                        "$data": {
                            "a_reducer": "TestTractor[tt1]#a_reducer",
                            "details": {
                                "$data": {},
                                "$type": {
                                    "args": [
                                        {"args": [], "type": "str", "module": "builtins"},
                                        {"args": [], "type": "str", "module": "builtins"},
                                    ],
                                    "type": "TDict",
                                    "module": "pytractions.base",
                                },
                            },
                            "errors": {
                                "$data": [],
                                "$type": {
                                    "args": [{"args": [], "type": "str", "module": "builtins"}],
                                    "type": "TList",
                                    "module": "pytractions.base",
                                },
                            },
                            "o_out1": {
                                "$data": {"data": 5.5},
                                "$type": {
                                    "args": [{"args": [], "type": "float", "module": "builtins"}],
                                    "type": "Port",
                                    "module": "pytractions.base",
                                },
                            },
                            "i_in1": "TTest1[tt1::1]#o_out1",
                            "skip": False,
                            "skip_reason": "",
                            "state": "finished",
                            "stats": {
                                "$data": {
                                    "finished": "1990-01-01T00:00:03.00000Z",
                                    "skipped": False,
                                    "started": "1990-01-01T00:00:02.00000Z",
                                },
                                "$type": {
                                    "args": [],
                                    "module": "pytractions.traction",
                                    "type": "TractionStats",
                                },
                            },
                            "uid": "tt1::2",
                        },
                        "$type": {
                            "args": [],
                            "module": "tests.test_base_traction",
                            "type": "test_tractor_to_json.<locals>.TTest2",
                        },
                    },
                    "t_ttest3": {
                        "$data": {
                            "a_reducer": "TestTractor[tt1]#a_reducer",
                            "details": {
                                "$data": {},
                                "$type": {
                                    "args": [
                                        {"args": [], "type": "str", "module": "builtins"},
                                        {"args": [], "type": "str", "module": "builtins"},
                                    ],
                                    "type": "TDict",
                                    "module": "pytractions.base",
                                },
                            },
                            "errors": {
                                "$data": [],
                                "$type": {
                                    "args": [{"args": [], "type": "str", "module": "builtins"}],
                                    "type": "TList",
                                    "module": "pytractions.base",
                                },
                            },
                            "i_in1": "TTest2[tt1::2]#o_out1",
                            "o_out1": {
                                "$data": {"data": 3.25},
                                "$type": {
                                    "args": [{"args": [], "type": "float", "module": "builtins"}],
                                    "type": "Port",
                                    "module": "pytractions.base",
                                },
                            },
                            "skip": False,
                            "skip_reason": "",
                            "state": "finished",
                            "stats": {
                                "$data": {
                                    "finished": "1990-01-01T00:00:05.00000Z",
                                    "skipped": False,
                                    "started": "1990-01-01T00:00:04.00000Z",
                                },
                                "$type": {
                                    "args": [],
                                    "module": "pytractions.traction",
                                    "type": "TractionStats",
                                },
                            },
                            "uid": "tt1::3",
                        },
                        "$type": {
                            "args": [],
                            "module": "tests.test_base_traction",
                            "type": "test_tractor_to_json.<locals>.TTest2",
                        },
                    },
                    "t_ttest4": {
                        "$data": {
                            "a_reducer": "TestTractor[tt1]#a_reducer",
                            "details": {
                                "$data": {},
                                "$type": {
                                    "args": [
                                        {"args": [], "type": "str", "module": "builtins"},
                                        {"args": [], "type": "str", "module": "builtins"},
                                    ],
                                    "type": "TDict",
                                    "module": "pytractions.base",
                                },
                            },
                            "errors": {
                                "$data": [],
                                "$type": {
                                    "args": [{"args": [], "type": "str", "module": "builtins"}],
                                    "type": "TList",
                                    "module": "pytractions.base",
                                },
                            },
                            "i_in1": "TTest2[tt1::3]#o_out1",
                            "o_out1": {
                                "$data": {"data": 2.125},
                                "$type": {
                                    "args": [{"args": [], "type": "float", "module": "builtins"}],
                                    "type": "Port",
                                    "module": "pytractions.base",
                                },
                            },
                            "skip": False,
                            "skip_reason": "",
                            "state": "finished",
                            "stats": {
                                "$data": {
                                    "finished": "1990-01-01T00:00:07.00000Z",
                                    "skipped": False,
                                    "started": "1990-01-01T00:00:06.00000Z",
                                },
                                "$type": {
                                    "args": [],
                                    "module": "pytractions.traction",
                                    "type": "TractionStats",
                                },
                            },
                            "uid": "tt1::4",
                        },
                        "$type": {
                            "args": [],
                            "module": "tests.test_base_traction",
                            "type": "test_tractor_to_json.<locals>.TTest2",
                        },
                    },
                },
                "$type": {
                    "args": [
                        {"args": [], "type": "str", "module": "builtins"},
                        {"args": [], "type": "Traction", "module": "pytractions.traction"},
                    ],
                    "module": "pytractions.base",
                    "type": "TDict",
                },
            },
            "uid": "tt1",
        },
        "$type": {
            "args": [],
            "module": "tests.test_base_traction",
            "type": "test_tractor_to_json.<locals>.TestTractor",
        },
    }


def test_type_from_to_json():
    original = TypeNode.from_type(Port[TList[Port[int]]])
    assert TypeNode.from_json(TypeNode.from_type(Port[TList[Port[int]]]).to_json()) == original


def test_traction_simple_io():
    class TTest1(Traction):
        i_in: int
        o_out: int

        def _run(self) -> None:  # pragma: no cover
            self.o_out = self.i_in + 10

    t = TTest1(uid="1", i_in=10)
    t.run()
    assert t.o_out == 20
