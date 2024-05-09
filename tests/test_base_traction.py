from typing import Union, TypeVar
from dataclasses import field

import pytest

from pytractions.base import (
    Traction,
    TList,
    Out,
    In,
    Arg,
    Res,
    Base,
    NoData,
    TypeNode,
)


from pytractions.tractor import Tractor

T = TypeVar("T")


def test_traction_ok_args_1():
    class TTest(Traction):
        i_in1: In[int]
        o_out1: Out[int]
        r_res1: Res[int]
        a_arg1: Arg[int]


def test_traction_ok_args_description_1():
    class TTest(Traction):
        i_in1: In[int]
        o_out1: Out[int]
        r_res1: Res[int]
        a_arg1: Arg[int]
        d_i_in1: str = "Description of i_in1"


def test_traction_wrong_args_1():
    with pytest.raises(TypeError):

        class TTest(Traction):
            i_in1: Out[int]


def test_traction_wrong_args_2():
    with pytest.raises(TypeError):

        class TTest(Traction):
            o_out1: In[int]


def test_traction_wrong_args_3():
    with pytest.raises(TypeError):

        class TTest(Traction):
            a_arg1: In[int]


def test_traction_wrong_args_4():
    with pytest.raises(TypeError):

        class TTest(Traction):
            r_res1: In[int]


def test_traction_inputs_1():
    class TTest(Traction):
        i_in1: In[int] = In[int]()

    o: Out[int] = Out[int](data=10)
    TTest(uid="1", i_in1=o)


def test_traction_inputs_read():
    class TTest(Traction):
        i_in1: In[int] = field(default=In[int]())

    o: Out[int] = Out[int](data=10)
    t = TTest(uid="1", i_in1=o)
    assert id(t.i_in1.data) == id(o.data)


def test_traction_inputs_read_unset():
    class TTest(Traction):
        i_in1: In[int]

    t = TTest(uid="1")
    assert TypeNode.from_type(type(t.i_in1)) == TypeNode.from_type(NoData[int])


def test_traction_inputs_read_set():
    class TTest(Traction):
        i_in1: In[int]

    o: Out[int] = Out[int](data=10)
    t = TTest(uid="1", i_in1=o)
    assert TypeNode.from_type(type(t.i_in1)) == TypeNode.from_type(Out[int])
    assert t.i_in1.data == 10


def test_traction_to_json():
    class TTest(Traction):
        i_in1: In[int] = In[int]()
        d_i_in1: str = "description of i_in1"

    o: Out[int] = Out[int](data=10)
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
                    "type": "Out",
                    "module": "pytractions.base",
                },
            },
            "skip": False,
            "skip_reason": "",
            "state": "ready",
            "stats": {
                "$data": {"finished": "", "skipped": False, "started": ""},
                "$type": {"args": [], "type": "TractionStats", "module": "pytractions.base"},
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
        i_in1: In[TList[int]]

    print(TTest._fields)

    o: Out[TList[int]] = Out[TList[int]](data=TList[int]([10]))
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
                    "type": "Out",
                },
            },
            "skip": False,
            "skip_reason": "",
            "state": "ready",
            "stats": {
                "$data": {"finished": "", "skipped": False, "started": ""},
                "$type": {"args": [], "type": "TractionStats", "module": "pytractions.base"},
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
        i_in1: In[int]  # = In[int]()

    o: Out[int] = Out[int](data=10)
    t = TTest(uid="1", i_in1=o)
    t2 = TTest.from_json(t.to_json(), _locals=locals())
    assert t == t2


def test_traction_outputs_no_init():
    class TTest(Traction):
        o_out1: Out[int]

    t = TTest(uid="1")
    assert t.o_out1 == Out[int](data=int())


def test_traction_outputs_no_init_custom_default():
    class TTest(Traction):
        o_out1: Out[int] = Out[int](data=10)

    t = TTest(uid="1")
    assert t.o_out1 == Out[int](data=10)


def test_traction_outputs_uid():
    class TTest(Traction):
        o_out1: Out[int] = Out[int](data=10)

    t = TTest(uid="1")
    assert t.o_out1.uid == "TTest[1]::o_out1"


def test_traction_chain():
    class TTest1(Traction):
        o_out1: Out[int]

        def _run(self, on_update) -> None:  # pragma: no cover
            self.o_out1.data = 20

    class TTest2(Traction):
        i_in1: In[int]
        o_out1: Out[int]

        def _run(self, on_update) -> None:  # pragma: no cover
            self.o_out1.data = self.i_in1.data + 10

    t1 = TTest1(uid="1")
    t2 = TTest2(uid="1", i_in1=t1.o_out1)

    t1.run()
    t2.run()
    assert t2.o_out1.data == 30


def test_traction_chain_in_to_out():
    class TTest1(Traction):
        o_out1: Out[int]

        def _run(self, on_update) -> None:  # pragma: no cover
            self.o_out1.data = 20

    class TTest2(Traction):
        i_in1: In[int]
        o_out1: Out[int]

        def _run(self, on_update) -> None:  # pragma: no cover
            self.o_out1 = self.i_in1

    t1 = TTest1(uid="1")
    t2 = TTest2(uid="1", i_in1=t1.o_out1)

    t1.run()
    t2.run()
    assert t2.o_out1.data == 20
    t1.o_out1.data = 30

    assert t2.i_in1.data == 30


def test_traction_json(fixture_isodate_now):
    class TTest1(Traction):
        o_out1: Out[int]

        def _run(self, on_update) -> None:  # pragma: no cover
            self.o_out1.data = 20

    class TTest2(Traction):
        i_in1: In[Union[int, float]]
        o_out1: Out[int]

        def _run(self, on_update) -> None:  # pragma: no cover
            self.o_out1.data = self.i_in1.data + 10

    t1 = TTest1(uid="1")
    t2 = TTest2(uid="2", i_in1=t1.o_out1)

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
                    "type": "Out",
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
                "$type": {"args": [], "module": "pytractions.base", "type": "TractionStats"},
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
                    "type": "Out",
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
                "$type": {"args": [], "module": "pytractions.base", "type": "TractionStats"},
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
        o_out1: Out[float]
        a_multiplier: Arg[float]

        def _run(self, on_update) -> None:  # pragma: no cover
            self.o_out1.data = 20 * self.a_multiplier.a

    class TTest2(Traction):
        i_in1: In[float]
        o_out1: Out[float]
        a_reducer: Arg[float]

        def _run(self, on_update) -> None:  # pragma: no cover
            self.o_out1.data = (self.i_in1.data + 10) / float(self.a_reducer.a)

    class TestTractor(Tractor):
        a_multiplier: Arg[float] = Arg[float](a=0.0)
        a_reducer: Arg[float] = Arg[float](a=0.0)

        t_ttest1: TTest1 = TTest1(uid="1", a_multiplier=a_multiplier)
        t_ttest4: TTest2 = TTest2(uid="4", a_reducer=a_reducer)
        t_ttest3: TTest2 = TTest2(uid="3", a_reducer=a_reducer)
        t_ttest2: TTest2 = TTest2(uid="2", a_reducer=a_reducer)

        t_ttest2.i_in1 = t_ttest1.o_out1
        t_ttest3.i_in1 = t_ttest4.o_out1
        t_ttest4.i_in1 = t_ttest1.o_out1

        o_out1: Out[float] = t_ttest4.o_out1

    ttrac = TestTractor(uid="t1")

    tractions = []
    for k, v in ttrac.__dict__.items():
        if k.startswith("t_"):
            tractions.append(k)

    assert tractions == ["t_ttest1", "t_ttest4", "t_ttest3", "t_ttest2"]


def test_tractor_members_invalid_order() -> None:
    class TTest1(Traction):
        o_out1: Out[float]
        a_multiplier: Arg[float]

        def _run(self, on_update) -> None:  # pragma: no cover
            self.o_out1.data = 20 * self.a_multiplier.a

    class TTest2(Traction):
        i_in1: In[float]
        o_out1: Out[float]
        a_reducer: Arg[float]

        def _run(self, on_update) -> None:  # pragma: no cover
            self.o_out1.data = (self.i_in1.data + 10) / float(self.a_reducer.a)

    with pytest.raises(ValueError):

        class TestTractor(Tractor):
            a_multiplier: Arg[float] = Arg[float](a=0.0)
            a_reducer: Arg[float] = Arg[float](a=0.0)

            t_ttest1: TTest1 = TTest1(uid="1", a_multiplier=a_multiplier)
            t_ttest2: TTest2 = TTest2(uid="4", a_reducer=a_reducer)
            t_ttest4: TTest2 = TTest2(uid="3", a_reducer=a_reducer)
            t_ttest3: TTest2 = TTest2(uid="2", a_reducer=a_reducer)

            t_ttest2.i_in1 = t_ttest1.o_out1
            t_ttest4.i_in1 = t_ttest3.o_out1

            o_out1: Out[float] = t_ttest4.o_out1


def test_tractor_run() -> None:
    class TTest2(Traction):
        i_in1: In[float]
        o_out1: Out[float]
        a_reducer: Arg[float]

        def _run(self, on_update) -> None:  # pragma: no cover
            self.o_out1.data = (self.i_in1.data + 1) / float(self.a_reducer.a)

    class TestTractor(Tractor):
        a_multiplier: Arg[float] = Arg[float](a=0.0)
        a_reducer: Arg[float] = Arg[float](a=0.0)

        i_in1: In[float] = In[float](data=1.0)

        t_ttest1: TTest2 = TTest2(uid="1", a_reducer=a_reducer)
        t_ttest2: TTest2 = TTest2(uid="2", a_reducer=a_reducer)
        t_ttest3: TTest2 = TTest2(uid="3", a_reducer=a_reducer)
        t_ttest4: TTest2 = TTest2(uid="4", a_reducer=a_reducer)

        t_ttest1.i_in1 = i_in1
        t_ttest2.i_in1 = t_ttest1.o_out1
        t_ttest3.i_in1 = t_ttest2.o_out1
        t_ttest4.i_in1 = t_ttest3.o_out1

        o_out1: Out[float] = t_ttest4.o_out1

    tt = TestTractor(
        uid="tt1",
        a_multiplier=Arg[float](a=10.0),
        a_reducer=Arg[float](a=2.0),
        i_in1=In[float](data=10.0),
    )
    tt.run()
    assert tt.o_out1.data == 1.5625


class UselessResource(Base):
    """Testing resource."""

    values_stack: TList[int]

    def get_some_value(self) -> int:
        """Get first value from the init list."""
        return self.values_stack.pop(0)


def test_tractor_run_resources() -> None:

    class TTest2(Traction):
        i_in1: In[float]
        o_out1: Out[float]
        a_reducer: Arg[float]
        r_useless_res: Res[UselessResource]

        def _run(self, on_update) -> None:  # pragma: no cover
            useless_value = self.r_useless_res.r.get_some_value()
            self.o_out1.data = (self.i_in1.data + useless_value) / float(self.a_reducer.a)

    class TestTractor(Tractor):
        a_multiplier: Arg[float] = Arg[float](a=0.0)
        a_reducer: Arg[float] = Arg[float](a=0.0)

        r_useless: Res[UselessResource] = Res[UselessResource](
            r=UselessResource(values_stack=TList[int](TList[int]([])))
        )

        i_in1: In[float] = In[float](data=1.0)

        t_ttest1: TTest2 = TTest2(uid="1", a_reducer=a_reducer, r_useless_res=r_useless)
        t_ttest2: TTest2 = TTest2(uid="2", a_reducer=a_reducer, r_useless_res=r_useless)
        t_ttest3: TTest2 = TTest2(uid="3", a_reducer=a_reducer, r_useless_res=r_useless)
        t_ttest4: TTest2 = TTest2(uid="4", a_reducer=a_reducer, r_useless_res=r_useless)

        t_ttest1.i_in1 = i_in1
        t_ttest2.i_in1 = t_ttest1.o_out1
        t_ttest3.i_in1 = t_ttest2.o_out1
        t_ttest4.i_in1 = t_ttest3.o_out1

        o_out1: Out[float] = t_ttest4.o_out1

    tt = TestTractor(
        uid="tt1",
        a_multiplier=Arg[float](a=10.0),
        a_reducer=Arg[float](a=2.0),
        i_in1=In[float](data=10.0),
        r_useless=Res[UselessResource](r=UselessResource(values_stack=TList[int]([2, 5, 1, 7, 2]))),
    )

    tt.run()
    assert tt.o_out1.data == 5.125


def test_tractor_to_json(fixture_isodate_now) -> None:
    class TTest1(Traction):
        o_out1: Out[float]
        a_multiplier: Arg[float]

        def _run(self, on_update) -> None:  # pragma: no cover
            self.o_out1.data = 1 * self.a_multiplier.a

    class TTest2(Traction):
        i_in1: In[float]
        o_out1: Out[float]
        a_reducer: Arg[float]

        def _run(self, on_update) -> None:  # pragma: no cover
            self.o_out1.data = (self.i_in1.data + 1) / float(self.a_reducer.a)

    class TestTractor(Tractor):
        a_multiplier: Arg[float] = Arg[float](a=0.0)
        a_reducer: Arg[float] = Arg[float](a=0.0)

        t_ttest1: TTest1 = TTest1(uid="1", a_multiplier=a_multiplier)
        t_ttest2: TTest2 = TTest2(uid="2", a_reducer=a_reducer)
        t_ttest3: TTest2 = TTest2(uid="3", a_reducer=a_reducer)
        t_ttest4: TTest2 = TTest2(uid="4", a_reducer=a_reducer)

        t_ttest2.i_in1 = t_ttest1.o_out1
        t_ttest3.i_in1 = t_ttest2.o_out1
        t_ttest4.i_in1 = t_ttest3.o_out1

        o_out1: Out[float] = t_ttest4.o_out1

    tt = TestTractor(uid="tt1", a_multiplier=Arg[float](a=10.0), a_reducer=Arg[float](a=2.0))

    tt.run()
    assert tt.to_json() == {
        "$data": {
            "a_multiplier": {
                "$data": {"a": 10.0},
                "$type": {
                    "args": [{"args": [], "type": "float", "module": "builtins"}],
                    "type": "Arg",
                    "module": "pytractions.base",
                },
            },
            "a_reducer": {
                "$data": {"a": 2.0},
                "$type": {
                    "args": [{"args": [], "type": "float", "module": "builtins"}],
                    "type": "Arg",
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
                    "type": "Out",
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
                "$type": {"args": [], "module": "pytractions.base", "type": "TractionStats"},
            },
            "t_ttest1": {
                "$data": {
                    "a_multiplier": {
                        "$data": {"a": 0.0},
                        "$type": {
                            "args": [{"args": [], "type": "float", "module": "builtins"}],
                            "type": "Arg",
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
                            "type": "Out",
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
                            "module": "pytractions.base",
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
                        "$data": {"a": 0.0},
                        "$type": {
                            "args": [{"args": [], "type": "float", "module": "builtins"}],
                            "type": "Arg",
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
                    "i_in1": "TTest1[1]#o_out1",
                    "o_out1": {
                        "$data": {"data": 0.0},
                        "$type": {
                            "args": [{"args": [], "type": "float", "module": "builtins"}],
                            "type": "Out",
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
                            "module": "pytractions.base",
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
                    "a_reducer": {
                        "$data": {"a": 0.0},
                        "$type": {
                            "args": [{"args": [], "type": "float", "module": "builtins"}],
                            "type": "Arg",
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
                    "i_in1": "TTest2[2]#o_out1",
                    "o_out1": {
                        "$data": {"data": 0.0},
                        "$type": {
                            "args": [{"args": [], "type": "float", "module": "builtins"}],
                            "type": "Out",
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
                            "module": "pytractions.base",
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
                    "a_reducer": {
                        "$data": {"a": 0.0},
                        "$type": {
                            "args": [{"args": [], "type": "float", "module": "builtins"}],
                            "type": "Arg",
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
                    "i_in1": "TTest2[3]#o_out1",
                    "o_out1": {
                        "$data": {"data": 0.0},
                        "$type": {
                            "args": [{"args": [], "type": "float", "module": "builtins"}],
                            "type": "Out",
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
                            "module": "pytractions.base",
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
                            "a_multiplier": {
                                "$data": {"a": 10.0},
                                "$type": {
                                    "args": [{"args": [], "type": "float", "module": "builtins"}],
                                    "module": "pytractions.base",
                                    "type": "Arg",
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
                                "$data": {"data": 10.0},
                                "$type": {
                                    "args": [{"args": [], "type": "float", "module": "builtins"}],
                                    "type": "Out",
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
                                    "module": "pytractions.base",
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
                            "a_reducer": {
                                "$data": {"a": 2.0},
                                "$type": {
                                    "args": [{"args": [], "type": "float", "module": "builtins"}],
                                    "module": "pytractions.base",
                                    "type": "Arg",
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
                                "$data": {"data": 5.5},
                                "$type": {
                                    "args": [{"args": [], "type": "float", "module": "builtins"}],
                                    "type": "Out",
                                    "module": "pytractions.base",
                                },
                            },
                            "i_in1": {
                                "$data": {"data": 10.0},
                                "$type": {
                                    "args": [{"args": [], "type": "float", "module": "builtins"}],
                                    "type": "Out",
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
                                "$type": {
                                    "args": [],
                                    "module": "pytractions.base",
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
                            "a_reducer": {
                                "$data": {"a": 2.0},
                                "$type": {
                                    "args": [{"args": [], "type": "float", "module": "builtins"}],
                                    "module": "pytractions.base",
                                    "type": "Arg",
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
                                "$data": {"data": 5.5},
                                "$type": {
                                    "args": [{"args": [], "type": "float", "module": "builtins"}],
                                    "type": "Out",
                                    "module": "pytractions.base",
                                },
                            },
                            "o_out1": {
                                "$data": {"data": 3.25},
                                "$type": {
                                    "args": [{"args": [], "type": "float", "module": "builtins"}],
                                    "type": "Out",
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
                                    "module": "pytractions.base",
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
                            "a_reducer": {
                                "$data": {"a": 2.0},
                                "$type": {
                                    "args": [{"args": [], "type": "float", "module": "builtins"}],
                                    "module": "pytractions.base",
                                    "type": "Arg",
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
                                "$data": {"data": 3.25},
                                "$type": {
                                    "args": [{"args": [], "type": "float", "module": "builtins"}],
                                    "type": "Out",
                                    "module": "pytractions.base",
                                },
                            },
                            "o_out1": {
                                "$data": {"data": 2.125},
                                "$type": {
                                    "args": [{"args": [], "type": "float", "module": "builtins"}],
                                    "type": "Out",
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
                                    "module": "pytractions.base",
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
                        {"args": [], "type": "Traction", "module": "pytractions.base"},
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
    original = TypeNode.from_type(Out[TList[Out[int]]])
    assert TypeNode.from_json(TypeNode.from_type(Out[TList[Out[int]]]).to_json()) == original


def test_type_in_out():
    assert TypeNode.from_type(Out[int]) == TypeNode.from_type(In[int])


def test_type_in_out_complex():
    assert TypeNode.from_type(Out[TList[Out[int]]]) == TypeNode.from_type(In[TList[In[int]]])


def test_type_in_out_complex_2():
    print(TypeNode.from_json(TypeNode.from_type(Out[TList[Out[int]]]).to_json()).to_json())
    assert TypeNode.from_json(
        TypeNode.from_type(Out[TList[Out[int]]]).to_json()
    ) == TypeNode.from_type(In[TList[In[int]]])
