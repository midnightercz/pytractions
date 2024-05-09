import pytest

from pytractions.base import (
    Base,
    Traction,
    Arg,
    In,
    Out,
    Res,
    OnUpdateCallable,
)
from pytractions.exc import TractionFailedError


class NOOPResource(Base):
    """NOOP Resource."""

    pass


def test_tractor_attr():

    # wrong doc attribute
    with pytest.raises(TypeError):

        class TestTraction1(Traction):
            i_input: In[int]
            d_: int

    # wrong doc attr attribute
    with pytest.raises(TypeError):

        class TestTraction2(Traction):
            i_input: In[int]
            d_i_input: int

    # custom attribute
    with pytest.raises(TypeError):

        class TestTraction3(Traction):
            custom_attribute: int


def test_to_json_from_json():
    class TestTraction(Traction):
        i_input: In[int]
        o_output: Out[int]
        r_res: Res[NOOPResource]
        a_arg: Arg[str]

        def _run(self, on_update: OnUpdateCallable) -> None:
            self.o_output.data = self.i_input.data

    t = TestTraction(
        uid="test-traction-1",
        i_input=In[int](data=1),
        a_arg=Arg[str](a="test"),
        r_res=Res[NOOPResource](r=NOOPResource()),
    )
    assert t.to_json() == {
        "$data": {
            "a_arg": {
                "$data": {
                    "a": "test",
                },
                "$type": {
                    "args": [
                        {
                            "args": [],
                            "module": "builtins",
                            "type": "str",
                        },
                    ],
                    "module": "pytractions.base",
                    "type": "Arg",
                },
            },
            "details": {
                "$data": {},
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
                            "type": "str",
                        },
                    ],
                    "module": "pytractions.base",
                    "type": "TDict",
                },
            },
            "errors": {
                "$data": [],
                "$type": {
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
            },
            "i_input": {
                "$data": {
                    "data": 1,
                },
                "$type": {
                    "args": [
                        {
                            "args": [],
                            "module": "builtins",
                            "type": "int",
                        },
                    ],
                    "module": "pytractions.base",
                    "type": "In",
                },
            },
            "o_output": {
                "$data": {
                    "data": 0,
                },
                "$type": {
                    "args": [
                        {
                            "args": [],
                            "module": "builtins",
                            "type": "int",
                        },
                    ],
                    "module": "pytractions.base",
                    "type": "Out",
                },
            },
            "r_res": {
                "$data": {
                    "r": {
                        "$data": {},
                        "$type": {
                            "args": [],
                            "module": "tests.test_traction",
                            "type": "NOOPResource",
                        },
                    },
                },
                "$type": {
                    "args": [
                        {
                            "args": [],
                            "module": "tests.test_traction",
                            "type": "NOOPResource",
                        },
                    ],
                    "module": "pytractions.base",
                    "type": "Res",
                },
            },
            "skip": False,
            "skip_reason": "",
            "state": "ready",
            "stats": {
                "$data": {
                    "finished": "",
                    "skipped": False,
                    "started": "",
                },
                "$type": {
                    "args": [],
                    "module": "pytractions.base",
                    "type": "TractionStats",
                },
            },
            "uid": "test-traction-1",
        },
        "$type": {
            "args": [],
            "module": "tests.test_traction",
            "type": "test_to_json_from_json.<locals>.TestTraction",
        },
    }
    t2 = TestTraction.from_json(t.to_json(), _locals=locals())
    assert t == t2


def test_to_run_failed():
    class TestTraction(Traction):
        i_input: In[int]
        o_output: Out[int]
        r_res: Res[NOOPResource]
        a_arg: Arg[str]

        def _run(self, on_update: OnUpdateCallable) -> None:
            self.o_output.data = self.i_input.data
            raise TractionFailedError

    t = TestTraction(
        uid="test-traction-1",
        i_input=In[int](data=1),
        a_arg=Arg[str](a="test"),
        r_res=Res[NOOPResource](r=NOOPResource()),
    )
    t.run()
    assert t.state == "failed"


def test_to_run_error():
    class TestTraction(Traction):
        i_input: In[int]
        o_output: Out[int]
        r_res: Res[NOOPResource]
        a_arg: Arg[str]

        def _run(self, on_update: OnUpdateCallable) -> None:
            self.o_output.data = self.i_input.data
            raise ValueError("test error")

    t = TestTraction(
        uid="test-traction-1",
        i_input=In[int](data=1),
        a_arg=Arg[str](a="test"),
        r_res=Res[NOOPResource](r=NOOPResource()),
    )

    with pytest.raises(ValueError):
        t.run()
    assert t.state == "error"
