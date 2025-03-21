import pytest

from pytractions.base import (
    Base,
    Port,
)
from pytractions.traction import Traction
from pytractions.exc import TractionFailedError


class NOOPResource(Base):
    """NOOP Resource."""

    pass


def test_tractor_attr():

    # wrong doc attribute
    with pytest.raises(TypeError):

        class TestTraction1(Traction):
            i_input: Port[int]
            d_: int

    # wrong doc attr attribute
    with pytest.raises(TypeError):

        class TestTraction2(Traction):
            i_input: Port[int]
            d_i_input: int

    # custom attribute
    with pytest.raises(TypeError):

        class TestTraction3(Traction):
            custom_attribute: int


def test_to_json_from_json():
    class TestTraction(Traction):
        i_input: Port[int]
        o_output: Port[int]
        r_res: Port[NOOPResource]
        a_arg: Port[str]

        def _run(self) -> None:
            self.o_output.data = self.i_input.data

    t = TestTraction(
        uid="test-traction-1",
        i_input=Port[int](data=1),
        a_arg=Port[str](data="test"),
        r_res=Port[NOOPResource](data=NOOPResource()),
    )
    assert t.to_json() == {
        "$data": {
            "a_arg": {
                "$data": {
                    "data": "test",
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
                    "type": "Port",
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
                    "type": "Port",
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
                    "type": "Port",
                },
            },
            "r_res": {
                "$data": {
                    "data": {
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
                    "type": "Port",
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
                    "module": "pytractions.traction",
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
        i_input: Port[int]
        o_output: Port[int]
        r_res: Port[NOOPResource]
        a_arg: Port[str]

        def _run(self) -> None:
            self.o_output = self.i_input
            raise TractionFailedError

    t = TestTraction(
        uid="test-traction-1",
        i_input=Port[int](data=1),
        a_arg=Port[str](data="test"),
        r_res=Port[NOOPResource](data=NOOPResource()),
    )
    t.run()
    assert t.state == "failed"


def test_to_run_error():
    class TestTraction(Traction):
        i_input: Port[int]
        o_output: Port[int]
        r_res: Port[NOOPResource]
        a_arg: Port[str]

        def _run(self) -> None:
            self.o_output = self.i_input
            raise ValueError("test error")

    t = TestTraction(
        uid="test-traction-1",
        i_input=Port[int](data=1),
        a_arg=Port[str](data="test"),
        r_res=Port[NOOPResource](data=NOOPResource()),
    )

    with pytest.raises(ValueError):
        t.run()
    assert t.state == "error"


def test_traction_log():
    class TestTraction(Traction):
        i_input: Port[int]
        o_output: Port[int]
        r_res: Port[NOOPResource]
        a_arg: Port[str]

        def _run(self) -> None:
            self.o_output = self.i_input
            self.log("test log")
