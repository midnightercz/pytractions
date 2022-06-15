from typing import Generator, Dict, Any, Callable, TypeVar, Generic, TypedDict, ClassVar, Type
from unittest import mock

import pydantic
import pytest


from pytraction.traction import (
    Step, StepResults, ArgsTypeCls, NoInputs, StepInputs, NoResources,
    ExtResourcesCls, StepOnUpdateCallable,
    NoArgs, SharedResults,
    StepFailedError, Tractor, Secret,
    StepOnErrorCallable, StepOnUpdateCallable,
    TractorDumpDict, StepResults)


class TResults(StepResults):
    x: int = 10


class TArgs(ArgsTypeCls):
    arg1: int


class TResources(ExtResourcesCls):
    service1: int

class TInputs(StepInputs):
    input1: TResults


@pytest.fixture
def fixture_shared_results():
    yield SharedResults(results={})

def test_step_initiation_no_generic(fixture_shared_results):
    class TStep(Step):
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
         
    with pytest.raises(TypeError) as exc:
        TStep("test-step-1", {}, fixture_shared_results)

    print(exc)
    assert str(exc.value).startswith("Missing generic annotations for Step class")


def test_step_initiation_no_run_method(fixture_shared_results):
    class TStep(Step[TResults, TArgs, TResources, NoInputs]):
        pass
    
    with pytest.raises(TypeError) as exc:
        step = TStep("test-step-1", fixture_shared_results)
    assert str(exc.value) == "Can't instantiate abstract class TStep with abstract method _run"


def test_step_initiation_succesful_no_args_no_inputs(fixture_shared_results):
    class TStep(Step[TResults, NoArgs, TResources, NoInputs]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
    
    step = TStep("test-step-1", NoArgs(), fixture_shared_results)
    assert step.inputs == NoInputs()

def test_step_initiation_succesful_no_args(fixture_shared_results):
    class TStep(Step[TResults, NoArgs, TResources, TInputs]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
 
    step = TStep("test-step-1", NoArgs(), fixture_shared_results,  TResources(service1=1), TInputs(input1=TResults()))
    assert step.inputs.input1.x == 10


def test_step_initiation_succesful_no_inputs(fixture_shared_results):
    class TStep(Step[TResults, TArgs, TResources, NoInputs]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
    
    step = TStep("test-step-1", TArgs(arg1=10), fixture_shared_results)
    assert step.args.arg1 == 10


def test_step_initiation_wrong_arg_type(fixture_shared_results):
    """Step expects NoArgs but TArgs are given."""
    class TStep(Step[TResults, NoArgs, TResources, NoInputs]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
    
    with pytest.raises(TypeError):
        step = TStep("test-step-1", TArgs(arg1=10), fixture_shared_results)

def test_step_initiation_wrong_inputs_type(fixture_shared_results):
    """Step expects NoInputs but TInputs are given."""

    class TStep(Step[TResults, NoArgs, TResources, NoInputs]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
    
    with pytest.raises(TypeError) as exc:
        step = TStep("test-step-1", NoArgs(), fixture_shared_results, TResources(service1=1), TInputs(input1=TResults(x=1)))
    assert str(exc.value).startswith("Step inputs are not type of <class 'pytraction.traction.NoInputs'>")

def test_step_initiation_wrong_external_resources(fixture_shared_results):
    """Step expects NoInputs but TInputs are given."""
    class TInputs(StepInputs):
        input1: TResults

    class TStep(Step[TResults, NoArgs, NoResources, NoInputs]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
    
    with pytest.raises(TypeError):
        step = TStep("test-step-1", NoArgs(), fixture_shared_results, NoInputs())

def test_step_initiation_missing_arguments(fixture_shared_results):
    """Step initiation is missing shared_reults."""
    class TInputs(StepInputs):
        input1: TResults

    class TStep(Step[TResults, NoArgs, NoResources, NoInputs]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
    
    with pytest.raises(pydantic.ValidationError):
        step = TStep("test-step-1", NoArgs(), NoInputs())

def test_inputs_invalid_field_type():
    class TInputs(StepInputs):
        input1: int
    with pytest.raises(pydantic.ValidationError):
        TInputs(input1=1)


def test_results_no_default():
    with pytest.raises(TypeError) as exc:
        class TResults(StepResults):
            x: int
    assert str(exc.value) == "Attribute x is missing default value"

def test_results_default():
    class TResults(StepResults):
        x: int = 10

    res = TResults()
    assert res.x == 10
