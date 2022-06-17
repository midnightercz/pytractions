from typing import Generator, Dict, Any, Callable, TypeVar, Generic, TypedDict, ClassVar, Type
from unittest import mock

import pydantic
import pytest


from pytraction.traction import (
    Step, StepResults, ArgsTypeCls, NoInputs, StepInputs, NoResources,
    ExtResourcesCls, StepOnUpdateCallable, StepErrors, StepDetails,
    NoArgs, SharedResults,
    StepFailedError, Tractor, Secret,
    StepOnErrorCallable, StepOnUpdateCallable,
    TractorDumpDict, StepResults, NoDetails, StepState)


class TResults(StepResults):
    x: int = 10


class TArgs(ArgsTypeCls):
    arg1: int


class TSecretArgs(ArgsTypeCls):
    arg1: Secret


class TResources(ExtResourcesCls):
    service1: int


class TInputs(StepInputs):
    input1: TResults


class TDetails(StepDetails):
    status: str = ""


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
    class TStep(Step[TResults, TArgs, TResources, NoInputs, NoDetails]):
        pass
    
    with pytest.raises(TypeError) as exc:
        step = TStep("test-step-1", fixture_shared_results)
    assert str(exc.value) == "Can't instantiate abstract class TStep with abstract method _run"


def test_step_initiation_succesful_no_args_no_inputs(fixture_shared_results):
    class TStep(Step[TResults, NoArgs, TResources, NoInputs, NoDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
    
    step = TStep("test-step-1", NoArgs(), fixture_shared_results)
    assert step.inputs == NoInputs()
    assert step.state == StepState.READY
    assert step.skip == False
    assert step.skip_reason == ""
    assert step.stats == {
        "started": None,
        "finished": None,
        "skip": False,
        "skip_reason": "",
        "skipped": False,
        "state": StepState.READY
    }
    assert step.errors == StepErrors()


def test_step_initiation_succesful_no_args(fixture_shared_results):
    class TStep(Step[TResults, NoArgs, TResources, TInputs, NoDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
 
    step = TStep("test-step-1", NoArgs(), fixture_shared_results,  TResources(service1=1), TInputs(input1=TResults()))
    assert step.inputs.input1.x == 10


def test_step_initiation_succesful_no_inputs(fixture_shared_results):
    class TStep(Step[TResults, TArgs, TResources, NoInputs, NoDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
    
    step = TStep("test-step-1", TArgs(arg1=10), fixture_shared_results)
    assert step.args.arg1 == 10


def test_step_initiation_wrong_arg_type(fixture_shared_results):
    """Step expects NoArgs but TArgs are given."""
    class TStep(Step[TResults, NoArgs, TResources, NoInputs, NoDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
    
    with pytest.raises(TypeError):
        step = TStep("test-step-1", TArgs(arg1=10), fixture_shared_results)


def test_step_initiation_wrong_inputs_type(fixture_shared_results):
    """Step expects NoInputs but TInputs are given."""

    class TStep(Step[TResults, NoArgs, TResources, NoInputs, NoDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
    
    with pytest.raises(TypeError) as exc:
        step = TStep("test-step-1", NoArgs(), fixture_shared_results, TResources(service1=1), TInputs(input1=TResults(x=1)))
    assert str(exc.value).startswith("Step inputs are not type of <class 'pytraction.traction.NoInputs'>")


def test_step_initiation_wrong_external_resources(fixture_shared_results):
    """Step expects NoInputs but TInputs are given."""

    class TStep(Step[TResults, NoArgs, NoResources, NoInputs, NoDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
    
    with pytest.raises(TypeError):
        step = TStep("test-step-1", NoArgs(), fixture_shared_results, NoInputs())


def test_step_initiation_missing_arguments(fixture_shared_results):
    """Step initiation is missing shared_reults."""

    class TStep(Step[TResults, NoArgs, NoResources, NoInputs, NoDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
    
    with pytest.raises(pydantic.ValidationError):
        step = TStep("test-step-1", NoArgs(), NoInputs())


def test_step_run_results(fixture_shared_results):
    """Step initiation is missing shared_reults."""
    class TStep(Step[TResults, NoArgs, NoResources, NoInputs, NoDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results.x = 10
            
    step = TStep("test-step-1", NoArgs(), fixture_shared_results, NoResources(), NoInputs())
    step.run()
    assert step.results.x == 10


def test_step_run_details(fixture_shared_results):
    """Step initiation is missing shared_reults."""
    class TStep(Step[TResults, NoArgs, NoResources, NoInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.details.status = "ok"
            self.results.x = 10
            
    step = TStep("test-step-1", NoArgs(), fixture_shared_results, NoResources(), NoInputs())
    step.run()
    assert step.results.x == 10
    assert step.details.status == 'ok'


def test_step_run_status_update(fixture_shared_results):
    """Step initiation is missing shared_reults."""

    states_collected = []

    def state_collect(step):
        states_collected.append(step.state)

    class TStep(Step[TResults, NoArgs, NoResources, NoInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.details.status = "ok"
            on_update(self)
            self.results.x = 10
            
    step = TStep("test-step-1", NoArgs(), fixture_shared_results, NoResources(), NoInputs())
    step.run(on_update=state_collect)
    assert step.results.x == 10
    assert step.details.status == 'ok'
    assert states_collected == [StepState.PREP, StepState.RUNNING, StepState.RUNNING, StepState.FINISHED]


def test_step_run_secret_arg(fixture_shared_results):
    """Step initiation is missing shared_reults."""

    class TStep(Step[TResults, TSecretArgs, NoResources, NoInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results.x = self.args.arg1

    step = TStep("test-step-1", TSecretArgs(arg1=Secret("supersecret")), fixture_shared_results, NoResources(), NoInputs())
    step.run()
            
    assert step.args.arg1 == 'supersecret'


def test_step_dict(fixture_shared_results):
    """Step initiation is missing shared_reults."""

    class TStep(Step[TResults, TSecretArgs, NoResources, NoInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results.x = self.args.arg1

    step = TStep("test-step-1", TSecretArgs(arg1=Secret("supersecret")), fixture_shared_results, NoResources(), NoInputs())
    assert step.dict()['args']['arg1'] == "*CENSORED*"


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


def test_results_no_default():
    with pytest.raises(TypeError) as exc:
        class TDetails(StepDetails):
            x: int
    assert str(exc.value) == "Attribute x is missing default value"


def test_results_default():
    class TDetails(StepDetails):
        x: int = 10

    dets = TDetails()
    assert dets.x == 10
