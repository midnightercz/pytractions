from typing import List, TypeVar, Generic, TypedDict, ClassVar, Type, Optional, get_args
from unittest import mock

import pydantic
import pytest

import pytraction

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
    arg2: str


class TResources(ExtResourcesCls):
    service1: int


class TInputs(StepInputs):
    input1: TResults


class TDetails(StepDetails):
    status: str = ""


@pytest.fixture
def fixture_shared_results():
    yield SharedResults(results={})


@pytest.fixture
def fixture_isodate_now():
    with mock.patch("pytraction.traction.isodate_now") as mocked:
        mocked.side_effect = (
            '1990-01-01T00:00:00.00000Z',
            '1990-01-01T00:00:01.00000Z',
            '1990-01-01T00:00:02.00000Z',
            '1990-01-01T00:00:03.00000Z',
            '1990-01-01T00:00:04.00000Z',
            '1990-01-01T00:00:05.00000Z',
            '1990-01-01T00:00:06.00000Z',
            '1990-01-01T00:00:07.00000Z',
            '1990-01-01T00:00:08.00000Z',
            '1990-01-01T00:00:09.00000Z',
        )
        yield mocked



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
    """Step run results test."""
    class TStep(Step[TResults, NoArgs, NoResources, NoInputs, NoDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results.x = 10
            
    step = TStep("test-step-1", NoArgs(), fixture_shared_results, NoResources(), NoInputs())
    step.run()
    assert step.results.x == 10


def test_step_run_details(fixture_shared_results):
    """Step run with details."""
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
    """Step run update status test."""

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
    """Step run with secret args."""
    
    class TTResults(StepResults):
        x: str = 10


    class TStep(Step[TTResults, TSecretArgs, NoResources, NoInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results.x = str(self.args.arg1)

    step = TStep("test-step-1", TSecretArgs(arg1=Secret("supersecret"), arg2='test-arg'), fixture_shared_results, NoResources(), NoInputs())
    step.run()
            
    assert step.args.arg1 == 'supersecret'


def test_step_run_invalid_state(fixture_shared_results):
    """Step run in invalid state."""

    class TStep(Step[TResults, TArgs, NoResources, NoInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results.x = self.args.arg1

    step = TStep("test-step-1", TArgs(arg1=1), fixture_shared_results, NoResources(), NoInputs())
    step.state = StepState.RUNNING
    step.run()
    assert step.results.x == 10 # step hasn't run, so result should be default value


def test_step_run_failed(fixture_shared_results):
    """Step initiation is missing shared_reults."""

    class TStep(Step[TResults, TArgs, NoResources, NoInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            raise StepFailedError("step run failed")

    step = TStep("test-step-1", TArgs(arg1=1), fixture_shared_results, NoResources(), NoInputs())
    step.run()
    assert step.state == StepState.FAILED


def test_step_run_error(fixture_shared_results):
    """Step initiation is missing shared_reults."""

    class TStep(Step[TResults, TArgs, NoResources, NoInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            raise ValueError("unexpected error")

    step = TStep("test-step-1", TArgs(arg1=1), fixture_shared_results, NoResources(), NoInputs())
    with pytest.raises(ValueError):
        step.run()
    assert step.state == StepState.ERROR


def test_step_dump_load(fixture_shared_results, fixture_isodate_now):
    """Step run with secret args."""

    class TStep(Step[TResults, TSecretArgs, NoResources, TInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results.x = 1
            self.details.status = 'done'
            print("step run")


    standalone_input = TResults(x=55)

    step = TStep("test-step-1",
                 TSecretArgs(arg1=Secret("supersecret"), arg2='test-arg'),
                 fixture_shared_results,
                 NoResources(),
                 TInputs(input1=standalone_input))
    step.run()

    assert step.state == StepState.FINISHED
            
    dumped = step.dump()
    assert dumped == {
        'args': {'arg1': '*CENSORED*', 'arg2': 'test-arg'},
        "details": {'status':'done'},
        'errors': {'errors': {}},
        'inputs': {},
        'inputs_standalone': {"input1":{"x":55}},
        'skip': False,
        'skip_reason': '',
        'state': StepState.FINISHED,
        'stats': {
            'skip': False,
            'skipped': False,
            'skip_reason': '',
            'state': StepState.FINISHED,
            'started': '1990-01-01T00:00:00.00000Z',
            'finished': '1990-01-01T00:00:01.00000Z'
        },
        'uid': 'test-step-1',
        'type': step.NAME,
        'results': {'step': None, 'x':1}
    }
    step2 = TStep("test-step-1", TSecretArgs(arg1=Secret("supersecret"), arg2='test-arg'), fixture_shared_results, NoResources(), TInputs(input1=standalone_input))
    step2.load(dumped)
    assert step2.args.arg1 == "supersecret"
    assert step2.args.arg2 == 'test-arg'
    assert step2.results.x == 1
    assert step2.skip == False
    assert step2.skip_reason == ''
    assert step2.state == StepState.FINISHED
    assert step2.stats == {
            'skip': False,
            'skipped': False,
            'skip_reason': '',
            'state': StepState.FINISHED,
            'started': '1990-01-01T00:00:00.00000Z',
            'finished': '1990-01-01T00:00:01.00000Z'
    } 
    assert step2.uid == 'test-step-1'

    step3 = TStep.load_cls(dumped, {'arg1': Secret('supersecret')},  {}, fixture_shared_results, NoResources())
    assert step2.uid == step3.uid
    assert step2.state == step3.state
    assert step2.skip == step3.skip
    assert step2.skip_reason == step3.skip_reason
    assert step2.results == step3.results
    assert step2.errors == step3.errors
    assert step2.details == step3.details
    assert step2.stats == step3.stats
    assert step2.inputs == step3.inputs
    assert step2.args == step3.args



def test_step_dict(fixture_shared_results):
    """Step initiation is missing shared_reults."""

    class TStep(Step[TResults, TSecretArgs, NoResources, NoInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results.x = self.args.arg1

    step = TStep("test-step-1", TSecretArgs(arg1=Secret("supersecret"), arg2='test-arg'), fixture_shared_results, NoResources(), NoInputs())
    assert step.dict()['args']['arg1'] == "*CENSORED*"
    assert step.dict()['args']['arg2'] == "test-arg"


def test_step_generic(fixture_shared_results):
    """Step initiation is missing shared_reults."""

    LoaderModel = TypeVar("LoaderModel")

    class Model1(pydantic.BaseModel):
        attribute1: str = 'attr1'

    class GTResults(StepResults, Generic[LoaderModel]):
        models: List[LoaderModel] = []

    class TGenericLoader(Step[GTResults[LoaderModel], TSecretArgs, NoResources, NoInputs, TDetails], Generic[LoaderModel]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            generic_args = get_args(self.__orig_bases__[0])
            modelcls = generic_args[0]
            self.results.models.append(modelcls())

    class Model1Loader(TGenericLoader[Model1]):
        pass


    step = Model1Loader("test-step-1", TSecretArgs(arg1=Secret("supersecret"), arg2='test-arg'), fixture_shared_results, NoResources(), NoInputs())
    step.run()
    assert step.dict()['args']['arg1'] == "*CENSORED*"
    assert step.dict()['args']['arg2'] == "test-arg"


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


def test_results_invalid_type():
    class TResults(StepResults):
        x: int = 10
    with pytest.raises(pydantic.ValidationError):
        res = TResults(x="a")


def test_invalid_secret_arguments():
    with pytest.raises(pydantic.ValidationError):
        assert TSecretArgs(arg1="1", arg2='a2')


def test_invalid_secret_arguments_compare():
    sec2 = TSecretArgs(arg1=Secret("a"), arg2='a2')
    sec = TSecretArgs(arg1=Secret("a"), arg2='a2')
    assert sec.arg1 == "a"
    assert sec.arg1 == sec2.arg1


def test_secret_str():
    sec = TSecretArgs(arg1=Secret("a"), arg2='a2')
    assert str(sec.arg1) == "a"
    assert sec.arg2 == "a2"


def test_results_assignment(fixture_shared_results):
    class TStep(Step[TResults, TArgs, NoResources, NoInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TResults(x=200)

    step = TStep("test-step-1", TArgs(arg1=1), fixture_shared_results, NoResources(), NoInputs())
    results = step.results
    step.run()
    assert results.x == 200
    

def test_tractor_add_steps(fixture_shared_results):
    class TStep(Step[TResults, TArgs, NoResources, NoInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TResults(x=200)

    step = TStep("test-step-1", TArgs(arg1=1), fixture_shared_results, NoResources(), NoInputs())
    results = step.results
    step.run()
    assert results.x == 200
    


