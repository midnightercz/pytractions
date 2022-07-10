from typing import List, TypeVar, Generic, TypedDict, ClassVar, Type, Optional, get_args
from unittest import mock

import pydantic
import pytest

import pytraction

from pytraction.traction import (
    Step, StepResults, StepArgs, NoInputs, StepInputs, NoResources,
    ExtResources, StepOnUpdateCallable, StepErrors, StepDetails,
    ExtResource, NoArgs,
    StepFailedError, Tractor, Secret,
    StepOnErrorCallable, StepOnUpdateCallable,
    TractorDumpDict, StepResults, NoDetails, StepState)

from pytraction.exc import (LoadWrongStepError, LoadWrongExtResourceError, MissingSecretError)

class TResults(StepResults):
    x: int = 10


class TArgs(StepArgs):
    arg1: int


class TSecretArgs(StepArgs):
    arg1: Secret
    arg2: int

class TResource(ExtResource):
    NAME: ClassVar[str] = 'Resource1'
    env: str


class TResourceWithSecrets(ExtResource):
    NAME: ClassVar[str] = 'TResourceWithSecrets'
    SECRETS: ClassVar[List[str]] = ['secret']
    env: str
    secret: str


class TResources(ExtResources):
    NAME: ClassVar[str] = "TResources"
    service1: TResource

class TResourcesWithSecrets(ExtResources):
    NAME: ClassVar[str] = "TResourcesWithSecrets"
    service1: TResourceWithSecrets


class TResources2(ExtResources):
    NAME: ClassVar[str] = "TResources2"
    service1: TResourceWithSecrets


class TInputs(StepInputs):
    input1: TResults


class TDetails(StepDetails):
    status: str = ""


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



def test_step_initiation_no_generic():
    class TStep(Step):
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
         
    with pytest.raises(TypeError) as exc:
        TStep("test-step-1", {},)

    print(exc)
    assert str(exc.value).startswith("Missing generic annotations for Step class")


def test_step_initiation_no_run_method():
    class TStep(Step[TResults, TArgs, TResources, NoInputs, NoDetails]):
        pass
    
    with pytest.raises(TypeError) as exc:
        step = TStep("test-step-1")
    assert str(exc.value) == "Can't instantiate abstract class TStep with abstract method _run"


def test_step_initiation_succesful_no_args_no_inputs():
    class TStep(Step[TResults, NoArgs, TResources, NoInputs, NoDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
    
    step = TStep("test-step-1", NoArgs())
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


def test_step_initiation_succesful_no_args():
    class TStep(Step[TResults, NoArgs, TResources, TInputs, NoDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
 
    step = TStep("test-step-1", NoArgs(),  TResources(service1=TResource(env='test', uid='res1')), TInputs(input1=TResults()))
    assert step.inputs.input1.x == 10


def test_step_initiation_succesful_no_inputs():
    class TStep(Step[TResults, TArgs, TResources, NoInputs, NoDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
    
    step = TStep("test-step-1", TArgs(arg1=10))
    assert step.args.arg1 == 10


def test_step_initiation_wrong_arg_type():
    """Step expects NoArgs but TArgs are given."""
    class TStep(Step[TResults, NoArgs, TResources, NoInputs, NoDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
    
    with pytest.raises(TypeError):
        step = TStep("test-step-1", TArgs(arg1=10))


def test_step_initiation_wrong_inputs_type():
    """Step expects NoInputs but TInputs are given."""

    class TStep(Step[TResults, NoArgs, TResources, NoInputs, NoDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
    
    with pytest.raises(TypeError) as exc:
        step = TStep("test-step-1", NoArgs(), TResources(service1=TResource(env='test', uid='res1')), TInputs(input1=TResults(x=1)))
    assert str(exc.value).startswith("Step inputs are not type of <class 'pytraction.traction.NoInputs'>")


def test_step_initiation_wrong_external_resources():
    """Step expects NoInputs but TInputs are given."""

    class TStep(Step[TResults, NoArgs, NoResources, NoInputs, NoDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
    
    with pytest.raises(TypeError):
        step = TStep("test-step-1", NoArgs(), NoInputs())


def test_step_initiation_missing_arguments():
    """Step initiation is missing shared_reults."""

    class TStep(Step[TResults, NoArgs, NoResources, NoInputs, NoDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
    
    with pytest.raises(TypeError):
        step = TStep("test-step-1", NoArgs(), NoInputs())


def test_step_run_results():
    """Step run results test."""
    class TStep(Step[TResults, NoArgs, NoResources, NoInputs, NoDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results.x = 10
            
    step = TStep("test-step-1", NoArgs(), NoResources(), NoInputs())
    step.run()
    assert step.results.x == 10


def test_step_run_details():
    """Step run with details."""
    class TStep(Step[TResults, NoArgs, NoResources, NoInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.details.status = "ok"
            self.results.x = 10
            
    step = TStep("test-step-1", NoArgs(), NoResources(), NoInputs())
    step.run()
    assert step.results.x == 10
    assert step.details.status == 'ok'




def test_step_run_status_update():
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
            
    step = TStep("test-step-1", NoArgs(), NoResources(), NoInputs())
    step.run(on_update=state_collect)
    assert step.results.x == 10
    assert step.details.status == 'ok'
    assert states_collected == [StepState.PREP, StepState.RUNNING, StepState.RUNNING, StepState.FINISHED]


def test_step_run_secret_arg():
    """Step run with secret args."""
    
    class TTResults(StepResults):
        x: str = 10


    class TStep(Step[TTResults, TSecretArgs, NoResources, NoInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results.x = str(self.args.arg1)

    step = TStep("test-step-1", TSecretArgs(arg1=Secret("supersecret"), arg2=100), NoResources(), NoInputs())
    step.run()
            
    assert step.args.arg1 == 'supersecret'


def test_step_run_invalid_state():
    """Step run in invalid state."""

    class TStep(Step[TResults, TArgs, NoResources, NoInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results.x = self.args.arg1

    step = TStep("test-step-1", TArgs(arg1=1), NoResources(), NoInputs())
    step.state = StepState.RUNNING
    step.run()
    assert step.results.x == 10 # step hasn't run, so result should be default value


def test_step_run_failed():
    """Step initiation is missing shared_reults."""

    class TStep(Step[TResults, TArgs, NoResources, NoInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            raise StepFailedError("step run failed")

    step = TStep("test-step-1", TArgs(arg1=1), NoResources(), NoInputs())
    step.run()
    assert step.state == StepState.FAILED


def test_step_run_error():
    """Step initiation is missing shared_reults."""

    class TStep(Step[TResults, TArgs, NoResources, NoInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            raise ValueError("unexpected error")

    step = TStep("test-step-1", TArgs(arg1=1), NoResources(), NoInputs())
    with pytest.raises(ValueError):
        step.run()
    assert step.state == StepState.ERROR


def test_step_dump_load(fixture_isodate_now):

    class TStep(Step[TResults, TSecretArgs, TResources2, TInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results.x = 1
            self.details.status = 'done'


    standalone_input = TResults(x=55)

    step = TStep("test-step-1",
                 TSecretArgs(arg1=Secret("supersecret"), arg2=200),
                 TResources2(service1=TResourceWithSecrets(env='test', uid='res1', secret='secret')),
                 TInputs(input1=standalone_input))
    step.run()

    assert step.state == StepState.FINISHED
            
    dumped = step.dump()
    assert dumped == {
        'args': {'arg1': '*CENSORED*', 'arg2': 200},
        "details": {'status':'done'},
        'errors': {'errors': {}},
        'inputs': {},
        'inputs_standalone': {"input1": {"x": 55}},
        'external_resources': {'type': 'TResources2',
                               'service1': {'env': 'test',
                                            'uid': 'res1',
                                            'secret': "*CENSORED*",
                                            'type': 'TResourceWithSecrets'}},
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
        'results': {'x':1}
    }
    step2 = TStep("test-step-1",
                  TSecretArgs(arg1=Secret("supersecret"), arg2=200),
                  TResources2(service1=TResourceWithSecrets(env='test', uid='res1', secret='secret value')),
                  TInputs(input1=standalone_input))
    step2.load(dumped, secrets={'TResourceWithSecrets:res1':{'secret': 'secret value'}})
    assert step2.args.arg1 == "supersecret"
    assert step2.args.arg2 == 200
    assert step2.results.x == 1
    assert step2.skip is False
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

    step3 = TStep.load_cls(
        dumped,
        {},
        secrets={'%s:%s' % (step2.NAME, step.uid): {'arg1': 'supersecret'},
                 'TResourceWithSecrets:res1':{'secret': 'secret value'}})
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

def test_step_dump_load_missing_secrets(fixture_isodate_now):

    class TStep(Step[TResults, TSecretArgs, TResources2, TInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results.x = 1
            self.details.status = 'done'


    standalone_input = TResults(x=55)

    step = TStep("test-step-1",
                 TSecretArgs(arg1=Secret("supersecret"), arg2=200),
                 TResources2(service1=TResourceWithSecrets(env='test', uid='res1', secret='secret')),
                 TInputs(input1=standalone_input))
    step.run()

    assert step.state == StepState.FINISHED
            
    dumped = step.dump()
    assert dumped == {
        'args': {'arg1': '*CENSORED*', 'arg2': 200},
        "details": {'status':'done'},
        'errors': {'errors': {}},
        'inputs': {},
        'inputs_standalone': {"input1": {"x": 55}},
        'external_resources': {'type': 'TResources2',
                               'service1': {'env': 'test',
                                            'uid': 'res1',
                                            'secret': "*CENSORED*",
                                            'type': 'TResourceWithSecrets'}},
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
        'results': {'x':1}
    }
    step2 = TStep("test-step-1",
                  TSecretArgs(arg1=Secret("supersecret"), arg2=200),
                  TResources2(service1=TResourceWithSecrets(env='test', uid='res1', secret='secret value')),
                  TInputs(input1=standalone_input))
    with pytest.raises(MissingSecretError) as e:
        step3 = TStep.load_cls(
            dumped,
            {})


def test_step_dump_load_multiple(fixture_isodate_now):
    """Step run with secret args."""

    class TStep(Step[TResults, TSecretArgs, NoResources, TInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results.x = 1
            self.details.status = 'done'
            print("step run")

    standalone_input = TResults(x=55)

    step = TStep("test-step-1",
                 TSecretArgs(arg1=Secret("supersecret"), arg2=200),
                 NoResources(),
                 TInputs(input1=standalone_input))
    step2 = TStep("test-step-2",
                  TSecretArgs(arg1=Secret("supersecret"), arg2=200),
                  NoResources(),
                  TInputs(input1=step.results))
    step.run()
    step2.run()

    assert step.state == StepState.FINISHED
    assert step2.state == StepState.FINISHED
 
    dumped = step.dump()
    assert dumped == {
        'args': {'arg1': '*CENSORED*', 'arg2': 200},
        "details": {'status': 'done'},
        'errors': {'errors': {}},
        'inputs': {},
        'inputs_standalone': {"input1": {"x": 55}},
        'skip': False,
        'skip_reason': '',
        'external_resources': {'type': 'NoResources'},
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
        'results': {'x': 1}
    }
    dumped2 = step2.dump()
    assert dumped2 == {
        'args': {'arg1': '*CENSORED*', 'arg2': 200},
        "details": {'status': 'done'},
        'errors': {'errors': {}},
        'inputs': {'input1': 'TestStep:test-step-1'},
        'inputs_standalone': {},
        'skip': False,
        'skip_reason': '',
        'state': StepState.FINISHED,
        'external_resources': {'type': 'NoResources'},
        'stats': {
            'skip': False,
            'skipped': False,
            'skip_reason': '',
            'state': StepState.FINISHED,
            'started': '1990-01-01T00:00:02.00000Z',
            'finished': '1990-01-01T00:00:03.00000Z'
        },
        'uid': 'test-step-2',
        'type': step.NAME,
        'results': {'x': 1}
    }

    step3 = TStep.load_cls(dumped, {}, secrets={'%s:%s' % (step.NAME, step.uid): {'arg1': 'supersecret'}})
    step4 = TStep.load_cls(dumped2, {step3.fullname: step3}, secrets={'%s:%s' % (step.NAME, step2.uid): {'arg1':'supersecret'}})

    assert step3.uid == step.uid
    assert step3.state == step.state
    assert step3.skip == step.skip
    assert step3.skip_reason == step.skip_reason
    assert step3.results == step.results
    assert step3.errors == step.errors
    assert step3.details == step.details
    assert step3.stats == step.stats
    assert step3.inputs == step.inputs
    assert step3.args == step.args

    assert step4.uid == step2.uid
    assert step4.state == step2.state
    assert step4.skip == step2.skip
    assert step4.skip_reason == step2.skip_reason
    assert step4.results == step2.results
    assert step4.errors == step2.errors
    assert step4.details == step2.details
    assert step4.stats == step2.stats
    assert step4.inputs == step2.inputs
    assert step4.args == step2.args


def test_step_dump_load_cls_wrong(fixture_isodate_now):
    """Step run with secret args."""

    class TStep(Step[TResults, TSecretArgs, NoResources, TInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results.x = 1
            self.details.status = 'done'
            print("step run")


    standalone_input = TResults(x=55)

    step = TStep("test-step-1",
                 TSecretArgs(arg1=Secret("supersecret"), arg2=200),
                 NoResources(),
                 TInputs(input1=standalone_input))
    dumped = {
        'args': {'arg1': '*CENSORED*', 'arg2': 200},
        "details": {'status': 'done'},
        'errors': {'errors': {}},
        'inputs': {},
        'inputs_standalone': {"input1": {"x": 55}},
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
        'type': "WrongStep",
        'results': {'x': 1}
    }
    with pytest.raises(LoadWrongStepError):
        step2 = TStep.load_cls(dumped, {'arg1': Secret('supersecret')},  {step.fullname: step})


def test_step_dump_load_wrong(fixture_isodate_now):
    """Step run with secret args."""

    class TStep(Step[TResults, TSecretArgs, NoResources, TInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results.x = 1
            self.details.status = 'done'
            print("step run")


    standalone_input = TResults(x=55)

    step = TStep("test-step-1",
                 TSecretArgs(arg1=Secret("supersecret"), arg2=200),
                 NoResources(),
                 TInputs(input1=standalone_input))
    dumped = {
        'args': {'arg1': '*CENSORED*', 'arg2': 200},
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
        'type': "WrongStep",
        'results': {'x':1}
    }
    with pytest.raises(LoadWrongStepError):
        step.load(dumped)


def test_step_dict():
    """Step initiation is missing shared_reults."""

    class TStep(Step[TResults, TSecretArgs, NoResources, NoInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results.x = self.args.arg1

    step = TStep("test-step-1", TSecretArgs(arg1=Secret("supersecret"), arg2=200), NoResources(), NoInputs())
    assert step.dict()['args']['arg1'] == "*CENSORED*"
    assert step.dict()['args']['arg2'] == 200


def test_step_generic():
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


    step = Model1Loader("test-step-1", TSecretArgs(arg1=Secret("supersecret"), arg2=200), NoResources(), NoInputs())
    step.run()
    assert step.dict()['args']['arg1'] == "*CENSORED*"
    assert step.dict()['args']['arg2'] == 200


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
        assert TSecretArgs(arg1="1", arg2=200)


def test_invalid_secret_arguments_compare():
    sec2 = TSecretArgs(arg1=Secret("a"), arg2=200)
    sec = TSecretArgs(arg1=Secret("a"), arg2=200)
    assert sec.arg1 == "a"
    assert sec.arg1 == sec2.arg1


def test_secret_str():
    sec = TSecretArgs(arg1=Secret("a"), arg2=200)
    assert str(sec.arg1) == "a"
    assert sec.arg2 == 200


def test_ext_resources_wrong_type():
    with pytest.raises(ValueError) as exc:
        class TResources(ExtResources):
            NAME: ClassVar[str] = "TResources"
            service1: str
    assert str(exc.value) == "service1 has to be type ExtResource"

def test_ext_resources_wrong_type_init():
    with pytest.raises(ValueError) as exc:
        res = TResources(service1=1, uid='res1')
    assert str(exc.value) == "service1 has to be type ExtResource not <class 'int'>"

def test_ext_resources_load():
    dump = {'type': 'TResourcesWithSecrets',
            'service1': {'env':'test2', 'uid':'res1', "type": "TResourceWithSecrets", "secret": "*CENSORED*"}}
    res = TResourcesWithSecrets.load_cls(dump, secrets={"TResourceWithSecrets:res1": {"secret": "secret value"}})
    assert res.service1.env == 'test2'
    assert res.service1.secret == 'secret value'

    res = TResourcesWithSecrets(uid='resource1', service1=TResourceWithSecrets(env='test', uid='res1', secret="secret value2")).load(dump)
    assert res.service1.env == 'test2'
    assert res.service1.secret == 'secret value2'

def test_ext_resource_load_wrong_type():
    dump = {'type': 'TResources',
            'service1': {'env':'test2', 'uid':'res1', "type": "Resource2"}}
    with pytest.raises(LoadWrongExtResourceError) as exc:
        res = TResources.load_cls(dump)
    assert str(exc.value) == "Cannot load Resource2 into Resource1"
    with pytest.raises(LoadWrongExtResourceError) as exc:
        res = TResources(uid='resource1', service1=TResource(env='test', uid='res1')).load(dump)
    assert str(exc.value) == "Cannot load Resource2 into Resource1"
    
def test_ext_resources_load_wrong_type():
    dump = {'type': 'TResources2',
            'service1': {}}
    with pytest.raises(LoadWrongExtResourceError) as exc:
        res = TResources.load_cls(dump)
    assert str(exc.value) == "Cannot load TResources2 into TResources"

    with pytest.raises(LoadWrongExtResourceError) as exc:
        res = TResources(service1=TResource(env='test', uid='res1')).load(dump)
    assert str(exc.value) == "Cannot load TResources2 into TResources"

def test_results_assignment():
    class TStep(Step[TResults, TArgs, NoResources, NoInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TResults(x=200)

    step = TStep("test-step-1", TArgs(arg1=1), NoResources(), NoInputs())
    results = step.results
    step.run()
    assert results.x == 200


def test_tractor_add_steps():
    class TStep(Step[TResults, TArgs, NoResources, TInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TResults(x=self.inputs.input1 + self.args.arg1)

    tractor = Tractor(step_map={"TestStep": TStep}, resources_map={'Resource1': TResource})

    step1 = TStep("test-step-1", TArgs(arg1=1), NoResources(), TInputs(input1=TResults(x=1)))
    step2 = TStep("test-step-2", TArgs(arg1=2), NoResources(), TInputs(input1=step1.results))
    step3 = TStep("test-step-3", TArgs(arg1=3), NoResources(), TInputs(input1=step2.results))
    
    tractor.add_step(step1)
    tractor.add_step(step2)
    tractor.add_step(step3)

    assert tractor.current_step == None
    assert tractor.steps == [step1, step2, step3]


def test_tractor_dump_load():
    class TStep(Step[TResults, TSecretArgs, TResources2, TInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TResults(x=self.inputs.input1 + self.args.arg1)

    tractor = Tractor(step_map={"TestStep": TStep}, resources_map={'Resource1': TResource})

    step1 = TStep("test-step-1",
                  TSecretArgs(arg1=Secret('1'), arg2=1),
                  TResources2(service1=TResourceWithSecrets(env='test', secret='1', uid='res1')),
                  TInputs(input1=TResults(x=1)))
    step2 = TStep("test-step-2", 
                   TSecretArgs(arg1=Secret('2'), arg2=2),
                   TResources2(service1=TResourceWithSecrets(env='test', secret='1', uid='res1')),
                   TInputs(input1=step1.results))
    step3 = TStep("test-step-3",
                  TSecretArgs(arg1=Secret('3'), arg2=3),
                  TResources2(service1=TResourceWithSecrets(env='test', secret='1', uid='res1')),
                  TInputs(input1=step2.results))
    
    tractor.add_step(step1)
    tractor.add_step(step2)
    tractor.add_step(step3)

    dumped = tractor.dump()
    assert dumped == {
        'resources': {'TResourceWithSecrets:res1': {
                        'secret': "*CENSORED*",
                        'env': 'test',
                        'uid': 'res1',
                        'type': 'TResourceWithSecrets'},
                     },
        'steps': [{'args': {'arg1': '*CENSORED*', 'arg2': 1},
                   'details': {'status': ''},
                   'errors': {'errors': {}},
                   'external_resources': {'service1': 'TResourceWithSecrets:res1',
                                          'type': 'TResources2'},
                   'inputs': {},
                   'inputs_standalone': {'input1': {'x': 1}},
                   'results': {'x': 10},
                   'skip': False,
                   'skip_reason': '',
                   'state': StepState.READY,
                   'stats': {'finished': None,
                             'skip': False,
                             'skip_reason': '',
                             'skipped': False,
                             'started': None,
                             'state': StepState.READY},
                   'type': 'TestStep',
                   'uid': 'test-step-1'},
                  {'args': {'arg1': '*CENSORED*', 'arg2': 2},
                   'details': {'status': ''},
                   'errors': {'errors': {}},
                   'external_resources': {'service1': 'TResourceWithSecrets:res1',
                                          'type': 'TResources2'},
                   'inputs': {'input1': 'TestStep:test-step-1'},
                   'inputs_standalone': {},
                   'results': {'x': 10},
                   'skip': False,
                   'skip_reason': '',
                   'state': StepState.READY,
                   'stats': {'finished': None,
                             'skip': False,
                             'skip_reason': '',
                             'skipped': False,
                             'started': None,
                             'state': StepState.READY},
                   'type': 'TestStep',
                   'uid': 'test-step-2'},
                  {'args': {'arg1': '*CENSORED*', 'arg2': 3},
                   'details': {'status': ''},
                   'errors': {'errors': {}},
                   'external_resources': {'service1': 'TResourceWithSecrets:res1',
                                          'type': 'TResources2',
                                         },
                   'inputs': {'input1': 'TestStep:test-step-2'},
                   'inputs_standalone': {},
                   'results': {'x': 10},
                   'skip': False,
                   'skip_reason': '',
                   'state': StepState.READY,
                   'stats': {'finished': None,
                             'skip': False,
                             'skip_reason': '',
                             'skipped': False,
                             'started': None,
                             'state': StepState.READY},
                   'type': 'TestStep',
                   'uid': 'test-step-3'}]
        }


    tractor2 = Tractor(step_map={"TestStep": TStep}, resources_map={'Resource1': TResource, 'TResourceWithSecrets': TResourceWithSecrets})
    with pytest.raises(MissingSecretError):
        tractor2.load(dumped, {})

    tractor2.load(dumped, {"TestStep:test-step-1": {'arg1': '1'},
                           "TestStep:test-step-2": {'arg1': '2'},
                           "TestStep:test-step-3": {'arg1': '3'},
                           'TResourceWithSecrets:res1': {'secret': 'secret value'}})
    assert tractor2.steps[0].args.arg1 == '1'
    assert tractor2.steps[0].details.status == ''
    assert tractor2.steps[0].errors.errors == {}
    assert tractor2.steps[0].external_resources.service1 == TResourceWithSecrets(env='test', uid='res1', secret='secret value')
    assert tractor2.steps[0].inputs.input1 == TResults(x=1)
    assert tractor2.steps[0].skip == False
    assert tractor2.steps[0].skip_reason == ''
    assert tractor2.steps[0].state == StepState.READY

    assert tractor2.steps[1].args.arg1 == '2'
    assert tractor2.steps[1].details.status == ''
    assert tractor2.steps[1].errors.errors == {}
    assert tractor2.steps[1].external_resources.service1 == TResourceWithSecrets(env='test', uid='res1', secret='secret value')
    assert tractor2.steps[1].inputs.input1 is tractor2.steps[0].results
    assert tractor2.steps[1].skip == False
    assert tractor2.steps[1].skip_reason == ''
    assert tractor2.steps[1].state == StepState.READY

    assert tractor2.steps[2].args.arg1 == '3'
    assert tractor2.steps[2].details.status == ''
    assert tractor2.steps[2].errors.errors == {}
    assert tractor2.steps[2].external_resources.service1 == TResourceWithSecrets(env='test', uid='res1', secret='secret value')
    assert tractor2.steps[2].inputs.input1 is tractor2.steps[1].results
    assert tractor2.steps[2].skip == False
    assert tractor2.steps[2].skip_reason == ''
    assert tractor2.steps[2].state == StepState.READY


def test_tractor_run():
    class TStep(Step[TResults, TSecretArgs, TResources, TInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TResults(x=self.inputs.input1.x + self.args.arg2)

    tractor = Tractor(step_map={"TestStep": TStep}, resources_map={'Resource1': TResource})

    step1 = TStep("test-step-1", TSecretArgs(arg1=Secret('1'), arg2=1), TResources(service1=TResource(env='test', uid='res1')), TInputs(input1=TResults(x=1)))
    step2 = TStep("test-step-2", TSecretArgs(arg1=Secret('2'), arg2=2), TResources(service1=TResource(env='test', uid='res1')), TInputs(input1=step1.results))
    step3 = TStep("test-step-3", TSecretArgs(arg1=Secret('3'), arg2=3), TResources(service1=TResource(env='test', uid='res1')), TInputs(input1=step2.results))
    
    tractor.add_step(step1)
    tractor.add_step(step2)
    tractor.add_step(step3)

    tractor.run()

    assert step1.state == StepState.FINISHED
    assert step2.state == StepState.FINISHED
    assert step3.state == StepState.FINISHED

    assert step3.results.x == 7


def test_tractor_run_error():
    class TStep(Step[TResults, TSecretArgs, TResources, TInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TResults(x=self.inputs.input1.x + self.args.arg2)
            if self.results.x > 4:
                raise ValueError("Value too high")

    on_error_called = {"count": 0}

    def on_error(step: Step):
        print("on error")
        on_error_called['count'] = 1

    tractor = Tractor(step_map={"TestStep": TStep}, resources_map={'Resource1': TResource})

    step1 = TStep("test-step-1", TSecretArgs(arg1=Secret('1'), arg2=1), TResources(service1=TResource(env='test', uid='res1')), TInputs(input1=TResults(x=1)))
    step2 = TStep("test-step-2", TSecretArgs(arg1=Secret('2'), arg2=2), TResources(service1=TResource(env='test', uid='res1')), TInputs(input1=step1.results))
    step3 = TStep("test-step-3", TSecretArgs(arg1=Secret('3'), arg2=3), TResources(service1=TResource(env='test', uid='res1')), TInputs(input1=step2.results))
    
    tractor.add_step(step1)
    tractor.add_step(step2)
    tractor.add_step(step3)

    with pytest.raises(ValueError):
        tractor.run(on_error=on_error)

    assert step1.state == StepState.FINISHED
    assert step2.state == StepState.FINISHED
    assert step3.state == StepState.ERROR

    assert step3.results.x == 7
    assert on_error_called == {"count": 1}
