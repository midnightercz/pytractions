from typing import List, TypeVar, Generic, TypedDict, ClassVar, Type, Optional, get_args
from unittest import mock

import pydantic
import pytest

from pytraction.traction import (
    Step, StepResults, StepArgs, NoInputs, StepInputs, NoResources,
    ExtResources, StepOnUpdateCallable, StepErrors, StepDetails,
    ExtResource, NoArgs,
    StepFailedError, Tractor, Secret,
    StepOnErrorCallable, StepOnUpdateCallable,
    TractorDumpDict, NoDetails, StepState, StepStats,
    NamedTractor, NTInput, STMD)

from pytraction.exc import (LoadWrongStepError, LoadWrongExtResourceError, MissingSecretError, DuplicateStepError, DuplicateTractorError)

from .models import (
    TResults, TArgs, TResources, TSecretArgs, TResource, TResourceWithSecrets, TResources, TResourcesWithSecrets, TResources2,
    TInputs, TDetails)

def test_tractor_add_steps():
    class TStep(Step[TResults, TArgs, NoResources, TInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TResults(x=self.inputs.input1 + self.args.arg1)

    tractor = Tractor(uid='t1')

    step1 = TStep("test-step-1", TArgs(arg1=1), NoResources(), TInputs(input1=TResults(x=1)))
    step2 = TStep("test-step-2", TArgs(arg1=2), NoResources(), TInputs(input1=step1.results))
    step3 = TStep("test-step-3", TArgs(arg1=3), NoResources(), TInputs(input1=step2.results))
    
    tractor.add_step(step1)
    tractor.add_step(step2)
    tractor.add_step(step3)

    assert tractor.current_step == None
    assert tractor.steps == [step1, step2, step3]


def test_tractor_add_steps_duplicate():
    class TStep(Step[TResults, TArgs, NoResources, TInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TResults(x=self.inputs.input1 + self.args.arg1)

    tractor = Tractor(uid='t1')
    step1 = TStep("test-step-1", TArgs(arg1=1), NoResources(), TInputs(input1=TResults(x=1)))
    
    tractor.add_step(step1)
    with pytest.raises(DuplicateStepError):
        tractor.add_step(step1)


def test_tractor_dump_load():
    class TStep(Step[TResults, TSecretArgs, TResources2, TInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TResults(x=self.inputs.input1 + self.args.arg1)

    tractor = Tractor(uid='t1')

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
        'uid': 't1',
        'resources': {'TResourceWithSecrets:res1': {
                        'secret': "*CENSORED*",
                        'env': 'test',
                        'uid': 'res1',
                        'type': 'TResourceWithSecrets'},
                     },
        "current_step":None,
        'steps': [{
                   "type": "step",
                   "data": {
                       'args': {'arg1': '*CENSORED*', 'arg2': 1},
                       'details': {'status': ''},
                       'errors': {'errors': {}},
                       'resources': {'service1': 'TResourceWithSecrets:res1',
                                              'type': 'TResources2'},
                       'inputs': {},
                       'inputs_standalone': {'input1': {'x': 1}},
                       'results': {'x': 10},
                       'skip': False,
                       'skip_reason': '',
                       'state': 'ready',
                       'stats': {'finished': None,
                                 'skipped': False,
                                 'started': None},
                       'type': 'TestStep',
                       'uid': 'test-step-1'}
                  },
                  {"type": "step",
                   "data": {
                       'args': {'arg1': '*CENSORED*', 'arg2': 2},
                       'details': {'status': ''},
                       'errors': {'errors': {}},
                       'resources': {'service1': 'TResourceWithSecrets:res1',
                                              'type': 'TResources2'},
                       'inputs': {'input1': 'TestStep:test-step-1'},
                       'inputs_standalone': {},
                       'results': {'x': 10},
                       'skip': False,
                       'skip_reason': '',
                       'state': 'ready',
                       'stats': {'finished': None,
                                 'skipped': False,
                                 'started': None},
                       'type': 'TestStep',
                       'uid': 'test-step-2'}
                   },
                   {"type": "step",
                    "data": {
                       'args': {'arg1': '*CENSORED*', 'arg2': 3},
                        'details': {'status': ''},
                        'errors': {'errors': {}},
                        'resources': {'service1': 'TResourceWithSecrets:res1',
                                               'type': 'TResources2',
                                              },
                        'inputs': {'input1': 'TestStep:test-step-2'},
                        'inputs_standalone': {},
                        'results': {'x': 10},
                        'skip': False,
                        'skip_reason': '',
                        'state': 'ready',
                        'stats': {'finished': None,
                                  'skipped': False,
                                  'started': None},
                        'type': 'TestStep',
                        'uid': 'test-step-3'}
                   }
                ]
        }


    tractor2 = Tractor(uid='t1')
    with pytest.raises(MissingSecretError):
        tractor2.load(dumped, {"TestStep": TStep}, {"TResourceWithSecrets": TResourceWithSecrets}, {})

    tractor2.load(
        dumped, 
        {"TestStep": TStep},
        {"TResourceWithSecrets": TResourceWithSecrets},
        {"TestStep:test-step-1": {'arg1': '1'},
         "TestStep:test-step-2": {'arg1': '2'},
         "TestStep:test-step-3": {'arg1': '3'},
         'TResourceWithSecrets:res1': {'secret': 'secret value'}})
    assert tractor2.steps[0].args.arg1 == '1'
    assert tractor2.steps[0].details.status == ''
    assert tractor2.steps[0].errors.errors == {}
    assert tractor2.steps[0].resources.service1 == TResourceWithSecrets(env='test', uid='res1', secret='secret value')
    assert tractor2.steps[0].inputs.input1 == TResults(x=1)
    assert tractor2.steps[0].skip == False
    assert tractor2.steps[0].skip_reason == ''
    assert tractor2.steps[0].state == StepState.READY

    assert tractor2.steps[1].args.arg1 == '2'
    assert tractor2.steps[1].details.status == ''
    assert tractor2.steps[1].errors.errors == {}
    assert tractor2.steps[1].resources.service1 == TResourceWithSecrets(env='test', uid='res1', secret='secret value')
    assert tractor2.steps[1].inputs.input1 is tractor2.steps[0].results
    assert tractor2.steps[1].skip == False
    assert tractor2.steps[1].skip_reason == ''
    assert tractor2.steps[1].state == StepState.READY

    assert tractor2.steps[2].args.arg1 == '3'
    assert tractor2.steps[2].details.status == ''
    assert tractor2.steps[2].errors.errors == {}
    assert tractor2.steps[2].resources.service1 == TResourceWithSecrets(env='test', uid='res1', secret='secret value')
    assert tractor2.steps[2].inputs.input1 is tractor2.steps[1].results
    assert tractor2.steps[2].skip == False
    assert tractor2.steps[2].skip_reason == ''
    assert tractor2.steps[2].state == StepState.READY


def test_tractor_run():
    class TStep(Step[TResults, TSecretArgs, TResources, TInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TResults(x=self.inputs.input1.x + self.args.arg2)

    tractor = Tractor(uid='t1')

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

    tractor = Tractor(uid='t1')

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


def test_tractor_add_subtractor():
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

    tractor = Tractor(uid='t1')
    step1 = TStep("test-step-1", TSecretArgs(arg1=Secret('1'), arg2=1), TResources(service1=TResource(env='test', uid='res1')), TInputs(input1=TResults(x=1)))
    tractor.add_step(step1)

    tractor2 = Tractor(uid='t2')
    step2 = TStep("test-step-2", TSecretArgs(arg1=Secret('2'), arg2=2), TResources(service1=TResource(env='test', uid='res1')), TInputs(input1=step1.results))
    tractor2.add_step(step2)

    tractor.add_step(tractor2)

    tractor.run(on_error=on_error)

    assert step1.state == StepState.FINISHED
    assert step2.state == StepState.FINISHED


def test_tractor_add_subtractor_duplicate_step():
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

    tractor = Tractor(uid='t1')
    step1 = TStep("test-step-1", TSecretArgs(arg1=Secret('1'), arg2=1), TResources(service1=TResource(env='test', uid='res1')), TInputs(input1=TResults(x=1)))
    tractor.add_step(step1)

    tractor2 = Tractor(uid='t2')
    step2 = TStep("test-step-2", TSecretArgs(arg1=Secret('2'), arg2=2), TResources(service1=TResource(env='test', uid='res1')), TInputs(input1=step1.results))
    tractor2.add_step(step2)
    tractor.add_step(tractor2)

    tractor3 = Tractor(uid='t2')
    tractor3.add_step(step2)
    
    with pytest.raises(DuplicateStepError):
        tractor.add_step(tractor3)


def test_tractor_add_subtractor_duplicate_tractor():
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

    tractor = Tractor(uid='t1')
    step1 = TStep("test-step-1", TSecretArgs(arg1=Secret('1'), arg2=1), TResources(service1=TResource(env='test', uid='res1')), TInputs(input1=TResults(x=1)))
    tractor.add_step(step1)

    tractor2 = Tractor(uid='t2')
    step2 = TStep("test-step-2", TSecretArgs(arg1=Secret('2'), arg2=2), TResources(service1=TResource(env='test', uid='res1')), TInputs(input1=step1.results))
    tractor2.add_step(step2)
    tractor.add_step(tractor2)

    tractor3 = Tractor(uid='t3')
    tractor3.add_step(tractor)
    
    with pytest.raises(DuplicateTractorError):
        tractor.add_step(tractor3)


def test_named_tractor():
    class TStep(Step[TResults, TArgs, NoResources, TInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TResults(x=self.inputs.input1.x + self.args.arg1)

    class TNamedTractor(NamedTractor, nt_steps=[('step1', TStep), ('step2', TStep)], nt_inputs={'input1': TResults}):
        INPUTS_MAP = {
            "step2": {"input1": "step1"},
            "step1": {"input1": NTInput(name="input1")}
        }

    nt = TNamedTractor(
        uid='nt1',
        args=TNamedTractor.ArgsModel(step1=TArgs(arg1=20), step2=TArgs(arg1=50)),
        resources=TNamedTractor.ResourcesModel(step1=NoResources(), step2=NoResources()),
        inputs=TNamedTractor.InputsModel(input1=TResults(x=10))
    )

    nt.run()
    assert nt.steps[0].state == StepState.FINISHED
    assert nt.steps[1].state == StepState.FINISHED
    assert nt.state == StepState.FINISHED
    assert nt.results.step1.x == 30
    assert nt.results.step2.x == 80


def test_named_tractor_step_failed():
    class TStep(Step[TResults, TArgs, NoResources, TInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TResults(x=self.inputs.input1.x + self.args.arg1)
            if self.inputs.input1.x + self.args.arg1>30:
                raise StepFailedError

    class TNamedTractor(NamedTractor, nt_steps=[('step1', TStep), ('step2', TStep)], nt_inputs={'input1': TResults}):
        INPUTS_MAP = {
            "step2": {"input1": "step1"},
            "step1": {"input1": NTInput(name="input1")}
        }

    nt = TNamedTractor(
        uid='nt1',
        args=TNamedTractor.ArgsModel(step1=TArgs(arg1=20), step2=TArgs(arg1=50)),
        resources=TNamedTractor.ResourcesModel(step1=NoResources(), step2=NoResources()),
        inputs=TNamedTractor.InputsModel(input1=TResults(x=10))
    )

    nt.run()
    assert nt.steps[0].state == StepState.FINISHED
    assert nt.steps[1].state == StepState.FAILED
    assert nt.state == StepState.FAILED
    assert nt.results.step1.x == 30
    assert nt.results.step2.x == 80


def test_named_tractor_nested():
    class TStep(Step[TResults, TArgs, NoResources, TInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TResults(x=self.inputs.input1.x + self.args.arg1)

    class TNamedTractor2(NamedTractor, nt_steps=[('step1', TStep), ('step2', TStep)], nt_inputs={'input1': TResults}):
        INPUTS_MAP = {
            "step2": {"input1": "step1"},
            "step1": {"input1": NTInput(name="input1")}
        }

    class TNamedTractor(NamedTractor, nt_steps=[('step1', TStep), ('nt', TNamedTractor2)], nt_inputs={'input1': TResults}):
        INPUTS_MAP = {
            "nt": {"input1": "step1"},
            "step1": {"input1": NTInput(name="input1")}
        }

    nt = TNamedTractor(
        uid='nt1',
        args=TNamedTractor.ArgsModel(
            step1=TArgs(arg1=20), 
            nt=TNamedTractor2.ArgsModel(
                step1=TArgs(arg1=30),
                step2=TArgs(arg1=40))),
        resources=TNamedTractor.ResourcesModel(step1=NoResources(), nt=TNamedTractor2.ResourcesModel(step1=NoResources(), step2=NoResources())),
        inputs=TNamedTractor.InputsModel(input1=TResults(x=10))
    )
    nt.run()
    assert nt.steps[0].state == StepState.FINISHED
    assert nt.steps[1].state == StepState.FINISHED
    assert nt.results.step1.x == 30
    assert nt.results.nt.step1.x == 60
    assert nt.results.nt.step2.x == 100


def test_named_tractor_nested_dump(fixture_isodate_now):
    class TStep(Step[TResults, TArgs, NoResources, TInputs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TResults(x=self.inputs.input1.x + self.args.arg1)

    class TNamedTractor2(NamedTractor, nt_steps=[('step1', TStep), ('step2', TStep)], nt_inputs={'input1': TResults}):
        INPUTS_MAP = {
            "step2": {"input1": "step1"},
            "step1": {"input1": NTInput(name="input1")}
        }

    class TNamedTractor(NamedTractor, nt_steps=[('step1', TStep), ('nt', TNamedTractor2)], nt_inputs={'input1': TResults}):
        INPUTS_MAP = {
            "nt": {"input1": "step1"},
            "step1": {"input1": NTInput(name="input1")}
        }

    nt = TNamedTractor(
        uid='nt1',
        args=TNamedTractor.ArgsModel(
            step1=TArgs(arg1=20), 
            nt=TNamedTractor2.ArgsModel(
                step1=TArgs(arg1=30),
                step2=TArgs(arg1=40))),
        resources=TNamedTractor.ResourcesModel(step1=NoResources(), nt=TNamedTractor2.ResourcesModel(step1=NoResources(), step2=NoResources())),
        inputs=TNamedTractor.InputsModel(input1=TResults(x=10))
    )
    nt.run()
    assert nt.dump(full=False) == {
      "state": "finished",
      "current_step": "nt1::nt",
      "steps": [
        {
          "type": "step",
          "data": {
            "uid": "nt1::step1",
            "state": StepState.FINISHED,
            "skip": False,
            "skip_reason": "",
            "errors": {
              "errors": {}
            },
            "details": {
              "status": ""
            },
            "stats": {
              "started": "1990-01-01T00:00:00.00000Z",
              "finished": "1990-01-01T00:00:01.00000Z",
              "skipped": False,
            },
            "resources": {
              "type": "NoResources"
            },
            "args": {
              "arg1": 20
            },
            "type": "TestStep",
            "inputs": {},
            "inputs_standalone": {
              "input1": {
                "x": 10
              }
            },
            "results": {
              "x": 30
            }
          }
        },
        {
          "type": "tractor",
          "data": {
            "current_step": "TestStep:nt1::nt::step2",
            "state": "finished",
            "steps": [{
                "type": "step",
                "data": {
                  "uid": "nt1::nt::step1",
                  "state": StepState.FINISHED,
                  "skip": False,
                  "skip_reason": "",
                  "errors": {
                    "errors": {}
                  },
                  "details": {
                    "status": ""
                  },
                  "stats": {
                    "started": "1990-01-01T00:00:02.00000Z",
                    "finished": "1990-01-01T00:00:03.00000Z",
                    "skipped": False,
                  },
                  "resources": {
                    "type": "NoResources"
                  },
                  "args": {
                    "arg1": 30
                  },
                  "type": "TestStep",
                  "inputs": {},
                  "inputs_standalone": {
                    "input1": {
                      "x": 30
                    }
                  },
                  "results": {
                    "x": 60
                  }
                }
              },
              {
                "type": "step",
                "data": {
                  "uid": "nt1::nt::step2",
                  "state": StepState.FINISHED,
                  "skip": False,
                  "skip_reason": "",
                  "errors": {
                    "errors": {}
                  },
                  "details": {
                    "status": ""
                  },
                  "stats": {
                    "started": "1990-01-01T00:00:04.00000Z",
                    "finished": "1990-01-01T00:00:05.00000Z",
                    "skipped": False,
                  },
                  "resources": {
                    "type": "NoResources"
                  },
                  "args": {
                    "arg1": 40
                  },
                  "type": "TestStep",
                  "inputs": {},
                  "inputs_standalone": {
                    "input1": {
                      "x": 60
                    }
                  },
                  "results": {
                    "x": 100
                  }
                }
              }
            ],
            "resources": {},
            "uid": "nt1::nt"
          }
        }
      ],
      "resources": {},
      "uid": "nt1"
    }


class STMD_TStep(Step[TResults, TArgs, NoResources, TInputs, TDetails]):
    NAME: ClassVar[str] = "TestStep"
    def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
        self.results = TResults(x=self.inputs.input1.x + self.args.arg1)

class STMD_TNamedTractor(NamedTractor, nt_steps=[('step1', STMD_TStep), ('step2', STMD_TStep)], nt_inputs={'input1': TResults}):
    INPUTS_MAP = {
        "step2": {"input1": "step1"},
        "step1": {"input1": NTInput(name="input1")}
    }


def test_stmd():

    class TSTMD(STMD, tractor_type=STMD_TNamedTractor):
       pass

    tstmd = TSTMD('test-stmd-1',
        args=TSTMD.ArgsModel(step1=TArgs(arg1=50),
                             step2=TArgs(arg1=100),
                             tractors=10),
        resources=TSTMD.ResourcesModel(step1=NoResources(), step2=NoResources()),
        inputs=TSTMD.InputsModel(
        data=TSTMD.MultiDataModel(
            multidata=[
                      STMD_TNamedTractor.InputsModel(input1=TResults(x=10)),
                      STMD_TNamedTractor.InputsModel(input1=TResults(x=20)),
                ]
            )
        )
    )
    tstmd.run()


