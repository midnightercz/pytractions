from typing import List, TypeVar, Generic, TypedDict, ClassVar, Type, Optional, get_args
from unittest import mock

import pydantic
import pytest

from pytraction.traction import (
    Step, StepIOs, StepArgs, NoInputs, NoResources,
    ExtResources, StepOnUpdateCallable, StepErrors, StepDetails,
    ExtResource, NoArgs,
    StepFailedError, Tractor, Secret,
    StepOnErrorCallable, StepOnUpdateCallable,
    TractorDumpDict, NoDetails, StepState, StepStats,
    NamedTractor, NTInput, STMD)

from pytraction.exc import (LoadWrongStepError, LoadWrongExtResourceError, MissingSecretError, DuplicateStepError, DuplicateTractorError)

from .models import (
    TIOs, IntIO, TArgs, TResources, TSecretArgs, TResource, TResourceWithSecrets, TResources, TResourcesWithSecrets, TResources2,
    TDetails)

def test_tractor_add_steps():
    class TStep(Step[TIOs, TArgs, NoResources, TIOs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results.int_io.x = self.inputs.input1 + self.args.arg1

    tractor = Tractor(uid='t1')

    step1 = TStep("test-step-1", TArgs(arg1=1), NoResources(), TIOs(int_io=IntIO(x=1)))
    step2 = TStep("test-step-2", TArgs(arg1=2), NoResources(), TIOs(int_io=step1.results.int_io))
    step3 = TStep("test-step-3", TArgs(arg1=3), NoResources(), TIOs(int_io=step2.results.int_io))
    
    tractor.add_step(step1)
    tractor.add_step(step2)
    tractor.add_step(step3)

    assert tractor.current_step == None
    assert tractor.steps == [step1, step2, step3]


def test_tractor_add_steps_duplicate():
    class TStep(Step[TIOs, TArgs, NoResources, TIOs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TIOs(int_io=IntIO(x=self.inputs.int_io.x + self.args.arg1))

    tractor = Tractor(uid='t1')
    step1 = TStep("test-step-1", TArgs(arg1=1), NoResources(), TIOs(int_io=IntIO(x=1)))
    
    tractor.add_step(step1)
    with pytest.raises(DuplicateStepError):
        tractor.add_step(step1)


def test_tractor_dump_load():
    class TStep(Step[TIOs, TSecretArgs, TResources2, TIOs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TIOs(int_io=IntIO(x=self.inputs.int_io.x + self.args.arg1))

    tractor = Tractor(uid='t1')

    step1 = TStep("test-step-1",
                  TSecretArgs(arg1=Secret('1'), arg2=1),
                  TResources2(service1=TResourceWithSecrets(env='test', secret='1', uid='res1')),
                  TIOs(int_io=IntIO(x=1)))
    step2 = TStep("test-step-2", 
                   TSecretArgs(arg1=Secret('2'), arg2=2),
                   TResources2(service1=TResourceWithSecrets(env='test', secret='1', uid='res1')),
                   TIOs(int_io=step1.results.int_io))
    step3 = TStep("test-step-3",
                  TSecretArgs(arg1=Secret('3'), arg2=3),
                  TResources2(service1=TResourceWithSecrets(env='test', secret='1', uid='res1')),
                  TIOs(int_io=step2.results.int_io))
    
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
                       'inputs_standalone': {'int_io': {'x': 1}},
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
                       'inputs': {'int_io': 'TestStep:test-step-1'},
                       'inputs_standalone': {},
                       'results': {'int_io':{'x': 10}},
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
                        'inputs': {'int_io': 'TestStep:test-step-2'},
                        'inputs_standalone': {},
                        'results': {'int_io':{'x': 10}},
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
    assert tractor2.steps[0].inputs.int_io == IntIO(x=1)
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
    class TStep(Step[TIOs, TSecretArgs, TResources, TIOs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            print("-------- step run", self.fullname)
            print("input", self.inputs.int_io, id(self.inputs.int_io))
            print("result", self.results.int_io, id(self.results.int_io))
            self.results.int_io=IntIO(x=self.inputs.int_io.x + self.args.arg2)
            print("result", self.results.int_io, id(self.results.int_io))

    tractor = Tractor(uid='t1')

    step1 = TStep("test-step-1", TSecretArgs(arg1=Secret('1'), arg2=1), TResources(service1=TResource(env='test', uid='res1')), TIOs(int_io=IntIO(x=1)))
    s2_i = TIOs(int_io=step1.results.int_io)
    step2 = TStep("test-step-2", TSecretArgs(arg1=Secret('2'), arg2=2), TResources(service1=TResource(env='test', uid='res1')), s2_i)
    step3 = TStep("test-step-3", TSecretArgs(arg1=Secret('3'), arg2=3), TResources(service1=TResource(env='test', uid='res1')), TIOs(int_io=step2.results.int_io))
    
    tractor.add_step(step1)
    tractor.add_step(step2)
    tractor.add_step(step3)
    tractor.run()

    assert step1.state == StepState.FINISHED
    assert step2.state == StepState.FINISHED
    assert step3.state == StepState.FINISHED

    assert step1.results.int_io.x == 2
    assert step2.results.int_io.x == 4
    assert step3.results.int_io.x == 7


def test_tractor_run_error():
    class TStep(Step[TIOs, TSecretArgs, TResources, TIOs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results.int_io.x= self.inputs.int_io.x + self.args.arg2
            if self.results.int_io.x > 4:
                raise ValueError("Value too high")

    on_error_called = {"count": 0}

    def on_error(step: Step):
        print("on error")
        on_error_called['count'] = 1

    tractor = Tractor(uid='t1')

    step1 = TStep("test-step-1", TSecretArgs(arg1=Secret('1'), arg2=1), TResources(service1=TResource(env='test', uid='res1')), TIOs(int_io=IntIO(x=1)))
    step2 = TStep("test-step-2", TSecretArgs(arg1=Secret('2'), arg2=2), TResources(service1=TResource(env='test', uid='res1')), TIOs(int_io=step1.results.int_io))
    step3 = TStep("test-step-3", TSecretArgs(arg1=Secret('3'), arg2=3), TResources(service1=TResource(env='test', uid='res1')), TIOs(int_io=step2.results.int_io))
    
    tractor.add_step(step1)
    tractor.add_step(step2)
    tractor.add_step(step3)

    with pytest.raises(ValueError):
        tractor.run(on_error=on_error)

    assert step1.state == StepState.FINISHED
    assert step2.state == StepState.FINISHED
    assert step3.state == StepState.ERROR

    assert step3.results.int_io.x == 7
    assert on_error_called == {"count": 1}


def test_tractor_add_subtractor():
    class TStep(Step[TIOs, TSecretArgs, TResources, TIOs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TIOs(int_io=IntIO(x=self.inputs.int_io.x + self.args.arg2))
            if self.results.int_io.x > 4:
                raise ValueError("Value too high")

    on_error_called = {"count": 0}

    def on_error(step: Step):
        print("on error")
        on_error_called['count'] = 1

    tractor = Tractor(uid='t1')
    step1 = TStep("test-step-1", TSecretArgs(arg1=Secret('1'), arg2=1), TResources(service1=TResource(env='test', uid='res1')), TIOs(int_io=IntIO(x=1)))
    tractor.add_step(step1)

    tractor2 = Tractor(uid='t2')
    step2 = TStep("test-step-2", TSecretArgs(arg1=Secret('2'), arg2=2), TResources(service1=TResource(env='test', uid='res1')), TIOs(int_io=step1.results.int_io))
    tractor2.add_step(step2)

    tractor.add_step(tractor2)

    tractor.run(on_error=on_error)

    assert step1.state == StepState.FINISHED
    assert step2.state == StepState.FINISHED


def test_tractor_add_subtractor_duplicate_step():
    class TStep(Step[TIOs, TSecretArgs, TResources, TIOs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TIOs(x=self.inputs.int_io.x + self.args.arg2)
            if self.results.int_io.x > 4:
                raise ValueError("Value too high")

    on_error_called = {"count": 0}

    def on_error(step: Step):
        print("on error")
        on_error_called['count'] = 1

    tractor = Tractor(uid='t1')
    step1 = TStep("test-step-1", TSecretArgs(arg1=Secret('1'), arg2=1), TResources(service1=TResource(env='test', uid='res1')), TIOs(int_io=IntIO(x=1)))
    tractor.add_step(step1)

    tractor2 = Tractor(uid='t2')
    step2 = TStep("test-step-2", TSecretArgs(arg1=Secret('2'), arg2=2), TResources(service1=TResource(env='test', uid='res1')), TIOs(int_io=step1.results.int_io))
    tractor2.add_step(step2)
    tractor.add_step(tractor2)

    tractor3 = Tractor(uid='t2')
    tractor3.add_step(step2)
    
    with pytest.raises(DuplicateStepError):
        tractor.add_step(tractor3)


def test_tractor_add_subtractor_duplicate_tractor():
    class TStep(Step[TIOs, TSecretArgs, TResources, TIOs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TIOs(int_io=IntIO(x=self.inputs.int_io.x + self.args.arg2))
            if self.results.int_io.x > 4:
                raise ValueError("Value too high")

    on_error_called = {"count": 0}

    def on_error(step: Step):
        print("on error")
        on_error_called['count'] = 1

    tractor = Tractor(uid='t1')
    step1 = TStep("test-step-1", TSecretArgs(arg1=Secret('1'), arg2=1), TResources(service1=TResource(env='test', uid='res1')), TIOs(int_io=IntIO(x=1)))
    tractor.add_step(step1)

    tractor2 = Tractor(uid='t2')
    step2 = TStep("test-step-2", TSecretArgs(arg1=Secret('2'), arg2=2), TResources(service1=TResource(env='test', uid='res1')), TIOs(int_io=step1.results.int_io))
    tractor2.add_step(step2)
    tractor.add_step(tractor2)

    tractor3 = Tractor(uid='t3')
    tractor3.add_step(tractor)
    
    with pytest.raises(DuplicateTractorError):
        tractor.add_step(tractor3)


def test_named_tractor():
    class TStep(Step[TIOs, TArgs, TResources, TIOs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TIOs(int_io=IntIO(x=self.inputs.int_io.x + self.args.arg1))

    class IOTNamedTractor(StepIOs):
        int_io: IntIO = IntIO()

    class ATNamedTractor(StepArgs):
        arg1: int = 10

    class TNamedTractor(NamedTractor,
            nt_steps=[('step1', TStep), ('step2', TStep)],
            nt_inputs=IOTNamedTractor,
            nt_results=IOTNamedTractor,
            nt_args=ATNamedTractor,
            nt_resources=TResources
    ):
        INPUTS_MAP = {
            "step1": {"int_io": NTInput(name="int_io")},
            "step2": {"int_io": ("step1", 'int_io')},
        }
        ARGS_MAP = {
            "step1": {"arg1": "arg1"},
            "step2": {"arg1": "arg1"},
        }
        RESULTS_MAP = {
            "int_io": ("step2", "int_io")
        }
        RESOURCES_MAP = {
            "step1": {"service1": "service1"},
            "step2": {"service1": "service1"},
        }

    nt = TNamedTractor(
        uid='nt1',
        args=TNamedTractor.ArgsModel(arg1=20),
        resources=TNamedTractor.ResourcesModel(service1=TResourceWithSecrets(uid='res1', env='stage', secret='password')),
        inputs=TNamedTractor.InputsModel(int_io=IntIO(x=11))
    )

    nt.run()
    assert nt.steps[0].state == StepState.FINISHED
    assert nt.steps[1].state == StepState.FINISHED
    assert nt.state == StepState.FINISHED
    assert nt.results.int_io.x == 51

def test_multiple_named_tractors():
    class TStep(Step[TIOs, TArgs, NoResources, TIOs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TIOs(int_io=IntIO(x=self.inputs.int_io.x + self.args.arg1))

    class IOTNamedTractor(StepIOs):
        int_io: IntIO = IntIO()

    class ATNamedTractor(StepArgs):
        arg1: int = 10


    class TNamedTractor(NamedTractor, nt_steps=[('step1', TStep), ('step2', TStep)], nt_inputs=IOTNamedTractor, nt_results=IOTNamedTractor, nt_args=ATNamedTractor, nt_resources=NoResources):
        INPUTS_MAP = {
            "step1": {"int_io": NTInput(name="int_io")},
            "step2": {"int_io": ("step1", "int_io")},
        }
        ARGS_MAP = {
            "step1": {"arg1": "arg1"},
            "step2": {"arg1": "arg1"},
        }
        RESULTS_MAP = {
            "int_io": ("step2", "int_io")
        }
        RESOURCES_MAP = {
            "step1": {},
            "step2": {}
        }

    class TNamedTractor2(NamedTractor, nt_steps=[('step1', TStep), ('step2', TStep)], nt_inputs=IOTNamedTractor, nt_results=IOTNamedTractor, nt_args=ATNamedTractor, nt_resources=NoResources):
        INPUTS_MAP = {
            "step1": {"int_io": NTInput(name="int_io")},
            "step2": {"int_io": ("step1", "int_io")},
        }
        ARGS_MAP = {
            "step1": {"arg1": "arg1"},
            "step2": {"arg1": "arg1"},
        }
        RESULTS_MAP = {
            "int_io": ("step2", "int_io")
        }
        RESOURCES_MAP = {
            "step1": {},
            "step2": {}
        }

    nt = TNamedTractor(
        uid='nt1',
        args=TNamedTractor.ArgsModel(arg1=20),
        resources=TNamedTractor.ResourcesModel(service1=TResourceWithSecrets(uid='res1', env='stage', secret='password')),
        inputs=TNamedTractor.InputsModel(int_io=IntIO(x=11))
    )

    nt.run()
    assert nt.steps[0].state == StepState.FINISHED
    assert nt.steps[1].state == StepState.FINISHED
    assert nt.state == StepState.FINISHED
    assert nt.steps[0].results.int_io.x == 31
    assert nt.steps[1].results.int_io.x == 51
    assert nt.results.int_io.x == 51


def test_named_tractor_step_failed():
    class TStep(Step[TIOs, TArgs, NoResources, TIOs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            if self.inputs.int_io.x + self.args.arg1>70:
                raise StepFailedError
            self.results = TIOs(int_io=IntIO(x=self.inputs.int_io.x + self.args.arg1))

    class IOTNamedTractor(StepIOs):
        int_io: IntIO = IntIO()

    class ATNamedTractor(StepArgs):
        arg1: int = 10

    class TNamedTractor(NamedTractor, nt_steps=[('step1', TStep), ('step2', TStep)], nt_inputs=IOTNamedTractor, nt_results=IOTNamedTractor, nt_args=ATNamedTractor, nt_resources=NoResources):
        INPUTS_MAP = {
            "step1": {"int_io": NTInput(name="int_io")},
            "step2": {"int_io": ("step1", "int_io")},
        }
        ARGS_MAP = {
            "step1": {"arg1": "arg1"},
            "step2": {"arg1": "arg1"},
        }
        RESULTS_MAP = {
            "int_io": ("step2", "int_io")
        }
        RESOURCES_MAP = {
            "step1": {},
            "step2": {}
        }

    nt = TNamedTractor(
        uid='nt1',
        args=TNamedTractor.ArgsModel(arg1=50),
        resources=NoResources(),
        inputs=TNamedTractor.InputsModel(int_io=IntIO(x=11))
    )

    nt.run()
    assert nt.steps[0].state == StepState.FINISHED
    assert nt.steps[1].state == StepState.FAILED
    assert nt.state == StepState.FAILED
    assert nt.steps[0].results.int_io.x == 61
    # second step failed, so result should be default one
    assert nt.steps[1].results.int_io.x == 10
    # tractor result as well
    assert nt.results.int_io.x == 10


def test_named_tractor_nested():
    class TStep(Step[TIOs, TArgs, NoResources, TIOs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results.int_io = IntIO(x=self.inputs.int_io.x + self.args.arg1)

    class IOTNamedTractor2(StepIOs):
        int_io: IntIO = IntIO()

    class ATNamedTractor2(StepArgs):
        arg1: int = 10

    class TNamedTractor(NamedTractor, nt_steps=[('step1', TStep), ('step2', TStep)], nt_inputs=IOTNamedTractor2, nt_results=IOTNamedTractor2, nt_args=ATNamedTractor2, nt_resources=NoResources):
        INPUTS_MAP = {
            "step2": {"int_io": ("step1", "int_io")},
            "step1": {"int_io": NTInput(name="int_io")}
        }
        ARGS_MAP = {
            "step1": {"arg1": "arg1"},
            "step2": {"arg1": "arg1"},
        }
        RESULTS_MAP = {
            "int_io": ("step2", "int_io")
        }
        RESOURCES_MAP = {
            "step1": {},
            "step2": {}
        }

    class TNamedTractor2(NamedTractor, nt_steps=[('step1', TStep), ('nt1', TNamedTractor)], nt_inputs=IOTNamedTractor2, nt_results=IOTNamedTractor2, nt_args=ATNamedTractor2, nt_resources=NoResources):
        INPUTS_MAP = {
            "step1": {"int_io": NTInput(name="int_io")},
            "nt1": {"int_io": ("step1", "int_io")}
        }
        ARGS_MAP = {
            "step1": {"arg1": "arg1"},
            "nt1": {"arg1": "arg1"},
        }
        RESULTS_MAP = {
            "int_io": ("nt1", "int_io")
        }
        RESOURCES_MAP = {
            "step1": {},
            "nt1": {}
        }

    nt = TNamedTractor2(
        uid='nt1',
        args=TNamedTractor2.ArgsModel(arg1=20),
        resources=TNamedTractor2.ResourcesModel(),
        inputs=TNamedTractor2.InputsModel(int_io=IntIO(x=10))
    )
    nt.run()
    assert nt.steps[0].state == StepState.FINISHED
    assert nt.steps[1].state == StepState.FINISHED
    assert nt.steps[0].results.int_io.x == 30
    assert nt.steps[1].results.int_io.x == 70
    assert nt.results.int_io.x == 70


def test_named_tractor_nested_dump(fixture_isodate_now):
    class TStep(Step[TIOs, TArgs, NoResources, TIOs, TDetails]):
        NAME: ClassVar[str] = "TestStep"
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results.int_io = IntIO(x=self.inputs.int_io.x + self.args.arg1)

    class IOTNamedTractor2(StepIOs):
        int_io: IntIO = IntIO()

    class ATNamedTractor2(StepArgs):
        arg1: int = 10

    class TNamedTractor(NamedTractor, nt_steps=[('step1', TStep), ('step2', TStep)], nt_inputs=IOTNamedTractor2, nt_results=IOTNamedTractor2, nt_args=ATNamedTractor2, nt_resources=NoResources):
        INPUTS_MAP = {
            "step2": {"int_io": ("step1", "int_io")},
            "step1": {"int_io": NTInput(name="int_io")}
        }
        ARGS_MAP = {
            "step1": {"arg1": "arg1"},
            "step2": {"arg1": "arg1"},
        }
        RESULTS_MAP = {
            "int_io": ("step2", "int_io")
        }
        RESOURCES_MAP = {
            "step1": {},
            "step2": {}
        }

    class TNamedTractor2(NamedTractor, nt_steps=[('step1', TStep), ('nt1', TNamedTractor)], nt_inputs=IOTNamedTractor2, nt_results=IOTNamedTractor2, nt_args=ATNamedTractor2, nt_resources=NoResources):
        INPUTS_MAP = {
            "step1": {"int_io": NTInput(name="int_io")},
            "nt1": {"int_io": ("step1", "int_io")}
        }
        ARGS_MAP = {
            "step1": {"arg1": "arg1"},
            "nt1": {"arg1": "arg1"},
        }
        RESULTS_MAP = {
            "int_io": ("nt1", "int_io")
        }
        RESOURCES_MAP = {
            "step1": {},
            "nt1": {}
        }

    nt = TNamedTractor2(
        uid='nt2',
        args=TNamedTractor2.ArgsModel(arg1=20),
        resources=TNamedTractor2.ResourcesModel(),
        inputs=TNamedTractor2.InputsModel(int_io=IntIO(x=10))
    )
    nt.run()
    assert nt.dump(full=False) == {
      "state": "finished",
      "current_step": "nt2::nt1",
      "steps": [
        {
          "type": "step",
          "data": {
            "uid": "nt2::step1",
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
            },
            "results": {
                "int_io":{ "x": 30}
            }
          }
        },
        {
          "type": "tractor",
          "data": {
            "current_step": "TestStep:nt2::nt1::step2",
            "state": "finished",
            "steps": [{
                "type": "step",
                "data": {
                  "uid": "nt2::nt1::step1",
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
                    "arg1": 20
                  },
                  "type": "TestStep",
                  "inputs": {},
                  "inputs_standalone": {
                    "int_io": {"x": 30}
                  },
                  "results": {
                      "int_io":{"x": 50}
                  }
                }
              },
              {
                "type": "step",
                "data": {
                  "uid": "nt2::nt1::step2",
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
                    "arg1": 20
                  },
                  "type": "TestStep",
                  "inputs": {},
                  "inputs_standalone": {
                    "int_io": {"x": 50}
                  },
                  "results": {
                      "int_io":{"x": 70}
                  }
                }
              }
            ],
            "resources": {},
            "uid": "nt2::nt1"
          }
        }
      ],
      "resources": {},
      "uid": "nt2"
    }


class STMD_TStep(Step[TIOs, TArgs, NoResources, TIOs, TDetails]):
    NAME: ClassVar[str] = "TestStep"
    def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
        self.results = TIOs(int_io=IntIO(x=self.inputs.int_io.x + self.args.arg1))

class IOSTMD_TNamedTractor(StepIOs):
    int_io: IntIO = IntIO()

class STMD_TNamedTractor(NamedTractor,
        nt_steps=[('step1', STMD_TStep), ('step2', STMD_TStep)],
        nt_inputs=IOSTMD_TNamedTractor,
        nt_args=TArgs,
        nt_resources=NoResources,
        nt_results=IOSTMD_TNamedTractor
):
        INPUTS_MAP = {
            "step2": {"int_io": ("step1", "int_io")},
            "step1": {"int_io": NTInput(name="int_io")}
        }
        ARGS_MAP = {
            "step1": {"arg1": "arg1"},
            "step2": {"arg1": "arg1"},
        }
        RESULTS_MAP = {
            "int_io": ("step2", "int_io")
        }
        RESOURCES_MAP = {
            "step1": {},
            "step2": {}
        }


def test_stmd():

    class TSTMD(STMD, tractor_type=STMD_TNamedTractor):
       pass

    tstmd = TSTMD('test-stmd-1',
        args=TSTMD.ArgsModel(arg1=10, tractors=10),
        resources=TSTMD.ResourcesModel(),
        inputs=TSTMD.InputsModel(
            multidata=TSTMD.MultiDataInput(
                inputs=[
                      STMD_TNamedTractor.InputsModel(int_io=IntIO(x=10)),
                      STMD_TNamedTractor.InputsModel(int_io=IntIO(x=20)),
                ]
            )
        )
    )
    tstmd.run()


