from typing import List, TypeVar, Generic, TypedDict, ClassVar, Type, Optional, get_args
from unittest import mock

import pydantic
import pytest

from pytraction.traction import (
    StepNG, StepIOs, StepArgs, NoInputs, NoResources,
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

def test_named_tractor():
    class TStep(StepNG):
        results: TIOs
        args: TArgs
        resources: TResources
        inputs: TIOs
        details: TDetails
        NAME: ClassVar[str] = "TestStep"

        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TIOs(int_io=IntIO(x=self.inputs.int_io.x + self.args.arg1))

    class IOTNamedTractor(StepIOs):
        int_io: IntIO = IntIO()

    class ATNamedTractor(StepArgs):
        arg1: int = 10

    class TNamedTractor(NamedTractor,
            nt_steps=[('step1', TStep), ('step2', TStep)]):
        inputs: IOTNamedTractor
        results: IOTNamedTractor
        args: ATNamedTractor
        resources: TResources

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
        NAME: ClassVar[str] = "TNamedTractor"

    nt = TNamedTractor(
        uid='nt1',
        args=ATNamedTractor(arg1=20),
        resources=TNamedTractor.ResourcesModel(service1=TResourceWithSecrets(uid='res1', env='stage', secret='password')),
        inputs=TNamedTractor.InputsModel(int_io=IntIO(x=11))
    )

    nt.run()
    assert nt.steps[0].state == StepState.FINISHED
    assert nt.steps[1].state == StepState.FINISHED
    assert nt.state == StepState.FINISHED
    assert nt.results.int_io.x == 51

def test_multiple_named_tractors():
    class TStep(StepNG):
        results: TIOs
        args: TArgs
        resources: NoResources
        inputs: TIOs
        details: TDetails
        NAME: ClassVar[str] = "TestStep"

        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results = TIOs(int_io=IntIO(x=self.inputs.int_io.x + self.args.arg1))

    class IOTNamedTractor(StepIOs):
        int_io: IntIO = IntIO()

    class ATNamedTractor(StepArgs):
        arg1: int = 10


    class TNamedTractor(NamedTractor, nt_steps=[('step1', TStep), ('step2', TStep)]):
        inputs: IOTNamedTractor
        results: IOTNamedTractor
        args: ATNamedTractor
        resources: NoResources

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
        NAME: ClassVar[str] = "TNamedTractor"

    class TNamedTractor2(NamedTractor, nt_steps=[('step1', TStep), ('step2', TStep)]):
        inputs: IOTNamedTractor
        results: IOTNamedTractor
        args: ATNamedTractor
        resources: NoResources

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
        NAME: ClassVar[str] = "TNamedTractor2"

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
    class TStep(StepNG):
        results: TIOs
        args: TArgs
        resources: NoResources
        inputs: TIOs
        details: TDetails
        NAME: ClassVar[str] = "TestStep"

        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            if self.inputs.int_io.x + self.args.arg1>70:
                raise StepFailedError
            self.results = TIOs(int_io=IntIO(x=self.inputs.int_io.x + self.args.arg1))

    class IOTNamedTractor(StepIOs):
        int_io: IntIO = IntIO()

    class ATNamedTractor(StepArgs):
        arg1: int = 10

    class TNamedTractor(NamedTractor, nt_steps=[('step1', TStep), ('step2', TStep)]):
        inputs: IOTNamedTractor
        results: IOTNamedTractor
        args: ATNamedTractor
        resources: NoResources

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
        NAME: ClassVar[str] = "TNamedTractor"

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
    class TStep(StepNG):
        results: TIOs
        args: TArgs
        resources: NoResources
        inputs: TIOs
        details: TDetails
        NAME: ClassVar[str] = "TestStep"

        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results.int_io = IntIO(x=self.inputs.int_io.x + self.args.arg1)

    class IOTNamedTractor2(StepIOs):
        int_io: IntIO = IntIO()

    class ATNamedTractor2(StepArgs):
        arg1: int = 10

    class TNamedTractor(NamedTractor, nt_steps=[('step1', TStep), ('step2', TStep)]):
        inputs: IOTNamedTractor2
        results: IOTNamedTractor2
        args: ATNamedTractor2
        resources: NoResources

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
        NAME: ClassVar[str] = "TNamedTractor"

    class TNamedTractor2(NamedTractor, nt_steps=[('step1', TStep), ('nt1', TNamedTractor)]):
        inputs: IOTNamedTractor2
        results: IOTNamedTractor2
        args: ATNamedTractor2
        resources: NoResources

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
        NAME: ClassVar[str] = "TNamedTractor2"

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
    class TStep(StepNG):
        results: TIOs
        args: TArgs
        resources: NoResources
        inputs: TIOs
        details: TDetails
        NAME: ClassVar[str] = "TestStep"

        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            self.results.int_io = IntIO(x=self.inputs.int_io.x + self.args.arg1)

    class IOTNamedTractor2(StepIOs):
        int_io: IntIO = IntIO()

    class ATNamedTractor2(StepArgs):
        arg1: int = 10

    class TNamedTractor(NamedTractor, nt_steps=[('step1', TStep), ('step2', TStep)]):
        inputs: IOTNamedTractor2
        results: IOTNamedTractor2
        args: ATNamedTractor2
        resources: NoResources

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
        NAME: ClassVar[str] = "TNamedTractor"

    class TNamedTractor2(NamedTractor, nt_steps=[('step1', TStep), ('nt1', TNamedTractor)]):
        inputs: IOTNamedTractor2
        results: IOTNamedTractor2
        args: ATNamedTractor2
        resources: NoResources

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
        NAME: ClassVar[str] = "TNamedTractor2"

    nt = TNamedTractor2(
        uid='nt2',
        args=TNamedTractor2.ArgsModel(arg1=20),
        resources=TNamedTractor2.ResourcesModel(),
        inputs=TNamedTractor2.InputsModel(int_io=IntIO(x=10))
    )
    nt.run()
    assert nt.dump(full=False) == {
      "state": "finished",
      "current_step": "TNamedTractor[nt2:2.nt1]",
      'results': {'int_io': {'x': 70}},
      "steps": [
        {
          "type": "stepng",
          "data": {
            "uid": "nt2:1.step1",
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
            "inputs_standalone": {'int_io': {'x': 10}},
            "results": {
                "int_io":{ "x": 30}
            }
          }
        },
        {
          "type": "tractor",
          "data": {
            "current_step": "TestStep[nt2:2.nt1:2.step2]",
            "state": "finished",
            'results': {'int_io': {'x': 70}},
            "steps": [{
                "type": "stepng",
                "data": {
                  "uid": "nt2:2.nt1:1.step1",
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
                  "inputs": {'int_io': ('TestStep[nt2:1.step1]', 'int_io')},
                  "inputs_standalone": {},
                  "results": {
                      "int_io":{"x": 50}
                  }
                }
              },
              {
                "type": "stepng",
                "data": {
                  "uid": "nt2:2.nt1:2.step2",
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
                  "inputs": {"int_io": ('TestStep[nt2:2.nt1:1.step1]', 'int_io')},
                  "inputs_standalone": {},
                  "results": {
                      "int_io":{"x": 70}
                  }
                }
              }
            ],
            "resources": {},
            "uid": "nt2:2.nt1"
          }
        }
      ],
      "resources": {},
      "uid": "nt2"
    }


class STMD_TStep(StepNG):
    results: TIOs
    args: TArgs
    resources: NoResources
    inputs: TIOs
    details: TDetails
    NAME: ClassVar[str] = "TestStep"

    def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
        self.results = TIOs(int_io=IntIO(x=self.inputs.int_io.x + self.args.arg1))

class IOSTMD_TNamedTractor(StepIOs):
    int_io: IntIO = IntIO()

class STMD_TNamedTractor(NamedTractor,
        nt_steps=[('step1', STMD_TStep), ('step2', STMD_TStep)]):

    inputs: IOSTMD_TNamedTractor
    args: TArgs
    resources: NoResources
    results: IOSTMD_TNamedTractor
    details: NoDetails

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
    NAME: ClassVar[str] = "STMD_TNamedTractor"


def test_stmd():
    class TSTMD(STMD, tractor_type=STMD_TNamedTractor):
        NAME: ClassVar[str] = "TSMTD"

    tstmd = TSTMD('test-stmd-1',
        args=TSTMD.ArgsModel(arg1=10, tractors=10),
        resources=TSTMD.ResourcesModel(),
        inputs=TSTMD.InputsModel(data=TSTMD.InputsModel.ListModel(list_data=[IOSTMD_TNamedTractor(int_io=IntIO(x=10))]))
        )
    tstmd.run()


