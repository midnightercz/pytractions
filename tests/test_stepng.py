
from typing import List, TypeVar, Generic, TypedDict, ClassVar, Type, Optional, get_args

import pydantic
import pytest

import pytraction

from pytraction.traction import (
    StepIO, StepIOs, StepArgs, NoInputs, NoResources,
    ExtResources, StepOnUpdateCallable, StepErrors, StepDetails,
    ExtResource, NoArgs,
    StepFailedError, Tractor, Secret,
    StepOnErrorCallable, StepOnUpdateCallable,
    Step)

from pytraction.exc import (LoadWrongStepError, LoadWrongExtResourceError, MissingSecretError, DuplicateStepError, DuplicateTractorError)


from .models import (
    TIOs, TArgs, TResources, TSecretArgs, TResource, TResourceWithSecrets, TResource, TResourcesWithSecrets, TResources2,
    TIOs, TDetails)



def test_step_ng():
    class TStep(Step):
        results: TIOs
        inputs: TIOs
        args: TArgs
        resources: TResources
        details: TDetails

        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass
         
    TStep(uid="test-step-1", args={"arg1": 1}, resources={"service1": TResource(uid='res1', env="test")}, inputs=TIOs())


def test_step_ng_wrong_results_type():
    with pytest.raises(TypeError):
        class TStep(Step):
            results: TArgs
            inputs: TIOs
            args: TArgs
            resources: TResources
            details: TDetails

            def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
                pass

def test_step_ng_wrong_inputs_type():
    with pytest.raises(TypeError):
        class TStep(Step):
            results: TIOs
            inputs: TArgs
            args: TArgs
            resources: TResources
            details: TDetails

            def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
                pass

def test_step_ng_wrong_args_type():
    with pytest.raises(TypeError):
        class TStep(Step, results_type=TIOs, inputs_type=TIOs, args_type=TIOs, resources_type=TResources, details_type=TDetails):
            results: TIOs
            inputs: TIOs
            args: TIOs
            resources: TResources
            details: TDetails

            def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
                pass

def test_step_ng_wrong_resources_type():
    with pytest.raises(TypeError):
        class TStep(Step):
            results: TIOs
            inputs: TIOs
            args: TArgs
            resources: TIOs
            details: TDetails

            def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
                pass

def test_step_ng_wrong_details_type():
    with pytest.raises(TypeError):
        class TStep(Step):
            results: TIOs
            inputs: TIOs
            args: TArgs
            resources: TResources
            details: TIOs

            def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
                pass


def test_step_ng_invalid_inputs():
    class TStep(Step):
        results: TIOs
        inputs: TIOs
        args: TArgs
        resources: TResources
        details: TDetails
        def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pylint: disable=unused-argument
            pass

    with pytest.raises(pydantic.ValidationError):
        TStep(uid="test-step-1", args={"arg1": 1}, resources={"service1": TResource(uid='res1', env="test")}, inputs={"input2": TIOs()})

