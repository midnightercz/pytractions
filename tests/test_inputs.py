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


def test_inputs_invalid_field_type():
    with pytest.raises(ValueError):
        class TInputs(StepInputs):
            input1: int


def test_inputs_reference():
    class TResults(StepResults):
        x: int = 10

    class TInputs(StepInputs):
        input1: TResults

    results = TResults(x=30)
    inputs = TInputs(input1=results)
    results.x=40
    assert inputs.input1.x == 40


def test_inputs_missing_field():
    class TResults(StepResults):
        x: int = 10
    class TInputs(StepInputs):
        input1: TResults

    results = TResults(x=30)
    with pytest.raises(pydantic.ValidationError):
        inputs = TInputs()


def test_inputs_missing_annotation():
    with pytest.raises(TypeError):
        class TInputs(StepInputs):
            input1 = 10


def test_inputs_wrong_type():
    class TResults(StepResults):
        x: int = 10

    class TInputs(StepInputs):
        input1: TResults

    with pytest.raises(TypeError):
        inputs = TInputs(input1=10)
