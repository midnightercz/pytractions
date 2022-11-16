from typing import List, TypeVar, Generic, TypedDict, ClassVar, Type, Optional, get_args
from unittest import mock

import pydantic
import pytest

from pytraction.traction import (
    StepIOs, StepIO, StepArgs, NoInputs, NoResources,
    ExtResources, StepOnUpdateCallable, StepErrors, StepDetails,
    ExtResource, NoArgs,
    StepFailedError, Tractor, Secret,
    StepOnErrorCallable, StepOnUpdateCallable,
    TractorDumpDict, NoDetails, StepState, StepStats,
    NamedTractor, NTInput, STMD)


from pytraction.exc import (LoadWrongStepError, LoadWrongExtResourceError, MissingSecretError, DuplicateStepError, DuplicateTractorError)

from .models import (
    IntIO)

def test_inputs_invalid_field_type():
    with pytest.raises(ValueError):
        class TInputs(StepIOs):
            input1: int = 10

def test_inputs_missing_default_value():
    with pytest.raises(TypeError):

        class TInput(StepIO):
            x: IntIO

        class TInputs(StepIOs):
            input1: TInput

def test_inputs_reference():
    class TResults(StepIOs):
        x: IntIO = IntIO()

    class TInputs(StepIOs):
        input1: IntIO = IntIO()

    results = TResults()
    results.x.x = 20
    inputs = TInputs(input1=results.x)
    print(inputs.input1._input_mode)
    results.x.x=40
    assert inputs.input1.x == 40


# def test_inputs_missing_field():
    # class TResults(StepIOs):
        # x: IntIO = IntIO()

    # class TInputs(StepIOs):
        # input1: IntIO = IntIO()

    # results = TResults(x=30)
    # with pytest.raises(pydantic.ValidationError):
        # inputs = TInputs()


def test_inputs_missing_annotation():
    with pytest.raises(TypeError):
        class TInputs(StepIOs):
            x = 10


def test_inputs_wrong_type():
    class TResults(StepIOs):
        x: IntIO = IntIO()

    class TInputs(StepIOs):
        input1: IntIO = IntIO()

    with pytest.raises(TypeError):
        inputs = TInputs(input1=10)
