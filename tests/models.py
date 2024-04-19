from typing import List, TypeVar, Generic, TypedDict, ClassVar, Type, Optional, get_args
from unittest import mock

import pydantic
import pytest

import pytraction

from pytraction.traction import (
    StepIOs, StepIO, StepArgs, NoInputs, NoResources,
    ExtResources, StepOnUpdateCallable, StepErrors, StepDetails,
    ExtResource, NoArgs,
    StepFailedError, Tractor, Secret,
    StepOnErrorCallable, StepOnUpdateCallable,
    TractorDumpDict, NoDetails, StepState, StepStats,
    NamedTractor, NTInput, STMD)

from pytraction.exc import (LoadWrongStepError, LoadWrongExtResourceError, MissingSecretError, DuplicateStepError, DuplicateTractorError)

class IntIO(StepIO):
    x: int = 10

class StrIO(StepIO):
    x: str = ""

class TIOs(StepIOs):
    int_io : IntIO = IntIO()


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


class TDetails(StepDetails):
    status: str = ""
