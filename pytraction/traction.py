import abc
import copy
import datetime
from dataclasses import make_dataclass, asdict
from functools import partial
import inspect

from typing import (
    Dict,
    List,
    Callable,
    Optional,
    TypedDict,
    Any,
    Type,
    Sequence,
    Generic,
    TypeVar,
    Protocol,
    Mapping,
    ClassVar,
    cast,
    get_args,
    Tuple,
    NamedTuple,
    Union,
    NewType,
    runtime_checkable,
    Callable,
    Iterator,
    _GenericAlias,
    Union,
    ForwardRef
)
import typing_inspect


import enum
from dataclasses_json import dataclass_json
import pydantic
import pydantic.generics
import pydantic.fields
import pydantic.main
from pydantic.dataclasses import dataclass

from .exc import LoadWrongStepError, LoadWrongExtResourceError, MissingSecretError, DuplicateStepError, DuplicateTractorError


Validator = Callable[Any, Any]


def empty_on_error_callback() -> None:
    return None


def isodate_now() -> str:
    """Return current datetime in iso8601 format."""
    return "%s%s" % (datetime.datetime.utcnow().isoformat(), "Z")


class StepFailedError(Exception):
    """Exception indidating failure of a step."""


class Secret:
    """Class for storing sensitive values used as argument for Step class."""

    value: str

    @classmethod
    def __get_validators__(cls) -> Iterator[Validator]:
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not isinstance(v, cls):
            raise TypeError("Secret required")
        return v

    def __init__(self, val: str):
        """Init secret instance."""
        self.value = val

    def __str__(self) -> str:
        return self.value

    def __eq__(self, other):
        if isinstance(other, Secret):
            return self.value == other.value
        else:
            return self.value == other


class StepState(str, enum.Enum):
    """Enum-like class to store step state."""

    READY = 'ready'
    PREP = 'prep'
    RUNNING = 'running'
    FINISHED = 'finished'
    FAILED = 'failed'
    ERROR = 'error'


class StepArgs(pydantic.generics.GenericModel, validate_assignment=True):
    """Class for Step arguments.

    Modified pydantic BaseModel which returns string *CENSORED* for Secret
    arguments on dict() method.
    """

    def dict(
        self,
        *,
        include: Union["AbstractSetIntStr", "MappingIntStrAny"] = None,
        exclude: Union["AbstractSetIntStr", "MappingIntStrAny"] = None,
        by_alias: bool = False,
        skip_defaults: bool = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ) -> Dict[str, Any]:
        model_mapping = {}
        for k, v in self.__dict__.items():
            if type(v) == Secret:
                model_mapping[k] = (str, "*CENSORED*")
            else:
                model_mapping[k] = (type(v), v)
        m = pydantic.create_model(self.__class__.__name__ + "Dump", **model_mapping)
        return pydantic.BaseModel.dict(m())


class ExtResource(pydantic.generics.GenericModel):
    """Step Resource class.

    Use SECRETS class variable to mask attributes at the output.
    """

    NAME: ClassVar[str]
    SECRETS: ClassVar[List[str]] = []
    uid: str

    def dump(self):
        ret = self.dict()
        for k in ret:
            if k in self.SECRETS:
                ret[k] = "*CENSORED*"
        ret["type"] = self.NAME
        return ret

    def load(self, dump):
        """Load resource configuration from dump.

        Dump has to be same resource type. Secret values are
        not loaded from the dump object as they are originaly
        produced as *CENSORED* string.
        """
        dump_copy = dump.copy()
        if dump_copy["type"] != self.NAME:
            raise LoadWrongExtResourceError(
                "Cannot load %s into %s" % (dump_copy["type"], self.NAME)
            )
        dump_copy.pop("type")
        parsed = self.parse_obj(dump)
        for f in self.__fields__:
            if f in self.SECRETS:
                continue
            setattr(self, f, getattr(parsed, f))
        return self

    @classmethod
    def load_cls(cls, dump, secrets: Dict[str, str] = None):
        """Create new resource instance from dump object.

        attributes listed in cls.SECRETS have to be provided as secrets parameter.
        """
        _secrets = secrets or {}
        dump_copy = dump.copy()
        if dump["type"] != cls.NAME:
            raise LoadWrongExtResourceError(
                "Cannot load %s into %s" % (dump_copy["type"], cls.NAME)
            )
        dump_copy.pop("type")
        for secret in cls.SECRETS:
            try:
                dump_copy[secret] = _secrets[secret]
            except KeyError as e:
                raise MissingSecretError(secret) from e
        return cls.parse_obj(dump_copy)

    @property
    def fullname(self) -> str:
        """Full name of class instance."""
        return "%s:%s" % (self.NAME, self.uid)


class ResourcesModelMeta(pydantic.main.ModelMetaclass):
    """Metaclass to ensure only ExtResource class attributes are defined."""

    def __new__(cls, name, bases, attrs):
        ret = super().__new__(cls, name, bases, attrs)
        for k, v in ret.__fields__.items():
            if not issubclass(v.type_, ExtResource):
                raise ValueError("%s has to be type ExtResource" % k)
        return ret


class ExtResources(pydantic.generics.BaseModel, metaclass=ResourcesModelMeta):
    NAME: ClassVar[str]

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if key == "uid":
                continue
            if not issubclass(type(val), ExtResource):
                raise ValueError(
                    "%s has to be type ExtResource not %s" % (key, type(val))
                )

        super().__init__(**kwargs)

    def dump(self, full=True):
        ret = {}
        ret["type"] = self.NAME
        for f in self.__fields__:
            if full:
                ret[f] = getattr(self, f).dump()
            else:
                ret[f] = getattr(self, f).fullname
        return ret

    def load(self, dump):
        dump_copy = dump.copy()
        if dump_copy["type"] != self.NAME:
            raise LoadWrongExtResourceError(
                "Cannot load %s into %s" % (dump_copy["type"], self.NAME)
            )
        dump_copy.pop("type")
        for key, val in dump_copy.items():
            setattr(self, key, getattr(self, key).load(val))  # , secrets=res_secrets))
        return self

    @classmethod
    def load_cls(cls, dump, secrets: Dict[str, str] = None):
        _secrets = secrets or {}
        dump_copy = dump.copy()
        if dump["type"] != cls.NAME:
            raise LoadWrongExtResourceError(
                "Cannot load %s into %s" % (dump_copy["type"], cls.NAME)
            )
        dump_copy.pop("type")
        ret = {}
        for key, val in dump_copy.items():
            res_secrets = _secrets.get("%s:%s" % (val["type"], val["uid"]), {})
            ret[key] = cls.__fields__[key].type_.load_cls(val, secrets=res_secrets)
        return cls(**ret)


ArgsType = TypeVar("ArgsType", bound=StepArgs)
ExtResourcesType = TypeVar("ExtResourcesType", bound=ExtResources)


class DefaultsModelMeta(pydantic.main.ModelMetaclass):
    """Metaclass to ensure all defined attributes in StepResults have default value."""

    def __new__(cls, name, bases, attrs):
        annotations = attrs.get("__annotations__", {})
        for attrk, attrv in attrs.items():
            if attrk in ["dump", "load", "Config", "dict"]:
                continue
            if attrk.startswith("__"):
                continue
            if inspect.ismethod(attrv):
                continue
            if inspect.ismethoddescriptor(attrv):
                continue
            if attrk not in annotations:
                raise TypeError("%s has to be annotated" % attrk)
        for annotated in annotations:
            if annotated not in attrs:
                raise TypeError("Attribute %s is missing default value" % annotated)

        return super().__new__(cls, name, bases, attrs)


class RequiredDefaultsModel(
    pydantic.generics.GenericModel, metaclass=DefaultsModelMeta
):
    class Config:
        validate_assignment = True

    pass


class StepResults(RequiredDefaultsModel):
    """Class to store results of step."""

    step: Optional["Step"] = pydantic.Field(default=None, repr=False)

    def __setattr__(self, key, value):
        """Override setattr to make sure assigning to step.results doesn't break
        existing references for step.results"""

        if key == "step" and value:
            dict_without_original_value = {
                k: v for k, v in self.__dict__.items() if k != key
            }
            self.__fields__["step"].validate(
                value, dict_without_original_value, loc=key
            )
            object.__setattr__(self, key, value)
        else:
            super().__setattr__(key, value)

    def dict(
        self,
        *,
        include: Union["AbstractSetIntStr", "MappingIntStrAny"] = None,
        exclude: Union["AbstractSetIntStr", "MappingIntStrAny"] = None,
        by_alias: bool = False,
        skip_defaults: bool = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ):
        return super().dict(exclude={"step"})


class StepDetails(RequiredDefaultsModel):
    """Class to store step details to."""

    pass


class StepInputs(pydantic.BaseModel, validate_assignment=True):
    @pydantic.validator("*", pre=True)
    def valid_fields(cls, v):
        if not isinstance(v, StepResults):
            raise ValueError("field must be StepResults subclass, but is %s" % type(v))
        return v

    def __setattr__(self, key, value):
        """Override setattr to make sure assigning to step.results doesn't break
        existing references for step.results"""

        dict_without_original_value = {
            k: v for k, v in self.__dict__.items() if k != key
        }
        self.__fields__[key].validate(value, dict_without_original_value, loc=key)
        object.__setattr__(self, key, value)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        for k, v in data.items():
            setattr(self, k, v)


class NoInputs(StepInputs):
    pass


ResultsType = TypeVar("ResultsType", bound=StepResults)
InputsType = TypeVar("InputsType", bound=StepInputs)
DetailsType = TypeVar("DetailsType", bound=StepDetails)


class StepErrors(pydantic.BaseModel):
    """Class to store results of step."""

    errors: Dict[Any, Any] = {}


class StepStats(pydantic.BaseModel):
    started: Optional[str]
    finished: Optional[str]
    skip: bool
    skip_reason: Optional[str]
    skipped: bool
    state: StepState


class StepDumpStats(TypedDict):
    started: Optional[str]
    finished: Optional[str]
    skip: bool
    skip_reason: Optional[str]
    skipped: bool
    state: str


class StepDict(TypedDict):
    name: str
    step_kwargs: Dict[str, Any]
    uid: str
    details: Dict[Any, Any]
    stats: StepStats
    results: Dict[Any, Any]
    errors: Dict[Any, Any]


@dataclass
class StepDumpDict(Generic[ResultsType]):
    name: str
    step_kwargs: Dict[str, Any]
    uid: str
    details: Dict[Any, Any]
    stats: StepDumpStats
    results: ResultsType
    errors: Dict[Any, Any]


StepOnUpdateCallable = Optional[Callable[["Step"], None]]
StepOnErrorCallable = Optional[Callable[["Step"], None]]

def get_step_types(cls):
    type_args = get_args(cls.__orig_bases__[0])  # type: ignore
    stack = [cls]
    item = None
    while stack:
        item = stack.pop(0)

        if hasattr(item, "__orig_bases__"):
            for base in item.__orig_bases__:
                stack.insert(0, base)
        if hasattr(item, "__origin__"):
            if item.__origin__ == Step:
                break
            stack.insert(0, item.__origin__)

    type_args = get_args(item)
    return type_args


class Step(
    pydantic.generics.BaseModel,
    Generic[ResultsType, ArgsType, ExtResourcesType, InputsType, DetailsType],
    validate_all=True,
    allow_population_by_field_name=False,
    extra=pydantic.Extra.forbid,
    underscore_attrs_are_private=False,
    validate_assignment=True,
):
    """Base class for a Step.

    How to use this class: Few things are needed to implement custom step class.
    First, user needs to overwrite _run method which should do include all the code
    which is meant to be do desired step operation.
    In run method, user can access following instance attributes:
    `step_args` - set when Step is initialized. These two variables
        are meant to hold data for the step. User needs to design step to work only
        with data which are json-compatible
    `shared_results` - shared dict-like object where step can store data for
        another steps or load data generated by previously ran steps.
    `external_resources` - resources which are needed for step to run but are not data.
        This can be for example logger, initialized client for external service and
        similar
    `results` - `StepResults` instance used to store data generated by the step.

    To provide detailed info about step status, step can store these details in _details
    attribute of the step instance. Details can contain anything json compatible. To
    set details to initial state, user needs to overwrite _init_details.
    Later, update_details method can be used to updating the details.
    User can overwrite `_pre_run` method to do any kind of 'lazy' preparation of data or
    set `skip` and `skip_reason` variables in the instance to prevent step from the
    execution.
    When there's data error or wrong data are provided, _run method can
    raise StepFailedError to set step to 'failed' state. Failed state indicates
    there's problem with data or configuration and step in this state cannot be
    executed again. If any other exception occurs and is not caught in _run, step is set
    to 'error' state and can be executed again. Only other two states indicating step is
    able to be executed are 'ready' and 'prep'. Ready state is set after initialization
    of the step instance. Step is switched to 'prep' state just before `_pre_run`
    is called. If step is in 'prep' state  `_pre_run` is not called again. After step
    finished the execution, and if there wasn't any error, `results` instance attribute
    is stored to `shared_results`.
    Last thing to do is to set NAME class attribute identify type of step
    """

    NAME: ClassVar[str]
    uid: str
    state: StepState
    skip: bool
    skip_reason: Optional[str]
    results: ResultsType
    errors: StepErrors
    details: DetailsType
    stats: StepStats
    external_resources: Optional[ExtResourcesType]
    inputs: Optional[InputsType]
    args: ArgsType

    def __setattr__(self, key, value):
        """Override setattr to make sure assigning to step.results doesn't break
        existing references for step.results"""

        if key == "results":
            for k in value.__fields__:
                v = getattr(value, k)
                setattr(self.results, k, v)

            super().__setattr__(key, self.results)
        elif key == "inputs":
            dict_without_original_value = {
                k: v for k, v in self.__dict__.items() if k != key
            }
            self.__fields__[key].validate(value, dict_without_original_value, loc=key)
            object.__setattr__(self, key, value)
        else:
            super().__setattr__(key, value)


    def __init__(
        self,
        uid: str,
        step_args: ArgsType,
        external_resources: Optional[ExtResourcesType] = None,
        inputs: Optional[InputsType] = None,
    ):
        """Initilize the step.

        Args:
            uid: (str)
                An unique id for indentifying two steps of the same class
            external_resources: any-object
                Reference for external resources (in any form) which are constant and
                shouldn't be part of the step state or step data
            inputs: dict(str, str)
                Mapping of inputs to results of steps identified by uid
        """

        type_args = get_step_types(self)
        if not type_args:
            raise TypeError(
                "Missing generic annotations for Step class. Use Step[ResultsCls, ArgsCls, ExtResources, InputsCls]"
            )

        results_type = type_args[0]
        args_type = type_args[1]
        resources_type = type_args[2]
        inputs_type = type_args[3]
        details_type = type_args[4]

        if type(step_args) != args_type:
            raise TypeError(
                "Step arguments are not type of %s but %s"
                % (args_type, type(step_args))
            )
        if (
            external_resources is not None
            and type(external_resources) != resources_type
        ):
            raise TypeError(
                "Step external resources are not type of %s but %s"
                % (resources_type, type(external_resources))
            )
        if inputs is not None and type(inputs) != inputs_type:
            raise TypeError(
                "Step inputs are not type of %s but %s" % (inputs_type, type(inputs))
            )

        results = results_type()
        details = details_type()

        stats = StepStats(
            started=None,
            finished=None,
            skip=False,
            skip_reason="",
            skipped=False,
            state=StepState.READY,
        )
        super().__init__(
            uid=uid,
            external_resources=external_resources,
            details=details,
            skip=False,
            skip_reason="",
            state=StepState.READY,
            results=results,
            inputs=inputs or StepInputs(),
            args=step_args,
            errors=StepErrors(),
            stats=stats,
        )

        # override init value copy and set original object via __setattr__
        self.inputs = inputs or StepInputs()

        self.results.step = self

    @property
    def fullname(self) -> str:
        """Full name of class instance."""
        return "%s:%s" % (self.NAME, self.uid)

    def run(
        self,
        on_update: StepOnUpdateCallable = None,
        on_error: StepOnErrorCallable = None,
    ) -> None:
        """Run the step code.

        Step is expected to run when step state is ready, prep or error. For other
        state running the step code is omitted. If step is in ready state,
        _pre_run method is executed first and state is switched to prep.
        After prep phase finishes, and skip is not set to True, _run method containing
        all the code for running the step is executed.
        After _run finishes, step state is set to failed, error or finished. Statistics
        of step are update and potential results of step are stored in shared data object
        After every change of step state, on_update callback is called if set
        """
        _on_update: StepOnUpdateCallable = lambda step: None
        if on_update:
            _on_update = on_update
        _on_error: StepOnErrorCallable = lambda step: None
        if on_error:
            _on_error = on_error
        self._reset_stats()
        if self.state == StepState.READY:
            self.stats.started = isodate_now()

            self.state = StepState.PREP
            self._pre_run()
            _on_update(self)  # type: ignore
        try:
            if self.state not in (StepState.PREP, StepState.ERROR):
                return
            if not self.skip:
                self.state = StepState.RUNNING
                _on_update(self)  # type: ignore
                print("Step run")
                self._run(on_update=_on_update)
        except StepFailedError:
            self.state = StepState.FAILED
        except Exception:
            self.state = StepState.ERROR
            _on_error(self)
            raise
        else:
            print("step finished")
            self.state = StepState.FINISHED
        finally:
            self._finish_stats()
            _on_update(self)  # type: ignore

    def _pre_run(self) -> None:
        """Execute code needed before step run.

        In this method, all neccesary preparation of data can be done.
        It can be also used to determine if step should run or not by setting
        self.skip to True and providing self.skip_reason string with explanation.
        """
        pass

    def _reset_stats(self) -> None:
        self.stats = StepStats(
            started=None,
            finished=None,
            skip=self.skip,
            skip_reason=self.skip_reason,
            skipped=False,
            state=self.state,
        )

    def _finish_stats(self) -> None:
        self.stats.finished = isodate_now()
        self.stats.skipped = self.skip
        self.stats.skip = self.skip
        self.stats.skip_reason = self.skip_reason
        self.stats.state = self.state

    @abc.abstractmethod
    def _run(self, on_update: StepOnUpdateCallable = None) -> None:  # pragma: no cover
        """Run code of the step.

        Method expects raise StepFailedError if step code fails due data error
        (incorrect configuration or missing/wrong data). That ends with step
        state set to failed.
        If error occurs due to uncaught exception in this method, step state
        will be set to error
        """
        raise NotImplementedError

    def dump(self, full=True) -> dict[str, Any]:
        """Dump step data into json compatible complex dictionary."""
        ret = self.dict(exclude={"inputs", "results"})
        ret["type"] = self.NAME
        ret["inputs"] = {}
        ret["inputs_standalone"] = {}
        ret["results"] = self.results.dict(exclude={"step"})
        ret["external_resources"] = self.external_resources.dump(full=full)
        for f, ftype in self.inputs.__fields__.items():
            field = getattr(self.inputs, f)
            if field.step:
                ret["inputs"][f] = getattr(self.inputs, f).step.fullname
            else:
                ret["inputs_standalone"][f] = getattr(self.inputs, f).dict(
                    exclude={"step"}
                )
        return ret

    def load(self, step_dump, secrets: Dict[str, Dict[str, str]] = None):
        """Load step data from dictionary produced by dump method."""
        if step_dump["type"] != self.NAME:
            raise LoadWrongStepError(
                "Cannot load %s dump to step %s" % (step_dump["type"], self.NAME)
            )

        self.details = self.details.parse_obj(step_dump["details"])
        self.skip = step_dump["skip"]
        self.skip_reason = step_dump["skip_reason"]
        self.state = step_dump["state"]
        self.results = self.results.parse_obj(step_dump["results"])
        loaded_args = {}
        for f in self.args.__fields__:
            if isinstance(getattr(self.args, f), Secret):
                loaded_args[f] = getattr(self.args, f)
            else:
                loaded_args[f] = step_dump["args"][f]

        self.args = self.args.parse_obj(loaded_args)
        self.errors = step_dump["errors"]
        self.stats = step_dump["stats"]
        self.results.step = self
        self.external_resources.load(
            step_dump["external_resources"]
        )  # , secrets=_secrets)

    @classmethod
    def load_cls(
        cls,
        step_dump,
        inputs_map: Dict[str, "Step[Any, Any, Any, Any, Any]"],
        external_resources: Optional[ExtResourcesType] = None,
        secrets: Dict[str, Dict[str, str]] = None,
    ):
        """Load step data from dictionary produced by dump method."""

        if step_dump["type"] != cls.NAME:
            raise LoadWrongStepError(
                "Cannot load %s dump to step %s" % (step_dump["type"], cls.NAME)
            )

        _secrets = secrets or {}

        type_args = get_args(cls.__orig_bases__[0])  # type: ignore
        args_type = type_args[1]
        inputs_type = type_args[3]
        external_resources_type = type_args[2]

        loaded_args = {}
        for f, ftype in args_type.__fields__.items():
            if ftype.type_ == Secret:
                try:
                    loaded_args[f] = Secret(
                        _secrets["%s:%s" % (cls.NAME, step_dump["uid"])][f]
                    )
                except KeyError as e:
                    raise MissingSecretError(f) from e
            else:
                loaded_args[f] = step_dump["args"][f]

        args = args_type.parse_obj(loaded_args)

        loaded_inputs = {}
        for iname, itype in step_dump["inputs"].items():
            loaded_inputs[iname] = inputs_map[itype].results
        for iname in step_dump["inputs_standalone"]:
            itype = inputs_type.__fields__[iname]
            loaded_result = itype.type_()
            for rfield in itype.type_.__fields__:
                if rfield == "step":
                    continue
                setattr(
                    loaded_result, rfield, step_dump["inputs_standalone"][iname][rfield]
                )
            loaded_inputs[iname] = loaded_result

        inputs = inputs_type.parse_obj(loaded_inputs)
        if not external_resources:
            _external_resources = external_resources_type.load_cls(
                step_dump["external_resources"], secrets=_secrets
            )
        else:
            _external_resources = external_resources

        ret = cls(step_dump["uid"], args, _external_resources, inputs)
        ret.details = ret.details.parse_obj(step_dump["details"])
        ret.skip = step_dump["skip"]
        ret.skip_reason = step_dump["skip_reason"]
        ret.state = step_dump["state"]
        ret.results = ret.results.parse_obj(step_dump["results"])
        ret.errors = step_dump["errors"]
        ret.stats = step_dump["stats"]
        ret.results.step = ret

        return ret


class TractorDumpDict(TypedDict):
    steps: List[Dict[str, Any]]


class TractorValidateResult(TypedDict):
    missing_inputs: List[Tuple[str, str, str]]
    valid: bool




Tractor= ForwardRef('Tractor')

class Tractor(
    pydantic.BaseModel,
    validate_all=True,
    allow_population_by_field_name=False,
    extra=pydantic.Extra.forbid,
    underscore_attrs_are_private=False,
    validate_assignment=True,
):
    """Class which runs sequence of steps."""

    steps: List[Step[Any, Any, Any, Any, Any]] = []
    current_step: Optional[Union[Step[Any, Any, Any, Any, Any]|Tractor]]
    step_map: Dict[str, Type[Step[Any, Any, Any, Any, Any]]] = {}
    resources_map: Dict[str, Type[ExtResource]] = {}
    uid: str

    def __init__(
        self,
        uid: str,
        #step_map: Dict[str, Type[Step[Any, Any, Any, Any, Any]]],
        #resources_map: Dict[str, Type[ExtResource]],
    ) -> None:
        """Initialize the stepper.

        Args:
            step_map: (mapping of "step-name": <step_class>)
                Mapping of step names to step classes. Used when loading stepper from
                json-compatible dict data
        """
        #step_map = {}
        resource_map = {}
        current_step = None
        super().__init__(
            uid=uid,
            #step_map=step_map,
            current_step=current_step,
            #resources_map=resources_map
        )

    def add_step(self, step: Union[Step[Any, Any, Any, Any, Any]|"Tractor"]) -> None:
        """Add step to step sequence."""
        tractor_stack = [s for s in self.steps if isinstance(s, Tractor)]
        if isinstance(step, Tractor):
            tractor_stack.append(step)
        found_steps = [s.fullname for s in self.steps if isinstance(s, Step)]
        if isinstance(step, Step):
            if step.fullname in found_steps:
                raise DuplicateStepError(step.fullname)
            found_steps.append(step.fullname)

        found_tractors = [self.uid]
        # check for duplicates and loops
        while tractor_stack:
            tractor = tractor_stack.pop(0)
            if tractor.uid in found_tractors:
                raise DuplicateTractorError(tractor.fullname)
            for tstep in tractor.steps:
                if isinstance(tstep, Step):
                    if tstep.fullname in found_steps:
                        raise DuplicateStepError(tstep.fullname)
                    found_steps.append(tstep.fullname)
                else:
                    tractor_stack.append(tstep)
        self.steps.append(step)

    @property
    def fullname(self):
        return self.uid

    def dump(self) -> TractorDumpDict:
        """Dump stepper state and shared_results to json compatible dict."""
        steps: List[Dict[str, Any]] = []
        out: TractorDumpDict = {"steps": steps, "resources": {}, "uid": self.uid}
        resources_dump: Dict[str, Any] = {}
        for step in self.steps:
            if isinstance(step, Step):
                steps.append({"type": "step", "data": step.dump(full=False)})
                for k in step.external_resources.__fields__:
                    res = getattr(step.external_resources, k)
                    resources_dump[res.fullname] = res.dump()
            else:
                steps.append({"type": "tractor", "data": step.dump(full=False)})
                for substep in step.steps:
                    for k in step.external_resources.__fields__:
                        res = getattr(substep.external_resources, k)
                        resources_dump[res.fullname] = res.dump()
        out["resources"] = resources_dump
        return out

    @classmethod
    def load_cls(
            cls, dump_obj: TractorDumpDict, step_map: Dict[str, Type[Step[Any, Any, Any, Any, Any]]], resources_map: Dict[str, Type[ExtResource]], secrets: Dict[str, Dict[str, str]]
    ) -> None:
        ret = cls(dump_obj['uid'])
        ret.load(dump_obj, step_map, resources_map, secrets)
        return ret

    def load(
        self, dump_obj: TractorDumpDict, step_map: Dict[str, Type[Step[Any, Any, Any, Any, Any]]], resources_map: Dict[str, Type[ExtResource]], secrets: Dict[str, Dict[str, str]]
    ) -> None:
        """Load and initialize stepper from data produced by dump method."""
        loaded_steps = {}
        loaded_resources = {}
        self.steps = []
        for fullname, resource_dump in dump_obj["resources"].items():
            resource_dump_copy = resource_dump.copy()
            loaded_resources[fullname] = resources_map[
                resource_dump["type"]
            ].load_cls(
                resource_dump_copy,
                secrets.get(
                    "%s:%s" % (resource_dump["type"], resource_dump["uid"]), {}
                ),
            )

        for step_obj in dump_obj["steps"]:
            if step_obj['type'] == 'step':
                step_resources = {}
                for resource, resource_fullname in step_obj['data']["external_resources"].items():
                    if resource == "type":
                        continue
                    step_resources[resource] = loaded_resources[resource_fullname]
                external_resources_type = get_step_types(step_map[step_obj['data']["type"]])[
                    2
                ]
                external_resources = external_resources_type(**step_resources)
                step = step_map[step_obj['data']["type"]].load_cls(
                    step_obj['data'],
                    loaded_steps,
                    external_resources=external_resources,
                    secrets=secrets,
                )  # .get("%s:%s" % (step_obj['type'], step_obj['uid']), {}))
                loaded_steps[step.fullname] = step
                self.steps.append(step)
            elif step_obj['type'] == 'tractor':
                tractor = self.load_cls(
                    step_obj['data'],
                    step_map,
                    resources_map,
                    secrets.get(step_obj['data']['uid'], {})
                )
                self.steps.append(tractor)


    def run(
        self,
        start_from: int = 0,
        on_error: StepOnErrorCallable = None,
        on_update: StepOnUpdateCallable = None,
    ) -> None:
        """Run the stepper sequence."""
        for step in self.steps[start_from:]:
            self.current_step = step
            step.run(on_update=on_update, on_error=on_error)

Tractor.update_forward_refs()

NTStepsType = TypeVar("NTStepsType", bound=Dict[str, Step])
NTArgsType = TypeVar("NTArgsType", bound=Dict[str, StepArgs])
NTResultsType = TypeVar("NTResultsType", bound=Dict[str, ResultsType])
NTInputsType = TypeVar("NTInputsType", bound=Dict[str, InputsType])

class NamedTractorConfig(pydantic.main.ModelMetaclass, Generic[NTStepsType]):
    def __new__(cls, name, bases, attrs, **kwargs):  # noqa C901
       
        type_args = get_args(cls.__orig_bases__[1])  # type: ignore
        nt_steps = type_args[0]

        nt_args_model = {}
        nt_results_model = {}
        nt_resources_model = {}
        for nt_step_name, nt_step in nt_steps.items():
            nt_step_types = get_step_types(nt_step)
            results_type = nt_step_types[0]
            args_type = nt_step_types[1]
            resources_type = nt_step_types[2]
            inputs_type = nt_step_types[3]
            
            nt_args_model[nt_step_name] = args_type
            nt_results_model[nt_step_name] = results_type
            nt_resources_model[nt_step_name] = resources_type

        attrs["ArgsModel"] = pydantic.create_model("%s_args" % (cls.__name__,), **nt_args_model)
        attrs["ResultsModel"] = pydantic.create_model("%s_results" % (cls.__name__,), **nt_results_model)
        attrs["ResourcesModel"] = pydantic.create_model("%s_resources" % (cls.__name__,), **nt_resources_model)
        ret = super().__new__(cls, name, bases, attrs)


class NamedTractor(pydantic.BaseModel):
    """Class which runs sequence of steps."""

    steps: List[Step[Any, Any, Any, Any, Any]] = []
    current_step: Optional[Union[Step[Any, Any, Any, Any, Any]|Tractor]]
    step_map: Dict[str, Type[Step[Any, Any, Any, Any, Any]]] = {}
    resources_map: Dict[str, Type[ExtResource]] = {}
    uid: str

    def __init__(
        self,
        uid: str,
        #step_map: Dict[str, Type[Step[Any, Any, Any, Any, Any]]],
        #resources_map: Dict[str, Type[ExtResource]],
    ) -> None:
        """Initialize the stepper.

        Args:
            step_map: (mapping of "step-name": <step_class>)
                Mapping of step names to step classes. Used when loading stepper from
                json-compatible dict data
        """

        self.args



        print(type_args)
        #step_map = {}
        resource_map = {}
        current_step = None
        super().__init__(
            uid=uid,
            current_step=current_step,
        )

    @property
    def fullname(self):
        return self.uid

    def dump(self) -> TractorDumpDict:
        """Dump stepper state and shared_results to json compatible dict."""
        steps: List[Dict[str, Any]] = []
        out: TractorDumpDict = {"steps": steps, "resources": {}, "uid": self.uid}
        resources_dump: Dict[str, Any] = {}
        for step in self.steps:
            if isinstance(step, Step):
                steps.append({"type": "step", "data": step.dump(full=False)})
                for k in step.external_resources.__fields__:
                    res = getattr(step.external_resources, k)
                    resources_dump[res.fullname] = res.dump()
            else:
                steps.append({"type": "tractor", "data": step.dump(full=False)})
                for substep in step.steps:
                    for k in step.external_resources.__fields__:
                        res = getattr(substep.external_resources, k)
                        resources_dump[res.fullname] = res.dump()
        out["resources"] = resources_dump
        return out

    @classmethod
    def load_cls(
            cls, dump_obj: TractorDumpDict, step_map: Dict[str, Type[Step[Any, Any, Any, Any, Any]]], resources_map: Dict[str, Type[ExtResource]], secrets: Dict[str, Dict[str, str]]
    ) -> None:
        ret = cls(dump_obj['uid'])
        ret.load(dump_obj, step_map, resources_map, secrets)
        return ret

    def load(
        self, dump_obj: TractorDumpDict, step_map: Dict[str, Type[Step[Any, Any, Any, Any, Any]]], resources_map: Dict[str, Type[ExtResource]], secrets: Dict[str, Dict[str, str]]
    ) -> None:
        """Load and initialize stepper from data produced by dump method."""
        loaded_steps = {}
        loaded_resources = {}
        self.steps = []
        for fullname, resource_dump in dump_obj["resources"].items():
            resource_dump_copy = resource_dump.copy()
            loaded_resources[fullname] = resources_map[
                resource_dump["type"]
            ].load_cls(
                resource_dump_copy,
                secrets.get(
                    "%s:%s" % (resource_dump["type"], resource_dump["uid"]), {}
                ),
            )

        for step_obj in dump_obj["steps"]:
            if step_obj['type'] == 'step':
                step_resources = {}
                for resource, resource_fullname in step_obj['data']["external_resources"].items():
                    if resource == "type":
                        continue
                    step_resources[resource] = loaded_resources[resource_fullname]
                external_resources_type = get_step_types(step_map[step_obj['data']["type"]])[
                    2
                ]
                external_resources = external_resources_type(**step_resources)
                step = step_map[step_obj['data']["type"]].load_cls(
                    step_obj['data'],
                    loaded_steps,
                    external_resources=external_resources,
                    secrets=secrets,
                )  # .get("%s:%s" % (step_obj['type'], step_obj['uid']), {}))
                loaded_steps[step.fullname] = step
                self.steps.append(step)
            elif step_obj['type'] == 'tractor':
                tractor = self.load_cls(
                    step_obj['data'],
                    step_map,
                    resources_map,
                    secrets.get(step_obj['data']['uid'], {})
                )
                self.steps.append(tractor)


    def run(
        self,
        start_from: int = 0,
        on_error: StepOnErrorCallable = None,
        on_update: StepOnUpdateCallable = None,
    ) -> None:
        """Run the stepper sequence."""
        for step in self.steps[start_from:]:
            self.current_step = step
            step.run(on_update=on_update, on_error=on_error)


class NoArgs(StepArgs):
    pass


class NoResources(ExtResources):
    NAME: ClassVar[str] = "NoResources"


class NoDetails(StepDetails):
    pass

class NoResults(StepResults):
    pass
