import abc
import copy
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import datetime
from dataclasses import make_dataclass, asdict
from functools import partial
import inspect
import os

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
    ForwardRef,
    get_origin
)
import typing_inspect


import enum
from dataclasses_json import dataclass_json
import pydantic
import pydantic.generics
import pydantic.fields
import pydantic.main
from pydantic.dataclasses import dataclass

from .exc import LoadWrongStepError, LoadWrongExtResourceError, MissingSecretError, DuplicateStepError, DuplicateTractorError, StepFailedError



Validator = Callable[Any, Any]

Step = ForwardRef('Step')

Tractor = ForwardRef('Tractor')

NamedTractor = ForwardRef('NamedTractor')

STMD = ForwardRef('STMD')


def empty_on_error_callback() -> None:
    return None


def isodate_now() -> str:
    """Return current datetime in iso8601 format."""
    return "%s%s" % (datetime.datetime.utcnow().isoformat(), "Z")

def check_type(to_check, expected):
    if get_origin(to_check) == Union:
        final_types = []
        types = [] + list(get_args(to_check))
        while types:
            _type = types.pop(0)
            if get_origin(_type) == Union:
                for uarg in get_args(_type):
                    types.insert(0, uarg)
            elif get_origin(_type) is not None:
                final_types.append(get_origin(_type))
            elif type(_type) == ForwardRef:
                final_types.append(_type._evaluate(globals(),locals(), set()))
            else:
                final_types.append(_type)
        if not any([issubclass(to_check, type_) for type_ in final_types]):
            #raise TypeError(f'"{to_check}" has to be one of {final_types} types not %s' % type(to_check))
            return False
    elif get_origin(to_check):
        if not issubclass(get_origin(to_check), expected):
            #raise TypeError(f'"{to_check}" has to be type "{expected}" not %s' % type(to_check))
            return False
    else:
        if not issubclass(to_check, expected):
            #raise TypeError(f'"{to_check}" has to be type "{expected}" not %s' % type(to_check))
            return False
    return True



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


class StepArgs(pydantic.generics.BaseModel, validate_assignment=True):
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


class ExtResource(pydantic.generics.BaseModel):
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
            if attrk not in annotations:
                raise TypeError("%s has to be annotated" % attrk)
        for annotated in annotations:
            if annotated not in attrs:
                raise TypeError("Attribute %s is missing default value" % annotated)

        return super().__new__(cls, name, bases, attrs)


class RequiredDefaultsModel(
    pydantic.generics.GenericModel, metaclass=DefaultsModelMeta,
):

    class Config:
        #validate_all=True
        #allow_population_by_field_name=False
        extra=pydantic.Extra.forbid
        #underscore_attrs_are_private=False
        validate_assignment=True

class StepResults(RequiredDefaultsModel):
    """Class to store results of step."""

    step: Optional[Step] = pydantic.Field(default=None, repr=False)

    def __setattr__(self, key, value):
        """Override setattr to make sure assigning to step.results doesn't break
        existing references for step.results"""

        if key == "step" and value:
            dict_without_original_value = {
                k: v for k, v in self.__dict__.items() if k != key
            }
            self.__fields__["step"].validate(
                value, dict_without_original_value, loc=key, cls=Step
            )
            object.__setattr__(self, key, value)
        elif key in ["__orig_class__", ]:
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
        return super().dict(exclude={"step","__orig_class__"})


class StepDetails(RequiredDefaultsModel):
    """Class to store step details to."""

    pass


class StepInputsMeta(pydantic.main.ModelMetaclass):
    """Metaclass to ensure all defined attributes in StepResults have default value."""

    def __new__(cls, name, bases, attrs):
        annotations = attrs.get("__annotations__", {})
        for attrk, attrv in attrs.items():
            if attrk.startswith("__"):
                continue
            if inspect.ismethoddescriptor(attrv):
                continue
            if attrk in ('Config',):
                continue
            if attrk not in annotations:
                raise TypeError("%s has to be annotated" % attrk)
        for annotated in annotations:
            ann_cls = annotations[annotated]
            if get_origin(ann_cls):
                ann_cls = get_origin(ann_cls)

            if not issubclass(ann_cls, StepResults):
                raise ValueError("%s attribute as to be StepResults type not %s" % (annotated, annotations[annotated]))

        return super().__new__(cls, name, bases, attrs)


class NoCopyModel(pydantic.generics.BaseModel):
    class Config:
        validate_all = True
        #allow_population_by_field_name=False
        extra = pydantic.Extra.forbid
        #underscore_attrs_are_private=False
        validate_assignment = True
        copy_on_model_validation = 'none'
        use_enum_values = True

    def check_type(self, key, value):
        if key in self.__fields__:
            if get_origin(self.__fields__[key].type_) == Union:
                final_types = []
                types = [] + list(get_args(self.__fields__[key].type_))
                while types:
                    _type = types.pop(0)
                    if get_origin(_type) == Union:
                        for uarg in get_args(_type):
                            types.insert(0, uarg)
                    elif get_origin(_type) is not None:
                        final_types.append(get_origin(_type))
                    elif type(_type) == ForwardRef:
                        final_types.append(_type._evaluate(globals(),locals(), set()))
                    else:
                        final_types.append(_type)
                if not any([isinstance(value, type_) for type_ in final_types]):
                    raise TypeError(f'"{self.__class__.__name__}->{key}" has to be one of {final_types} types not %s' % type(value))
            else:
                if not isinstance(value, self.__fields__[key].type_):
                    raise TypeError(f'"{self.__class__.__name__}->{key}" has to be type "{self.__fields__[key].type_}" not %s' % type(value))

    def __setattr__(self, key, value):
        """Override setattr to make sure assigning to step.results doesn't break
        existing references for step.results"""
   
        dict_without_original_value = {
            k: v for k, v in self.__dict__.items() if k != key
        }
        self.check_type(key, value)
        try:
            self.__fields__[key].validate(value, dict_without_original_value, loc=key)
        except KeyError:
            raise ValueError(f'"{self.__class__.__name__}" object has no field "{key}"')
        object.__setattr__(self, key, value)

    def __init__(self, **data: Any) -> None:
        for k, v in data.items():
            self.check_type(k, v)
        super().__init__(**data)
        for k, v in data.items():
           setattr(self, k, v)


class StepInputs(NoCopyModel, metaclass=StepInputsMeta):
    @pydantic.validator("*", pre=True)
    def valid_fields(cls, v):
        if not isinstance(v, StepResults):
            raise ValueError("field must be StepResults subclass, but is %s" % type(v))
        return v


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
    skipped: bool


class StepDumpStats(TypedDict):
    started: Optional[str]
    finished: Optional[str]
    skipped: bool


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
    #type_args = get_args(cls.__orig_bases__[0])  # type: ignore
    stack = [cls]
    item = None
    while stack:
        item = stack.pop(0)
        if hasattr(item, "__orig_bases__"):
            for base in item.__orig_bases__:
                if hasattr(item, "__args__") and item.__args__:
                    stack.insert(0, base[item.__args__])
                else:
                    stack.insert(0, base)
        if hasattr(item, "__origin__"):
            if item.__origin__ == Step:
                break
            stack.insert(0, item.__origin__)
    type_args = get_args(item)
    return type_args


class BaseTractionModel(pydantic.generics.BaseModel):
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


class Step(
    BaseTractionModel,
    Generic[ResultsType, ArgsType, ExtResourcesType, InputsType, DetailsType],
    validate_all=True,
    allow_population_by_field_name=False,
    extra=pydantic.Extra.forbid,
    underscore_attrs_are_private=False,
    validate_assignment=True,
):
    NAME: ClassVar[str]
    uid: str
    state: StepState
    skip: bool
    skip_reason: Optional[str]
    results: ResultsType
    errors: StepErrors
    details: DetailsType
    stats: StepStats
    resources: Optional[ExtResourcesType]
    inputs: Optional[InputsType]
    args: ArgsType


    def __init__(
        self,
        uid: str,
        args: ArgsType,
        resources: Optional[ExtResourcesType] = None,
        inputs: Optional[InputsType] = None,
    ):
        """Initilize the step.

        Args:
            uid: (str)
                An unique id for indentifying two steps of the same class
            resources: any-object
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

        if not issubclass(inputs_type, StepInputs):
            raise TypeError(
                "Step inputs type has to be subclass of StepInputs"
            )

        if type(args) != args_type:
            raise TypeError(
                "Step arguments are not type of %s but %s"
                % (args_type, type(args))
            )
        if (
            resources is not None
            and type(resources) != resources_type
        ):
            raise TypeError(
                "Step external resources are not type of %s but %s"
                % (resources_type, type(resources))
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
            skipped=False,
        )
        super().__init__(
            uid=uid,
            resources=resources,
            details=details,
            skip=False,
            skip_reason="",
            state=StepState.READY,
            results=results,
            inputs=inputs or StepInputs(),
            args=args,
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
                self._run(on_update=_on_update)
        except StepFailedError:
            self.state = StepState.FAILED
        except Exception:
            self.state = StepState.ERROR
            _on_error(self)
            raise
        else:
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
            skipped=False,
        )

    def _finish_stats(self) -> None:
        self.stats.finished = isodate_now()
        self.stats.skipped = self.skip

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
        ret["resources"] = self.resources.dump(full=full)
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
        self.resources.load(
            step_dump["resources"]
        )  # , secrets=_secrets)

    @classmethod
    def load_cls(
        cls,
        step_dump,
        inputs_map: Dict[str, "Step[Any, Any, Any, Any, Any]"],
        resources: Optional[ExtResourcesType] = None,
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
        resources_type = type_args[2]

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
        if not resources:
            _resources = resources_type.load_cls(
                step_dump["resources"], secrets=_secrets
            )
        else:
            _resources = resources

        ret = cls(step_dump["uid"], args, _resources, inputs)
        ret.details = ret.details.parse_obj(step_dump["details"])
        ret.skip = step_dump["skip"]
        ret.skip_reason = step_dump["skip_reason"]
        ret.state = step_dump["state"]
        ret.results = ret.results.parse_obj(step_dump["results"])
        ret.errors = step_dump["errors"]
        ret.stats = step_dump["stats"]
        ret.results.step = ret

        return ret

Step.update_forward_refs(**locals())

class StepNGMeta(pydantic.main.ModelMetaclass):
    def __new__(mcs, name, bases, namespace,
                results_type: Type[StepResults] = StepResults,
                inputs_type: Type[StepInputs] = StepInputs,
                args_type: Type[StepArgs] = StepArgs,
                resources_type: Type[ExtResources] = ExtResources,
                details_type: Type[StepDetails] = StepDetails,
                **kwargs):

        if not check_type(results_type, StepResults):
            raise TypeError("results_type has to be subclass of StepResults")

        if not check_type(inputs_type, StepInputs):
            raise TypeError("inputs_type has to be subclass of StepInputs")

        if not check_type(args_type, StepArgs):
            raise TypeError("args_type has to be subclass of StepArgs")

        if not check_type(resources_type, ExtResources):
            raise TypeError("args_type has to be subclass of ExtResources")

        if not check_type(details_type, StepDetails):
            raise TypeError("args_type has to be subclass of StepDetails")

        if name != "StepNG":
            if results_type is StepResults:
                raise TypeError("Cannot use abstract StepResults as results_type")

            if inputs_type is StepInputs:
                raise TypeError("Cannot use abstract StepInputs as results_type")

            if args_type is StepArgs:
                raise TypeError("Cannot use abstract StepArgs as results_type")

            if resources_type is ExtResources:
                raise TypeError("Cannot use abstract ExtResources as results_type")

            if details_type is StepDetails:
                raise TypeError("Cannot use abstract StepDetails as results_type")

        namespace['ResultsModel'] = results_type
        namespace['InputsModel'] = inputs_type
        namespace['ArgsModel'] = args_type
        namespace['ResourcesModel'] = resources_type
        namespace['DetailsModel'] = details_type

        namespace.setdefault('__annotations__', {})
        namespace['__annotations__']['args'] = args_type
        namespace['__annotations__']['results'] = results_type
        namespace['__annotations__']['resources'] = resources_type
        namespace['__annotations__']['inputs'] = inputs_type
        namespace['__annotations__']['details'] = details_type

        ret = super().__new__(mcs, name, bases, namespace, **kwargs)
        return ret


class StepNG(
    BaseTractionModel,
    metaclass=StepNGMeta,
    results_type=StepResults,
    inputs_type=StepInputs,
    args_type=StepArgs,
    resources_type=ExtResources,
    details_type=StepDetails,

    validate_all=True,
    allow_population_by_field_name=False,
    extra=pydantic.Extra.forbid,
    underscore_attrs_are_private=False,
    validate_assignment=True,
):
    NAME: ClassVar[str]
    uid: str
    state: StepState
    skip: bool
    skip_reason: Optional[str]
    errors: StepErrors
    details: DetailsType
    stats: StepStats


    def __init__(
        self,
        uid: str,
        args: StepArgs,
        resources: ExtResources,
        inputs: StepInputs
    ):
        """Initilize the step.

        Args:
            uid: (str)
                An unique id for indentifying two steps of the same class
            resources: any-object
                Reference for external resources (in any form) which are constant and
                shouldn't be part of the step state or step data
            inputs: dict(str, str)
                Mapping of inputs to results of steps identified by uid
        """
            

        results = self.ResultsModel()
        details = self.DetailsModel()

        stats = StepStats(
            started=None,
            finished=None,
            skipped=False,
        )

        super().__init__(
            uid=uid,
            resources=resources,
            details=details,
            skip=False,
            skip_reason="",
            state=StepState.READY,
            results=results,
            inputs=inputs,
            args=args,
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
                self._run(on_update=_on_update)
        except StepFailedError:
            self.state = StepState.FAILED
        except Exception:
            self.state = StepState.ERROR
            _on_error(self)
            raise
        else:
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
            skipped=False,
        )

    def _finish_stats(self) -> None:
        self.stats.finished = isodate_now()
        self.stats.skipped = self.skip

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
        ret["resources"] = self.resources.dump(full=full)
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

        self.args = self.ArgsModel.parse_obj(loaded_args)
        self.errors = step_dump["errors"]
        self.stats = step_dump["stats"]
        self.results.step = self
        self.resources.load(
            step_dump["resources"]
        )  # , secrets=_secrets)

    @classmethod
    def load_cls(
        cls,
        step_dump,
        inputs_map: Dict[str, "Step[Any, Any, Any, Any, Any]"],
        resources: Optional[ExtResourcesType] = None,
        secrets: Dict[str, Dict[str, str]] = None,
    ):
        """Load step data from dictionary produced by dump method."""

        if step_dump["type"] != cls.NAME:
            raise LoadWrongStepError(
                "Cannot load %s dump to step %s" % (step_dump["type"], cls.NAME)
            )

        _secrets = secrets or {}

        loaded_args = {}
        for f, ftype in cls.ArgsModel.__fields__.items():
            if ftype.type_ == Secret:
                try:
                    loaded_args[f] = Secret(
                        _secrets["%s:%s" % (cls.NAME, step_dump["uid"])][f]
                    )
                except KeyError as e:
                    raise MissingSecretError(f) from e
            else:
                loaded_args[f] = step_dump["args"][f]

        args = cls.ArgsModel.parse_obj(loaded_args)

        loaded_inputs = {}
        for iname, itype in step_dump["inputs"].items():
            loaded_inputs[iname] = inputs_map[itype].results
        for iname in step_dump["inputs_standalone"]:
            itype = cls.InputsModel.__fields__[iname]
            loaded_result = itype.type_()
            for rfield in itype.type_.__fields__:
                if rfield == "step":
                    continue
                setattr(
                    loaded_result, rfield, step_dump["inputs_standalone"][iname][rfield]
                )
            loaded_inputs[iname] = loaded_result

        inputs = cls.InputsModel.parse_obj(loaded_inputs)
        if not resources:
            _resources = cls.ResourcesModel.load_cls(
                step_dump["resources"], secrets=_secrets
            )
        else:
            _resources = resources

        ret = cls(step_dump["uid"], args, _resources, inputs)
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
    current_step: Optional[Union[Step[Any, Any, Any, Any, Any], StepNG, Tractor]]
    step_map: Dict[str, Type[Step[Any, Any, Any, Any, Any]]] = {}
    resources_map: Dict[str, Type[ExtResource]] = {}
    uid: str

    def __init__(
        self,
        uid: str,
    ) -> None:
        """Initialize the stepper.

        Args:
            step_map: (mapping of "step-name": <step_class>)
                Mapping of step names to step classes. Used when loading stepper from
                json-compatible dict data
        """
        resource_map = {}
        current_step = None
        super().__init__(
            uid=uid,
            current_step=current_step,
        )

    def add_step(self, step: Union[Step[Any, Any, Any, Any, Any], Tractor]) -> None:
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
        out: TractorDumpDict = {
            "steps": steps,
            "resources": {},
            "uid": self.uid,
            "current_step": self.current_step.fullname if self.current_step else None}
        resources_dump: Dict[str, Any] = {}
        for step in self.steps:
            if isinstance(step, Step):
                steps.append({"type": "step", "data": step.dump(full=False)})
                for k in step.resources.__fields__:
                    res = getattr(step.resources, k)
                    resources_dump[res.fullname] = res.dump()
            else:
                steps.append({"type": "tractor", "data": step.dump(full=False)})
                for substep in step.steps:
                    for k in step.resources.__fields__:
                        res = getattr(substep.resources, k)
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
                for resource, resource_fullname in step_obj['data']["resources"].items():
                    if resource == "type":
                        continue
                    step_resources[resource] = loaded_resources[resource_fullname]
                resources_type = get_step_types(step_map[step_obj['data']["type"]])[
                    2
                ]
                resources = resources_type(**step_resources)
                step = step_map[step_obj['data']["type"]].load_cls(
                    step_obj['data'],
                    loaded_steps,
                    resources=resources,
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

Tractor.update_forward_refs(**locals())

class NamedTractorMeta(pydantic.main.ModelMetaclass):
    def __new__(mcs, name, bases, namespace,
                nt_steps: List[Tuple[str, Union[Step[Any, Any, Any, Any, Any], NamedTractor]]] = [],
                nt_inputs: Dict[str, Dict[str, str]] = {},
                **kwargs):
        nt_args_model = {}
        nt_results_model = {}
        nt_resources_model = {}
        step_inputs_model = {}
        for (nt_step_name, nt_step) in nt_steps:
            if issubclass(nt_step, Step):
                nt_step_types = get_step_types(nt_step)
                results_type = nt_step_types[0]
                args_type = nt_step_types[1]
                resources_type = nt_step_types[2]
                inputs_type = nt_step_types[3]
            elif issubclass(nt_step, NamedTractor) or issubclass(nt_step, STMD):
                results_type = nt_step.ResultsModel
                args_type = nt_step.ArgsModel
                resources_type = nt_step.ResourcesModel
                inputs_type = nt_step.InputsModel
            elif issubclass(nt_step, StepNG):
                results_type = nt_step.ResultsModel
                args_type = nt_step.ArgsModel
                resources_type = nt_step.ResourcesModel
                inputs_type = nt_step.InputsModel


            nt_args_model[nt_step_name] = (args_type, ...)
            nt_results_model[nt_step_name] = (results_type, results_type())
            nt_resources_model[nt_step_name] = (resources_type, ...)
            step_inputs_model[nt_step_name] = inputs_type

        nt_results_model["__base__"] = NoCopyModel

        step_inputs_model["__base__"] = NoCopyModel
        nt_args_model["__base__"] = StepArgs

        ArgsModel = pydantic.create_model("%s_args" % (name,), **nt_args_model)
        ArgsModel.update_forward_refs(**globals())
        setattr(pydantic.main, "%s_args" % (name,), ArgsModel)
        ResultsModel = pydantic.create_model("%s_results" % (name,), **nt_results_model)
        ResultsModel.update_forward_refs(**globals())
        setattr(pydantic.main, "%s_results" % (name,), ResultsModel)
        ResourcesModel = pydantic.create_model("%s_resources" % (name,), **nt_resources_model)
        ResourcesModel.update_forward_refs(**globals())
        setattr(pydantic.main, "%s_resources" % (name,), ResourcesModel)
        StepInputsModel = pydantic.create_model("%s_step_inputs" % (name,), **step_inputs_model)
        StepInputsModel.update_forward_refs(**globals())
        setattr(pydantic.main, "%s_step_inputs" % (name,), StepInputsModel)

        nt_inputs_model = {"__base__": NoCopyModel}
        for input_name, input_type in nt_inputs.items():
            nt_inputs_model[input_name] = (input_type, ...)
        NTInputsModel = pydantic.create_model("%s_inputs" % name, **nt_inputs_model)
        setattr(pydantic.main, "%s_inputs" % (name,), NTInputsModel)

        namespace.setdefault('__annotations__', {})
        namespace['__annotations__']['args'] = ArgsModel
        namespace['__annotations__']['results'] = ResultsModel
        namespace['__annotations__']['resources'] = ResourcesModel
        namespace['__annotations__']['inputs'] = NTInputsModel
        namespace['ResultsModel'] = ResultsModel
        namespace['ArgsModel'] = ArgsModel
        namespace['ResourcesModel'] = ResourcesModel
        namespace['StepInputsModel'] = StepInputsModel
        namespace['InputsModel'] = NTInputsModel
        namespace['nt_steps'] = nt_steps
        namespace['results'] = ResultsModel()

        missing_inputs = set([x[0] for x in namespace['nt_steps']]) - set(namespace['INPUTS_MAP'].keys())
        if missing_inputs:
            raise TypeError("Missing steps in INPUTS_MAP definition: %s" % missing_inputs)

        ret = super().__new__(mcs, name, bases, namespace)
        return ret


class NTInput(pydantic.BaseModel):
    name: str




class NamedTractor(NoCopyModel, metaclass=NamedTractorMeta, nt_steps=[], nt_inputs={}):
    """Class which runs sequence of steps."""

    _step_map: Dict[str, Union[Step[Any, Any, Any, Any, Any],NamedTractor]] = {}
    steps: List[Union[Step[Any, Any, Any, Any, Any],NamedTractor]] = []
    current_step: Optional[Union[Step[Any, Any, Any, Any, Any], StepNG, Tractor, NamedTractor]] = None
    uid: str
    state: StepState = StepState.READY
    INPUTS_MAP: ClassVar[Dict[str, Dict[str, str]]] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        results_model_data = {}

        for step_name, step_type in self.nt_steps:
            step_inputs = self.INPUTS_MAP[step_name]
            inputs_model_data = {}
            for input_, output_from in step_inputs.items():
                if isinstance(output_from, str):
                    inputs_model_data[input_] = self._step_map[output_from].results
                else: # NTInput
                    inputs_model_data[input_] = getattr(self.inputs, output_from.name)

            inputs_model = getattr(self.StepInputsModel, step_name)
            inputs_model = getattr(self.StepInputsModel, step_name)(**inputs_model_data)
            step = step_type(uid='%s::%s' % (self.uid, step_name),
                          args=getattr(self.args, step_name),
                          resources=getattr(self.resources, step_name),
                          inputs=inputs_model)

            results_model_data[step_name] = step.results
            self._step_map[step_name] = step
            self.steps.append(step)
        self.results = self.ResultsModel(**results_model_data)
        self.state = StepState.READY

    @property
    def fullname(self):
        return self.uid

    def dump(self, full: bool) -> TractorDumpDict:
        """Dump stepper state and shared_results to json compatible dict."""
        steps: List[Dict[str, Any]] = []
        out: TractorDumpDict = {
            "state": self.state.value,
            "steps": steps,
            "resources": {},
            "uid": self.uid,
            "current_step": self.current_step.fullname}
        resources_dump: Dict[str, Any] = {}
        for step in self.steps:
            if isinstance(step, Step):
                steps.append({"type": "step", "data": step.dump(full=False)})
                for k in step.resources.__fields__:
                    res = getattr(step.resources, k)
                    resources_dump[res.fullname] = res.dump()
            elif isinstance(step, StepNG):
                steps.append({"type": "stepng", "data": step.dump(full=False)})
                for k in step.resources.__fields__:
                    res = getattr(step.resources, k)
                    resources_dump[res.fullname] = res.dump()
            elif isinstance(step, STMD):
                for tractor_step_name, resources in step.resources:
                    for resource in resources:
                        for k in resources.__fields__:
                            res = getattr(resources, k)
                            resources_dump[res.fullname] = res.dump()
                steps.append({"type": "stmd", "data":step.dump()})
            else:
                steps.append({"type": "tractor", "data": step.dump(full=False)})
                for step_name in step.resources.__fields__:
                    for resource in getattr(step.resources, step_name).__fields__:
                        res = getattr(getattr(step.resources, step_name), resource)
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
                for resource, resource_fullname in step_obj['data']["resources"].items():
                    if resource == "type":
                        continue
                    step_resources[resource] = loaded_resources[resource_fullname]
                resources_type = get_step_types(step_map[step_obj['data']["type"]])[
                    2
                ]
                resources = resources_type(**step_resources)
                step = step_map[step_obj['data']["type"]].load_cls(
                    step_obj['data'],
                    loaded_steps,
                    resources=resources,
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
        self.state = StepState.RUNNING
        for step in self.steps[start_from:]:
            self.current_step = step
            step.run(on_update=on_update, on_error=on_error)
            if step.state == StepState.ERROR:
                self.state = StepState.ERROR
                if on_error:
                    on_error(self)
                return
            if step.state == StepState.FAILED:
                self.state = StepState.FAILED
                if on_update:
                    on_update(self)
                return

        self.state = StepState.FINISHED
        return self

NamedTractor.update_forward_refs(**locals())

class STMDMeta(pydantic.main.ModelMetaclass):
    def __new__(mcs, name, bases, namespace,
                tractor_type: Type[NamedTractor],
                **kwargs):

        results_model = {}
        inputs_model = {}
        inputs_as_results_model = {}
        args_model = {}

        results_model["__base__"] = StepResults
        args_model["__base__"] = tractor_type.ArgsModel
        inputs_model["__base__"] = StepInputs
        #inputs_as_results_model["__base__"] = StepResults

        results_model["data"] = (List[tractor_type.ResultsModel], [])
        ResultsModel = pydantic.create_model("%s_results" % (name,), **results_model)
        setattr(pydantic.main, "%s_results" % (name,), ResultsModel)
        ResultsModel.update_forward_refs(**globals())

        args_model["tractors"] = (int, ...)
        ArgsModel = pydantic.create_model("%s_args" % (name,), **args_model)
        setattr(pydantic.main, "%s_args" % (name,), ArgsModel)
        ArgsModel.update_forward_refs(**globals())

        MultiDataModel = pydantic.create_model(
            "%s_multi_model" % name,
            **{ "__base__": StepResults,
                "multidata": (List[tractor_type.InputsModel], [])}
        )
        MultiDataModel.update_forward_refs(**globals())

        #inputs_as_results_model["multidata"] = (List[tractor_type.ResultsModel], [])
        #InputsAsResultsModel = pydantic.create_model("%s_inputs_as_results" % name, **inputs_as_results_model)
        #InputsAsResultsModel.update_forward_refs()

        inputs_model["data"] = (MultiDataModel, ...)
        InputsModel = pydantic.create_model("%s_inputs" % name, **inputs_model)
        InputsModel.update_forward_refs(**globals())
    
        namespace.setdefault('__annotations__', {})
        namespace['__annotations__']['results'] = ResultsModel
        namespace['__annotations__']['inputs'] = InputsModel
        namespace['__annotations__']['args'] = ArgsModel
        namespace['__annotations__']['resources'] = tractor_type.ResourcesModel
        #namespace['__annotations__']['steps'] = (List[tractor_type], ...)

        namespace['ResultsModel'] = ResultsModel
        namespace['InputsModel'] = InputsModel
        namespace['TractorType'] = tractor_type
        #namespace['InputsAsResultsModel'] = InputsAsResultsModel
        namespace['ArgsModel'] = ArgsModel
        namespace['ResourcesModel'] = tractor_type.ResourcesModel
        namespace['MultiDataModel'] = MultiDataModel
        namespace['results'] = ResultsModel()
        ret = super().__new__(mcs, name, bases, namespace)
        return ret


class STMD(
    BaseTractionModel, metaclass=STMDMeta, tractor_type=NamedTractor
):
    uid: str

    def __init__(
        self,
        uid: str,
        args: ArgsType,
        resources: Optional[ExtResourcesType] = None,
        inputs: Optional[InputsType] = None,
    ):
        """Initilize the step.

        Args:
            uid: (str)
                An unique id for indentifying two steps of the same class
            resources: any-object
                Reference for external resources (in any form) which are constant and
                shouldn't be part of the step state or step data
            inputs: dict(str, str)
                Mapping of inputs to results of steps identified by uid
        """
        super().__init__(
            uid=uid,
            resources=resources,
            results=self.ResultsModel(),
            inputs=inputs,
            args=args,
        )

        # override init value copy and set original object via __setattr__
        self.inputs = inputs or StepInputs()

        self.results.step = self

    @property
    def fullname(self):
        return self.uid

    def run(
        self,
        on_error: StepOnErrorCallable = None,
        on_update: StepOnUpdateCallable = None,
    ) -> None:
        _on_update: StepOnUpdateCallable = lambda step: None
        if on_update:
            _on_update = on_update
    
        with ProcessPoolExecutor(max_workers=self.args.tractors) as executor:
            ft_results = {}
            for i in range(0, len(self.inputs.data.multidata)):
                nt = self.TractorType(uid="%s:%d" % (self.uid, i),
                                 args=self.args,
                                 resources=self.resources,
                                 inputs=self.inputs.data.multidata[i])
                res = executor.submit(nt.run)
                ft_results[res] = (nt, i)
                self.results.data.append(self.TractorType.ResultsModel())
            _on_update(self)
            for ft in as_completed(ft_results):
                (_, i) = ft_results[ft]
                nt = ft.result()
                self.results.data[i] = nt.results
                _on_update(self)

    def dump(self, full=False):
        return self.dict()


STMD.update_forward_refs(**locals())



class NoArgs(StepArgs):
    pass


class NoResources(ExtResources):
    NAME: ClassVar[str] = "NoResources"


class NoDetails(StepDetails):
    pass

class NoResults(StepResults):
    pass
