import abc
import copy
import datetime
from dataclasses import make_dataclass, asdict
from functools import partial
import inspect

from typing import (
    Dict, List, Callable, Optional, TypedDict, Any, Type, Sequence,
    Generic, TypeVar, Protocol, Mapping, ClassVar, cast, get_args,
    Tuple, NamedTuple, Union, NewType, runtime_checkable, Callable, Iterator,
    _GenericAlias)
import typing_inspect


import enum
from dataclasses_json import dataclass_json
import pydantic
import pydantic.generics
import pydantic.fields
import pydantic.main
from pydantic.dataclasses import dataclass

from .exc import LoadWrongStepError, LoadWrongExtResourceError, MissingSecret


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
            raise TypeError('Secret required')
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


class StepState(enum.Enum):
    """Enum-like class to store step state."""
    
    READY=0
    PREP=1
    RUNNING=2
    FINISHED=3
    FAILED=4
    ERROR=5




class ArgsTypeCls(pydantic.generics.GenericModel, validate_assignment=True):

    def dict(self, *,
        include: Union['AbstractSetIntStr', 'MappingIntStrAny'] = None,
        exclude: Union['AbstractSetIntStr', 'MappingIntStrAny'] = None,
        by_alias: bool = False,
        skip_defaults: bool = None,
        exclude_unset: bool = False,
        exclude_defaults: bool = False,
        exclude_none: bool = False,
    ) -> Dict[str, Any]:
        model_mapping = {}
        for k,v in self.__dict__.items():
            if type(v) == Secret:
                model_mapping[k] = (str, "*CENSORED*")
            else:
                model_mapping[k] = (type(v), v)
        m = pydantic.create_model(
            self.__class__.__name__+"Dump",
            **model_mapping
        )
        return pydantic.BaseModel.dict(m())


class ExtResource(pydantic.generics.GenericModel):
    NAME: ClassVar[str]
    SECRETS: ClassVar[List[str]] = []
    uid: str

    def dump(self):
        ret = self.dict()
        for k in ret:
            if k in self.SECRETS:
                ret[k] = "*CENSORED*"
        ret['type'] = self.NAME
        return ret

    def load(self, dump, secrets: Dict[str, str] = None):
        _secrets = secrets or {}
        dump_copy = dump.copy()
        if dump_copy['type'] != self.NAME:
            raise LoadWrongExtResourceError("Cannot load %s into %s" % (dump_copy['type'], self.NAME))
        dump_copy.pop('type')
        parsed = self.parse_obj(dump)
        for f in self.__fields__:
            if f in _secrets:
                try:
                    setattr(self, f, getattr(parsed, _secrets[f]))
                except KeyError as e:
                    raise MissingSecret from e
            else:
                setattr(self, f, getattr(parsed, f))
        return self

    @classmethod
    def load_cls(cls, dump, secrets: Dict[str, str] = None):
        _secrets = secrets or {}
        dump_copy = dump.copy()
        if dump['type'] != cls.NAME:
            raise LoadWrongExtResourceError("Cannot load %s into %s" %  (dump_copy['type'], cls.NAME))
        dump_copy.pop('type')
        print(secrets)
        for secret in cls.SECRETS:
            try:
                dump_copy[secret] = _secrets[secret]
            except KeyError as e:
                raise MissingSecret from e
        return cls.parse_obj(dump_copy)

    @property
    def fullname(self) -> str:
        """Full name of class instance."""
        return "%s:%s" % (self.NAME, self.uid)

class ResourcesModelMeta(pydantic.main.ModelMetaclass):
    def __new__(cls, name, bases, attrs):
        ret = super().__new__(cls, name, bases, attrs)
        for k, v in ret.__fields__.items():
            if not issubclass(v.type_, ExtResource):
                raise ValueError("%s has to be type ExtResource" % k)
        return ret


class ExtResourcesCls(pydantic.generics.BaseModel, metaclass=ResourcesModelMeta):
    NAME: ClassVar[str]

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if key == 'uid':
                continue
            if not issubclass(type(val), ExtResource):
                raise ValueError("%s has to be type ExtResource not %s" % (key, type(val)))

        super().__init__(**kwargs)

    def dump(self, full=True):
        ret = {}
        ret['type'] = self.NAME
        for f in self.__fields__:
            if full:
                ret[f] = getattr(self, f).dump()
            else:
                ret[f] = getattr(self,f).fullname
        return ret

    def load(self, dump, secrets: Dict[str, str] = None):
        dump_copy = dump.copy()
        if dump_copy['type'] != self.NAME:
            raise LoadWrongExtResourceError("Cannot load %s into %s" %  (dump_copy['type'], self.NAME))
        dump_copy.pop('type')
        for key, val in dump_copy.items():
            setattr(self, key, getattr(self, key).load(val, secrets=secrets))
        return self


    @classmethod
    def load_cls(cls, dump, secrets: Dict[str, str] = None):
        dump_copy = dump.copy()
        if dump['type'] != cls.NAME:
            raise LoadWrongExtResourceError("Cannot load %s into %s" %  (dump_copy['type'], cls.NAME))
        dump_copy.pop('type')
        ret = {}
        for key, val  in dump_copy.items():
            ret[key] = cls.__fields__[key].type_.load_cls(val, secrets=secrets)
        return cls(**ret)


ArgsType = TypeVar("ArgsType", bound=ArgsTypeCls)
ExtResourcesType = TypeVar("ExtResourcesType", bound=ExtResourcesCls)


class DefaultsModelMeta(pydantic.main.ModelMetaclass):
    def __new__(cls, name, bases, attrs):
        annotations = attrs.get("__annotations__", {})
        for attrk, attrv in attrs.items():
            if attrk in ['dump', 'load', 'Config', 'dict']:
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


class RequiredDefaultsModel(pydantic.generics.GenericModel, metaclass=DefaultsModelMeta):
    class Config:
        validate_assignment = True
    pass


class StepResults(RequiredDefaultsModel):
    """Class to store results of step."""
    step: Optional["Step"] = pydantic.Field(default=None, repr=False)

    def __setattr__(self, key, value):
        """Override setattr to make sure assigning to step.results doesn't break
        existing references for step.results"""
    
        if key == 'step' and value:
            dict_without_original_value = {k: v for k, v in self.__dict__.items() if k != key}
            self.__fields__['step'].validate(value, dict_without_original_value, loc=key)
            object.__setattr__(self, key, value)
        else:
            super().__setattr__(key, value)

    def dict(self,
             *,
             include: Union['AbstractSetIntStr', 'MappingIntStrAny'] = None,
             exclude: Union['AbstractSetIntStr', 'MappingIntStrAny'] = None,
             by_alias: bool = False,
             skip_defaults: bool = None,
             exclude_unset: bool = False,
             exclude_defaults: bool = False,
             exclude_none: bool = False):
        return super().dict(exclude={'step'})


class StepDetails(RequiredDefaultsModel):
    """Class to store step details to."""
    pass


class StepInputs(pydantic.BaseModel, validate_assignment=True):
    @pydantic.validator('*', pre=True)
    def valid_fields(cls, v):
        if not isinstance(v, StepResults):
            raise ValueError("field must be StepResults subclass, but is %s" % type(v))
        return v

    def __setattr__(self, key, value):
        """Override setattr to make sure assigning to step.results doesn't break
        existing references for step.results"""
    
        dict_without_original_value = {k: v for k, v in self.__dict__.items() if k != key}
        self.__fields__[key].validate(value, dict_without_original_value, loc=key)
        object.__setattr__(self, key, value)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        for k, v in data.items():
            setattr(self, k, v)


class NoInputs(StepInputs):
    pass


class SharedResults(pydantic.BaseModel):
    results: Dict[str, Any]
    
    class Config:
        copy_on_model_validation=False


ResultsType = TypeVar("ResultsType", bound=StepResults)
InputsType = TypeVar("InputsType", bound=StepInputs)
DetailsType = TypeVar("DetailsType", bound=StepDetails)


class StepErrors(pydantic.BaseModel):
    """Class to store results of step."""

    errors: Dict[Any, Any] = {}


class StepStats(TypedDict):
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


class Step(pydantic.generics.BaseModel, Generic[ResultsType, ArgsType, ExtResourcesType, InputsType, DetailsType],
           validate_all=True, allow_population_by_field_name=False, extra=pydantic.Extra.forbid, underscore_attrs_are_private=False,
           validate_assignment=True):
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
    shared_results: SharedResults
    inputs: Optional[InputsType]
    args: ArgsType

    def __setattr__(self, key, value):
        """Override setattr to make sure assigning to step.results doesn't break
        existing references for step.results"""
    
        if key == 'results':
            for k in value.__fields__:
                v = getattr(value, k)
                setattr(self.results, k, v)
    
            super().__setattr__(key, self.results)
        elif key == 'inputs':
            dict_without_original_value = {k: v for k, v in self.__dict__.items() if k != key}
            self.__fields__[key].validate(value, dict_without_original_value, loc=key)
            object.__setattr__(self, key, value)
        else:
            super().__setattr__(key, value)

    @classmethod
    def get_step_types(cls):
        type_args = get_args(cls.__orig_bases__[0]) # type: ignore
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


    #@pydantic.validate_arguments(config=dict(arbitrary_types_allowed=False))
    def __init__(self,
                 uid: str,
                 step_args: ArgsType,
                 shared_results: SharedResults,
                 external_resources: Optional[ExtResourcesType]=None,
                 inputs: Optional[InputsType]=None):
        """Initilize the step.

        Args:
            uid: (str)
                An unique id for indentifying two steps of the same class
            shared_results: (dict-like object)
                Object to store shared data between steps
            external_resources: any-object
                Reference for external resources (in any form) which are constant and
                shouldn't be part of the step state or step data
            inputs: dict(str, str)
                Mapping of inputs to results of steps identified by uid
        """
        
        type_args = self.get_step_types()
        if not type_args:
            raise TypeError("Missing generic annotations for Step class. Use Step[ResultsCls, ArgsCls, ExtResourcesCls, InputsCls]")

        results_type = type_args[0]
        args_type = type_args[1]
        resources_type = type_args[2]
        inputs_type = type_args[3]
        details_type = type_args[4]

        if type(step_args) != args_type:
            raise TypeError("Step arguments are not type of %s but %s" % (args_type, type(step_args)))
        if type(shared_results) != SharedResults:
            raise TypeError("Step shared_results are not type of %s but %s" % (SharedResults, type(step_args)))
        if external_resources is not None and type(external_resources) != resources_type:
            raise TypeError("Step external resources are not type of %s but %s" % (resources_type, type(external_resources)))
        if inputs is not None and type(inputs) != inputs_type:
            raise TypeError("Step inputs are not type of %s but %s" % (inputs_type, type(inputs)))

        results = results_type()
        details = details_type()

        stats = {
            "started": None,
            "finished": None,
            "skip": False,
            "skip_reason": "",
            "skipped": False,
            "state": StepState.READY,
        }
        super().__init__(uid=uid,
                external_resources=external_resources,
                shared_results=shared_results,
                details=details,
                skip=False,
                skip_reason="",
                state=StepState.READY,
                results=results,
                inputs=inputs or StepInputs(),
                args=step_args,
                errors=StepErrors(),
                stats=stats
                )
        self.shared_results = shared_results

        # override init value copy and set original object via __setattr__
        self.inputs = inputs or StepInputs()

        self.results.step = self

    @property
    def fullname(self) -> str:
        """Full name of class instance."""
        return "%s:%s" % (self.NAME, self.uid)

    def run(self, on_update: StepOnUpdateCallable=None, on_error: StepOnErrorCallable=None) -> None:
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
            self.stats["started"] = isodate_now()

            self.state = StepState.PREP
            self._pre_run()
            _on_update(self) # type: ignore
        try:
            if self.state not in (StepState.PREP, StepState.ERROR):
                return
            if not self.skip:
                self.state = StepState.RUNNING
                _on_update(self) # type: ignore
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
            self._store_results()
            _on_update(self) # type: ignore

    def _pre_run(self) -> None:
        """Execute code needed before step run.

        In this method, all neccesary preparation of data can be done.
        It can be also used to determine if step should run or not by setting
        self.skip to True and providing self.skip_reason string with explanation.
        """
        pass

    def _reset_stats(self) -> None:
        self.stats = {
            "started": None,
            "finished": None,
            "skip": self.skip,
            "skip_reason": self.skip_reason,
            "skipped": False,
            "state": self.state,
        }

    def _finish_stats(self) -> None:
        self.stats["finished"] = isodate_now()
        self.stats["skipped"] = self.skip
        self.stats["skip"] = self.skip
        self.stats["skip_reason"] = self.skip_reason
        self.stats["state"] = self.state

    def _store_results(self) -> None:
        self.shared_results.results[self.fullname] = self.results

    @abc.abstractmethod
    def _run(self, on_update: StepOnUpdateCallable=None) -> None:  # pragma: no cover
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
        ret = self.dict(exclude={'shared_results', 'inputs', 'results'})
        ret['type'] = self.NAME
        ret['inputs'] = {}
        ret['inputs_standalone'] = {}
        ret['results'] = self.results.dict(exclude={'step'})
        ret['external_resources'] = self.external_resources.dump(full=full)
        for f,ftype in self.inputs.__fields__.items():
            field = getattr(self.inputs, f)
            if field.step:
                ret['inputs'][f] = getattr(self.inputs, f).step.fullname
            else:
                ret['inputs_standalone'][f] = getattr(self.inputs, f).dict(exclude={"step"})
        return ret

    def load(self, step_dump):
        """Load step data from dictionary produced by dump method."""

        if step_dump['type'] != self.NAME:
            raise LoadWrongStepError('Cannot load %s dump to step %s' % (step_dump['type'], self.NAME))

        self.details = self.details.parse_obj(step_dump['details'])
        self.skip = step_dump['skip']
        self.skip_reason = step_dump['skip_reason']
        self.state = step_dump['state']
        self.results = self.results.parse_obj(step_dump['results'])
        loaded_args = {}
        for f in self.args.__fields__:
            if isinstance(getattr(self.args, f), Secret):
                loaded_args[f] = getattr(self.args, f)
            else:
                loaded_args[f] = step_dump['args'][f]

        self.args = self.args.parse_obj(loaded_args)
        self.errors = step_dump['errors']
        self.stats = step_dump['stats']
        self.results.step = self
        self.external_resources.load(step_dump['external_resources'])

    @classmethod
    def load_cls(
            cls,
            step_dump,
            secret_args: Dict[str, Secret],
            inputs_map: Dict[str, "Step[Any, Any, Any, Any]"],
            shared_results: SharedResults,
            external_resources: Optional[ExtResourcesType]=None):
        """Load step data from dictionary produced by dump method."""

        if step_dump['type'] != cls.NAME:
            raise LoadWrongStepError('Cannot load %s dump to step %s' % (step_dump['type'], cls.NAME))

        type_args = get_args(cls.__orig_bases__[0]) # type: ignore
        args_type = type_args[1]
        inputs_type = type_args[3]
        external_resources_type = type_args[2]

        loaded_args = {}
        for f, ftype in args_type.__fields__.items():
            if ftype.type_ == Secret:
                try:
                    loaded_args[f] = secret_args[f]
                except KeyError as e:
                    raise MissingSecret(f) from e
            else:
                loaded_args[f] = step_dump['args'][f]

        args = args_type.parse_obj(loaded_args)
        
        loaded_inputs = {}
        for iname, itype in step_dump['inputs'].items():
            loaded_inputs[iname] = inputs_map[itype].results
        for iname in step_dump['inputs_standalone']:
            itype = inputs_type.__fields__[iname]
            loaded_result = itype.type_()
            for rfield in  itype.type_.__fields__:
                if rfield == 'step':
                    continue
                setattr(loaded_result, rfield, step_dump['inputs_standalone'][iname][rfield])
            loaded_inputs[iname] = loaded_result
        
        inputs = inputs_type.parse_obj(loaded_inputs)
        if not external_resources:
            _external_resources = external_resources_type.load_cls(step_dump['external_resources'])
        else:
            _external_resources = external_resources

        ret = cls(step_dump['uid'], args, shared_results, _external_resources, inputs)
        ret.details = ret.details.parse_obj(step_dump['details'])
        ret.skip = step_dump['skip']
        ret.skip_reason = step_dump['skip_reason']
        ret.state = step_dump['state']
        ret.results = ret.results.parse_obj(step_dump['results'])
        ret.errors = step_dump['errors']
        ret.stats = step_dump['stats']
        ret.results.step = ret

        return ret



class TractorDumpDict(TypedDict):
    steps: List[Dict[str, Any]]
    shared_results: Dict[str, Any]


class TractorValidateResult(TypedDict):
    missing_inputs: List[Tuple[str, str, str]]
    valid: bool


class Tractor(pydantic.BaseModel,
              validate_all=True, allow_population_by_field_name=False, extra=pydantic.Extra.forbid, underscore_attrs_are_private=False,
              validate_assignment=True):
    """Class which runs sequence of steps."""

    steps: List[Step[Any, Any, Any, Any, Any]] = []
    shared_results: SharedResults
    current_step: Optional[Step[Any, Any, Any, Any, Any]]
    step_map: Dict[str, Type[Step[Any, Any, Any, Any, Any]]] = {}
    resources_map: Dict[str, Type[ExtResource]] = {}


    def __init__(self, step_map: Dict[str, Type[Step[Any, Any, Any, Any, Any]]], resources_map: Dict[str, Type[ExtResource]]) -> None:
        """Initialize the stepper.

        Args:
            step_map: (mapping of "step-name": <step_class>)
                Mapping of step names to step classes. Used when loading stepper from
                json-compatible dict data
        """
        step_map = step_map
        shared_results = SharedResults(results={})
        current_step = None
        super().__init__(
            step_map=step_map,
            shared_results=shared_results,
            current_step=current_step,
            resources_map=resources_map
        )

    def add_step(self, step: Step[Any, Any, Any, Any, Any]) -> None:
        """Add step to step sequence."""
        self.steps.append(step)
        self.shared_results.results[step.fullname] = step.results

    def dump(self) -> TractorDumpDict:
        """Dump stepper state and shared_results to json compatible dict."""
        steps: List[Dict[str, Any]] = []
        out: TractorDumpDict = {"steps": steps, "resources": {}}
        for step in self.steps:
            steps.append(step.dump(full=False))
        for step in self.steps:
            for k in step.external_resources.__fields__:
                res = getattr(step.external_resources, k)
                out['resources'][res.fullname] = res.dump()
        return out

    def load(self, dump_obj: TractorDumpDict, secrets: Dict[str, Dict[str, str]]) -> None:
        """Load and initialize stepper from data produced by dump method."""
        loaded_steps = {}
        loaded_resources = {}
        self.steps = []
        for fullname, resource_dump in dump_obj["resources"].items():
            resource_dump_copy = resource_dump.copy()
            print("%s:%s" % (resource_dump['type'], resource_dump['uid']))
            print(secrets)
            loaded_resources[fullname] = self.resources_map[resource_dump['type']].load_cls(
                resource_dump_copy, 
                secrets.get("%s:%s" % (resource_dump['type'], resource_dump['uid']), {})
            )

        for step_obj in dump_obj["steps"]:
            step_resources = {}
            for resource, resource_fullname in step_obj['external_resources'].items():
                if resource == 'type':
                    continue
                step_resources[resource] = loaded_resources[resource_fullname]
            external_resources_type = self.step_map[step_obj["type"]].get_step_types()[2]
            external_resources = external_resources_type(**step_resources)
            step = self.step_map[step_obj["type"]].load_cls(
                step_obj,
                secrets.get("%s:%s" % (step_obj['type'], step_obj['uid']), {}),
                loaded_steps,
                self.shared_results,
                external_resources=external_resources)

            loaded_steps[step.fullname] = step
            self.steps.append(step)

    def run(self,
            start_from: int=0,
            on_error: StepOnErrorCallable=None,
            on_update: StepOnUpdateCallable=None) -> None:
        """Run the stepper sequence."""
        for step in self.steps[start_from:]:
            self.current_step = step
            step.run(on_update=on_update, on_error=on_error)



class NoArgs(ArgsTypeCls):
    pass


class NoResources(ExtResourcesCls):
    NAME: ClassVar[str] = "NoResources"

class NoDetails(StepDetails):
    pass

@dataclass
class NoResult:
    pass


