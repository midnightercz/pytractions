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

#Step = ForwardRef('Step')

StepNG = ForwardRef('StepNG')

Tractor = ForwardRef('Tractor')

NamedTractor = ForwardRef('NamedTractor')

STMD = ForwardRef('STMD')

Step = ForwardRef('Step')

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
        final_to_check = to_check
        final_expected = expected
        if type(to_check) == ForwardRef and type(expected) == ForwardRef:
            return (to_check.__forward_code__, to_check.__forward_arg__) == (expected.__forward_code__, expected.__forward_arg__)
        if not issubclass(final_to_check, final_expected):
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
    """Metaclass to ensure all defined attributes in StepIOs have default value."""

    NO_ANNOTATION: ClassVar[List[str]] = ["dump", "load", "Config", "dict", "input_mode", "_ios_type"]

    def __new__(cls, name, bases, attrs):
        annotations = attrs.get("__annotations__", {})
        NO_ANNOTATION = attrs.get('NO_ANNOTATION', cls.NO_ANNOTATION)
        for attrk, attrv in attrs.items():
            if attrk in NO_ANNOTATION:
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
        copy_on_model_validation=False




class StepIO(RequiredDefaultsModel):
    """Class to store results of step."""

    _step: Optional[Union[Step,STMD,NamedTractor]] = pydantic.PrivateAttr(default=None)
    _input_mode: bool = pydantic.PrivateAttr(default_factory=lambda: True)
    _ref: Optional["StepIO"] = pydantic.PrivateAttr(default=None)

    def __setattr__(self, key, value):
        """Override setattr to make sure assigning to step.results doesn't break
        existing references for step.results"""


        if key == "step" and value:
            dict_without_original_value = {
                k: v for k, v in self.__dict__.items() if k != key
            }
            #Step.update_forward_refs()

            self.__fields__["step"].validate(
                value, dict_without_original_value, loc=key, cls=Step
            )

            object.__setattr__(self, key, value)
        #elif key in ["__orig_class__", ]:
        #    object.__setattr__(self, key, value)
        #else:
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
        if not self._ref:
            return super().dict(exclude={"step", "_input_mode", "_ref"})
        else:
            return self._ref.dict(exclude={"step", "_input_mode", "_ref"})


    def input_mode(self, mode: bool = True):
        self._input_mode = mode



class StepIOList(StepIO):

    NO_ANNOTATION: ClassVar[List[str]] = [
        "dump", "load", "Config", "dict", "input_mode", "_check_type",
        "append", "count", "index", "insert", "pop", "reverse",
        "sort", "_ios_type"]

    def _check_type(self, item):
        if not issubclass(item, self._ios_type):
            raise TypeError("Item has to be type of %s, but is %s" % (self._ios_type, item))

    def __contains__(self, item):
        return self.list_data.__contains__(item)

    def __delitem__(self, n):
        return self.list_data.__delitem__(n)

    def __getitem__(self, n):
        return self.list_data[n]

    def __iter__(self):
        return iter(self.list_data)

    def __len__(self):
        return len(self.list_data)

    def __reversed__(self):
        return self.list_data.__reversed__()

    def __setitem__(self, n, item):
        self._check_type(type(item))
        _item = self.list_data[n]
        for f in _item.__fields__:
            object.__setattr__(_item, f, getattr(item, f))

    def append(self, item):
        self._check_type(type(item))
        ret = self.list_data.append(item)
        return ret

    def count(self, item):
        return self.list_data.count(item)

    def index(self, item):
        return self.list_data.index(item)

    def insert(self, item, n):
        self._check_type(type(item))
        ret = self.list_data.insert(item, n)
        self.list_data[n].input_mode(self._input_mode)

    def pop(self, n):
        return self.list_data.pop(n)

    def reverse(self):
        return self.list_data.reverse()

    def sort(self):
        return self.list_data.sort()

    #def extend(self, another):
    #    if not issubclass(type(another), type(self)):
    #        raise TypeError("List has to be type of %s" % type(self))
    #    return self._data.extend(another)



class StepIOsMeta(pydantic.main.ModelMetaclass):
    """Metaclass to ensure all defined attributes in StepIOs have default value."""

    def __new__(cls, name, bases, attrs):
        annotations = attrs.get("__annotations__", {})
        for attrk, attrv in attrs.items():
            if attrk.startswith("__"):
                continue
            if inspect.ismethoddescriptor(attrv):
                continue
            if attrk in ('Config', "input_mode", "set_step"):
                continue
            if attrk not in annotations:
                raise TypeError("%s has to be annotated" % attrk)
        for annotated in annotations:
            ann_cls = annotations[annotated]
            if annotated not in attrs:
                raise TypeError("Attribute %s is missing default value" % annotated)
            if get_origin(ann_cls):
                ann_cls = get_origin(ann_cls)

            if not issubclass(ann_cls, StepIO):
                raise ValueError("%s attribute as to be StepIO type not %s" % (annotated, annotations[annotated]))

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
        copy_on_model_validation=False

    def check_type(self, key, value):
        if key in self.__fields__:
            if get_origin(self.__fields__[key].outer_type_) == Union:
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
            elif get_origin(self.__fields__[key].outer_type_) ==  list:
                if not isinstance(value, list):
                    raise TypeError(f'"{self.__class__.__name__}->{key}" has to be type {self.__fields__[key].outer_type_}" not %s' % type(value))
                for x in value:
                    self.check_type(x, self.__fields__[key].outer_type_.__args__[0])
            else:
                if not isinstance(value, self.__fields__[key].outer_type_):
                    raise TypeError(f'"{self.__class__.__name__}->{key}" has to be type "{self.__fields__[key].type_}" not %s' % type(value))

    def __setattr__(self, key, value):
        """Override setattr to make sure assigning to step.results doesn't break
        existing references for step.results"""
   
        dict_without_original_value = {
            k: v for k, v in self.__dict__.items() if k != key
        }
        try:
            self.__fields__[key].validate(value, dict_without_original_value, loc=key)
        except KeyError as e:
            if not key.startswith("_"):
                raise ValueError(f'"{self.__class__.__name__}" object has no field "{key}"') from e
        self.check_type(key, value)
        if not isinstance(value, StepIO):
            object.__setattr__(self, key, value)
        else:
            # do set stepIO, rather copy its attributes.
            self_io = getattr(self, key)
            for f in value.__fields__:
                object.__setattr__(self_io, f, getattr(value, f))


    def __init__(self, **data: Any) -> None:
        for k, v in data.items():
            self.check_type(k, v)
        super().__init__(**data)
        for k, v in data.items():
           setattr(self, k, v)

class StepDetails(RequiredDefaultsModel, NoCopyModel):
    """Class to store step details to."""

    pass

class NoDetails(StepDetails):
    pass

class StepIOs(NoCopyModel, metaclass=StepIOsMeta):
    @pydantic.validator("*", pre=True)
    def valid_fields(cls, v):
        if not isinstance(v, StepIO):
            raise ValueError("field must be StepIOs subclass, but is %s" % type(v))
        return v

    def __init__(self, **data: Any) -> None:
        for k, v in data.items():
            self.check_type(k, v)

        no_io_data = {}
        for k in data.keys():
            if not isinstance(data[k], StepIO):
                no_io_data[k] = data[k]

        super().__init__(**no_io_data)
        for k, v in data.items():
           setattr(self, k, v)

    def input_mode(self, mode: bool = True):
        #self._input_mode = mode
        for f in self.__fields__.keys():
            v = getattr(self, f)
            v.input_mode(mode=mode)

    def set_step(self, step):
        for f in self.__fields__:
            getattr(self, f)._step = self

    def __setattr__(self, key, value):
        """Override setattr to make sure assigning to step.results doesn't break
        existing references for step.results"""
   
        dict_without_original_value = {
            k: v for k, v in self.__dict__.items() if k != key
        }
        try:
            self.__fields__[key].validate(value, dict_without_original_value, loc=key)
        except KeyError as e:
            if not key.startswith("_"):
                raise ValueError(f'"{self.__class__.__name__}" object has no field "{key}"') from e
        self.check_type(key, value)

        if not isinstance(value, StepIO):
            object.__setattr__(self, key, value)
        elif object.__getattribute__(self, key)._input_mode:
            object.__getattribute__(self, key)._ref = value
        else:
            # do set stepIO, rather copy its attributes.
            self_io = getattr(self, key)
            for f in value.__fields__:
                if f.startswith("_"):
                    continue
                object.__setattr__(self_io, f, getattr(value, f))

    def __getattribute__(self, key):
        if key.startswith("_"):
            return object.__getattribute__(self, key)
        else:
            v = object.__getattribute__(self, key)
            if not isinstance(v, StepIO):
                return v
            if object.__getattribute__(self, key)._ref:
                return object.__getattribute__(self, key)._ref
            else:
                return object.__getattribute__(self, key)

    def __str__(self):
        ret = []
        for f in self.__fields__:
            ret.append("%s=%s" % (f, str(getattr(self, f))))
        return " ".join(ret)


    @classmethod
    def from_dict(cls, d):
        res_data = {}
        for k, v in d.items():
            res_data[k] = cls.__fields__[k].type_.parse_obj(v)
        return cls(**res_data)


class StepIOsListMeta(pydantic.main.ModelMetaclass):
    def __new__(mcs, name, bases, attrs, ios_type=StepIOs):
        io_list_model_data = {
            "list_data": (List[ios_type], []),
            "_ios_type": ios_type,
            "__base__": StepIOList
        }
        IOList = pydantic.create_model("%s_list" % (name,), **io_list_model_data)
        setattr(pydantic.main, "%s_list" % (name,), IOList)
        attrs['_ios_type'] = ios_type
        attrs.setdefault('__annotations__', {})
        attrs['__annotations__']['data'] = IOList
        attrs['data'] = IOList()
        attrs['ListModel'] = IOList
        return super().__new__(mcs, name, bases, attrs)


class StepIOsListBase(NoCopyModel, metaclass=StepIOsListMeta):

    def __init__(self, **data: Any) -> None:
        for k, v in data.items():
            self.check_type(k, v)

        no_io_data = {}
        for k in data.keys():
            if not isinstance(data[k], StepIO):
                no_io_data[k] = data[k]

        super().__init__(**no_io_data)
        for k, v in data.items():
           setattr(self, k, v)

    def __str__(self):
        ret = []
        for f in self.__fields__:
            ret.append("%s=%s" % (f, str(getattr(self, f))))
        return " ".join(ret)

    def __getattribute__(self, key):
        if key.startswith("_"):
            return object.__getattribute__(self, key)
        else:
            v = object.__getattribute__(self, key)
            if not isinstance(v, StepIO):
                return v
            if object.__getattribute__(self, key)._ref:
                return object.__getattribute__(self, key)._ref
            else:
                return object.__getattribute__(self, key)

    def __setattr__(self, key, value):
        """Override setattr to make sure assigning to step.results doesn't break
        existing references for step.results"""
   
        dict_without_original_value = {
            k: v for k, v in self.__dict__.items() if k != key
        }
        try:
            self.__fields__[key].validate(value, dict_without_original_value, loc=key)
        except KeyError as e:
            if not key.startswith("_"):
                raise ValueError(f'"{self.__class__.__name__}" object has no field "{key}"') from e
        self.check_type(key, value)

        if not isinstance(value, StepIO):
            object.__setattr__(self, key, value)
        elif object.__getattribute__(self, key)._input_mode:
            object.__getattribute__(self, key)._ref = value
        else:
            # do set stepIO, rather copy its attributes.
            self_io = getattr(self, key)
            for f in value.__fields__:
                if f.startswith("_"):
                    continue
                object.__setattr__(self_io, f, getattr(value, f))

    @classmethod
    def from_dict(cls, d):
        res_data = {}
        for k, v in d.items():
            res_data[k] = cls.__fields__[k].type_.parse_obj(v)
        return cls(**res_data)

    def input_mode(self, mode: bool = True):
        for f in self.__fields__.keys():
            v = getattr(self, f)
            v.input_mode(mode=mode)

    def set_step(self, step):
        self.data._step = step


class StepIOsList(StepIOsListMeta):
    lists = {}
    def __new__(mcs, ios_type):
        if ios_type not in mcs.lists:
            ret = super().__new__(
                mcs, 
                "%sList" % ios_type.__name__,
                (StepIOsListBase, ), 
                {"_ios_type": ios_type},
                ios_type=ios_type)
            mcs.lists[ios_type] = ret
            setattr(abc, "%sList" % ios_type.__name__, ret)
            return ret
        else:
            return mcs.lists[ios_type]



class NoInputs(StepIOs):
    pass


ResultsType = TypeVar("ResultsType", bound=Union[StepIOs, StepIOsListBase])
InputsType = TypeVar("InputsType", bound=Union[StepIOs, StepIOsListBase])
DetailsType = TypeVar("DetailsType", bound=StepDetails)


class StepErrors(pydantic.BaseModel):
    """Class to store results of step."""

    errors: Dict[Any, Any] = {}


class StepStats(pydantic.BaseModel):
    started: Optional[str]
    finished: Optional[str]
    skipped: bool


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

        if key in ("inputs", "results"):
            dict_without_original_value = {
                k: v for k, v in self.__dict__.items() if k != key
            }
            self.__fields__[key].validate(value, dict_without_original_value, loc=key)
            self_field = getattr(self, key)
            if issubclass(type(self_field), StepIOs):
                for f in self_field.__fields__:
                    setattr(self_field, f, getattr(value, f))
            else: # StepIOsList
                if key == "results":
                    self_field.data.input_mode(mode=False)
                else:
                    self_field.data.input_mode(mode=True)
                for (n, item) in enumerate(value.data):
                    self_field.data[n] = item


        else:
            super().__setattr__(key, value)

class StepMeta(pydantic.main.ModelMetaclass):
    def __new__(mcs, name, bases, namespace, **kwargs):

        ret = super().__new__(mcs, name, bases, namespace, **kwargs)
        return ret


class Step(
    BaseTractionModel,
    Generic[ResultsType, ArgsType, ExtResourcesType, InputsType, DetailsType],
    metaclass=StepMeta,
    validate_all=False,
    allow_population_by_field_name=False,
    extra=pydantic.Extra.forbid,
    underscore_attrs_are_private=False,
    validate_assignment=True,
):
    TYPE: ClassVar[str] = "Step"
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
        resources: ExtResourcesType,
        inputs: InputsType
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

        if not issubclass(inputs_type, StepIOs) and not issubclass(inputs_type, StepIOsListBase):
            raise TypeError(
                "Step inputs type has to be subclass of StepIOs or StepIOsList"
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
        results.input_mode(mode=False)
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
            inputs=inputs,
            args=args,
            errors=StepErrors(),
            stats=stats,
        )

        self.inputs = inputs
        self.results = results
        self.results.set_step(self)

    @property
    def fullname(self) -> str:
        """Full name of class instance."""
        return "%s[%s]" % (self.NAME, self.uid)

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
        except Exception as e:
            self.state = StepState.ERROR
            _on_error(self)
            self.errors.errors['exception'] = str(e)
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
        ret['errors'] = self.errors.dict()
        ret["results"] = self.results.dict(exclude={"step"})
        ret["resources"] = self.resources.dump(full=full)
        for f, ftype in self.inputs.__fields__.items():
            field = getattr(self.inputs, f)
            if field._step:
                ret["inputs"][f] = getattr(self.inputs, f)._step.fullname
            else:
                ret["inputs_standalone"][f] = getattr(self.inputs, f).dict(
                    exclude={"_step"}
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
        self.results = self.results.from_dict(step_dump["results"])
        loaded_args = {}
        for f in self.args.__fields__:
            if isinstance(getattr(self.args, f), Secret):
                loaded_args[f] = getattr(self.args, f)
            else:
                loaded_args[f] = step_dump["args"][f]

        self.args = self.args.parse_obj(loaded_args)
        self.errors = step_dump["errors"]
        self.stats = step_dump["stats"]
        self.results.set_step(self)
        self.resources.from_dict(
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
        ret.results = ret.results.from_dict(step_dump["results"])
        ret.errors = step_dump["errors"]
        ret.stats = step_dump["stats"]
        ret.results.step = ret

        return ret

#Step.update_forward_refs(**locals())

class StepNGMeta(pydantic.main.ModelMetaclass):
    def __new__(mcs, name, bases, namespace,
                results_type: Type[Union[StepIOs, StepIOsListBase]] = StepIOs,
                inputs_type: Type[Union[StepIOs, StepIOsListBase]] = StepIOs,
                args_type: Type[StepArgs] = StepArgs,
                resources_type: Type[ExtResources] = ExtResources,
                details_type: Type[StepDetails] = StepDetails,
                **kwargs):

        if not check_type(results_type, StepIOs) and not check_type(results_type, StepIOsListBase):
            raise TypeError("results_type has to be subclass of StepIOs or StepIOsListBase")

        if not check_type(inputs_type, StepIOs) and not check_type(results_type, StepIOsListBase):
            raise TypeError("inputs_type has to be subclass of StepIOs or StepIOsListBase")

        if not check_type(args_type, StepArgs):
            raise TypeError("args_type has to be subclass of StepArgs")

        if not check_type(resources_type, ExtResources):
            raise TypeError("args_type has to be subclass of ExtResources")

        if not check_type(details_type, StepDetails):
            raise TypeError("args_type has to be subclass of StepDetails")

        if name != "StepNG":
            if results_type is StepIOs:
                raise TypeError("Cannot use abstract StepIOs as results_type")

            if inputs_type is StepIOs:
                raise TypeError("Cannot use abstract StepIOs as inputs_type")

            if args_type is StepArgs:
                raise TypeError("Cannot use abstract StepArgs as args_type")

            if resources_type is ExtResources:
                raise TypeError("Cannot use abstract ExtResources as resources_type")

            if details_type is StepDetails:
                raise TypeError("Cannot use abstract StepDetails as details_type")

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
    results_type=StepIOs,
    inputs_type=StepIOs,
    args_type=StepArgs,
    resources_type=ExtResources,
    details_type=StepDetails,

    validate_all=True,
    allow_population_by_field_name=False,
    extra=pydantic.Extra.forbid,
    underscore_attrs_are_private=False,
    validate_assignment=True,
):
    TYPE: ClassVar[str] = "Step"
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
        inputs: Union[StepIOs, StepIOsListBase]
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
        results.input_mode(mode=False)
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

        #self.inputs = inputs
        for f in self.results.__fields__:
            getattr(self.results, f)._step = self

    @property
    def fullname(self) -> str:
        """Full name of class instance."""
        return "%s[%s]" % (self.NAME, self.uid)

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
        except Exception as e:
            self.state = StepState.ERROR
            self.errors.errors['exception'] = str(e)
            _on_error(self)
            raise
        else:
            self.state = StepState.FINISHED
        finally:
            self._finish_stats()
            _on_update(self)  # type: ignore
        return self

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
        ret['errors'] = self.errors.dict()
        ret["resources"] = self.resources.dump(full=full)
        for f, ftype in self.inputs.__fields__.items():
            field = getattr(self.inputs, f)
            if issubclass(ftype.type_, StepIOs) and field.step:
                ret["inputs"][f] = getattr(self.inputs, f).step.fullname
            elif issubclass(ftype.type_, StepIOsList) and field.data.step:
                ret["inputs"][f] = getattr(self.inputs, f).data.step.fullname
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

StepNG.update_forward_refs()


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
    _current_step: Union[Step[Any, Any, Any, Any, Any], StepNG, Tractor, STMD, None] = pydantic.PrivateAttr()
    step_map: Dict[str, Type[Step[Any, Any, Any, Any, Any]]] = {}
    resources_map: Dict[str, Type[ExtResource]] = {}
    uid: str

    @property
    def current_step(self):
        return self._current_step


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
        #_current_step = None
        super().__init__(
            uid=uid,
        )
        self._current_step = None

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
            "current_step": self._current_step.fullname if self._current_step else None}
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
            self._current_step = step
            step.run(on_update=on_update, on_error=on_error)

Tractor.update_forward_refs()

class NamedTractorMeta(pydantic.main.ModelMetaclass):
    def __new__(mcs, name, bases, namespace,
                nt_steps: List[Tuple[str, Union[Step[Any, Any, Any, Any, Any], NamedTractor]]] = [],
                nt_args: StepArgs = StepArgs,
                nt_resources: ExtResources = ExtResources,
                nt_results: Union[StepIOs, StepIOsListBase] = StepIOs,
                nt_inputs: Union[StepIOs, StepIOsListBase] = StepIOs,
                nt_details: StepDetails = NoDetails,
                **kwargs):

        nt_step_args = {}
        nt_step_results = {}
        nt_step_inputs = {}
        nt_step_details = {}
        nt_step_resources = {}
        steps_inputs_model = {}
        steps_args_model = {}
        steps_resources_model = {}

        for (nt_step_name, nt_step) in nt_steps:
            if issubclass(nt_step, Step):
                nt_step_types = get_step_types(nt_step)
                results_type = nt_step_types[0]
                args_type = nt_step_types[1]
                resources_type = nt_step_types[2]
                inputs_type = nt_step_types[3]
                details_type = nt_step_types[4]
                if not issubclass(inputs_type, StepIOs) and not issubclass(inputs_type, StepIOsListBase):
                    raise TypeError("Step %s inputs type %s is not sub type of StepIOs or StepIOsListBase" % (nt_step, inputs_type))

            elif issubclass(nt_step, NamedTractor) or issubclass(nt_step, STMD):
                results_type = nt_step.ResultsModel
                args_type = nt_step.ArgsModel
                resources_type = nt_step.ResourcesModel
                inputs_type = nt_step.InputsModel
                details_type = nt_step.DetailsModel
            elif issubclass(nt_step, StepNG):
                results_type = nt_step.ResultsModel
                args_type = nt_step.ArgsModel
                resources_type = nt_step.ResourcesModel
                inputs_type = nt_step.InputsModel
                details_type = nt_step.DetailsModel

            nt_step_args[nt_step_name] = {k: v.type_ for k,v in args_type.__fields__.items()}
            nt_step_results[nt_step_name] = {k: v.type_ for k,v in results_type.__fields__.items()}
            nt_step_resources[nt_step_name] = {k: v.type_ for k,v in resources_type.__fields__.items()}
            nt_step_inputs[nt_step_name] = {k: v.type_ for k,v in inputs_type.__fields__.items()}
            nt_step_details[nt_step_name] = details_type

            steps_inputs_model[nt_step_name] = inputs_type
            steps_args_model[nt_step_name] = args_type
            steps_resources_model[nt_step_name] = resources_type

        namespace.setdefault('__annotations__', {})
        #namespace['__annotations__']['InputsModel'] = nt_inputs
        namespace['InputsModel'] = nt_inputs
        #namespace['__annotations__']['ResultsModel'] = nt_results
        namespace['ResultsModel'] = nt_results
        #namespace['__annotations__']['ArgsModel'] = nt_args
        namespace['ArgsModel'] = nt_args
        #namespace['__annotations__']['DetailsModel'] = nt_details
        namespace['DetailsModel'] = nt_details
        #namespace['__annotations__']['ResourcesModel'] = nt_resources
        namespace['ResourcesModel'] = nt_resources

        namespace['_steps_inputs'] = steps_inputs_model
        namespace['_steps_args'] = steps_args_model
        namespace['_steps_resources'] = steps_resources_model
        namespace['_nt_steps'] = nt_steps

        namespace['__annotations__']['args'] = nt_args
        namespace['__annotations__']['results'] = nt_results
        namespace['__annotations__']['resources'] = nt_resources
        namespace['__annotations__']['inputs'] = nt_inputs
        namespace['__annotations__']['details'] = nt_details

        if not inspect.isclass(nt_inputs):
            raise TypeError("nt_inputs has to be class")
        if not inspect.isclass(nt_results):
            raise TypeError("nt_results has to be class")

        if not issubclass(nt_inputs, StepIOs) and not issubclass(nt_inputs, StepIOsListBase):
            raise TypeError("nt_inputs has to be StepIO or StepIOsListBase subclass")
        if not issubclass(nt_results, StepIOs) and not issubclass(nt_results, StepIOsListBase):
            raise TypeError("nt_results has to be StepIOs  or StepIOsListBase subclass")

        if "NAME" not in namespace:
            raise TypeError("Tractor has to have NAME attribute")

        if "INPUTS_MAP" not in namespace:
            raise TypeError("Missing INPUTS_MAP definition")
        if "ARGS_MAP" not in namespace:
            raise TypeError("Missing ARGS_MAP definition")
        if "RESULTS_MAP" not in namespace:
            raise TypeError("Missing RESULTS_MAP definition")
        if "RESOURCES_MAP" not in namespace:
            raise TypeError("Missing RESOURCES_MAP definition")


        for step_name, inputs_map in namespace['INPUTS_MAP'].items():
            if not isinstance(step_name, str):
                raise TypeError("INPUTS_MAP %s has to be string" % step_name)
            if not isinstance(inputs_map, dict):
                raise TypeError("INPUTS_MAP value for step %s has to be dictionary" % step_name)
            if step_name not in nt_step_inputs:
                raise TypeError("Tractor doesn't have step %s mentioned in INPUTS_MAP" % step_name)
            for input_name, output_map in inputs_map.items():
                if input_name not in nt_step_inputs[step_name]:
                    raise TypeError("Step %s doesn't have input '%s' (%s)" % (step_name, input_name, nt_step_inputs[step_name]))
                if not isinstance(output_map, NTInput) and not isinstance(output_map, tuple):
                    raise TypeError("INPUTS_MAP->%s->%s value has to be tuple" % (step_name, input_name))
                if isinstance(output_map, NTInput):
                    if output_map.name not in nt_inputs.__fields__:
                        raise TypeError("output step INPUTS_MAP->%s->%s = NTInput(%s), %s is not in %s " % (step_name, input_name, output_map.name, output_map.name, nt_inputs))
                else:
                    output_step, output_key = output_map
                    if output_key not in nt_step_results[output_step]:
                        raise TypeError("output '%s' is not in step '%s' (%s)" % (output_key, output_step, nt_step_results[output_step].keys()))
                    if output_step not in nt_step_results:
                        raise TypeError("output step INPUTS_MAP->%s->%s = (%s, ...) is not in tractor steps" % (step_name, input_name, output_step))
                    if nt_step_inputs[step_name][input_name] != nt_step_results[output_step][output_key]:
                        raise TypeError("output step INPUTS_MAP->%s->%s = (%s, %s) result type is %s but %s is required" % (
                            step_name, input_name, output_step, output_key, nt_step_results[output_step][output_key],
                            nt_step_inputs[step_name][input_name]
                            )
                        )

        for step_name, args_map in namespace['ARGS_MAP'].items():
            if not isinstance(step_name, str):
                raise TypeError("ARGS_MAP %s has to be string" % step_name)
            if not isinstance(args_map, dict):
                raise TypeError("ARGS_MAP value for step %s has to be dictionary" % step_name)
            if step_name not in nt_step_args:
                raise TypeError("Tractor doesn't have step %s mentioned in ARGS_MAP" % step_name)
            for sarg, targ in args_map.items():
                if sarg not in nt_step_args[step_name]:
                    raise TypeError("Step %s doesn't have argument %s" % (step_name, sarg))
                if targ not in nt_args.__fields__:
                    raise TypeError("Tractor doesn't have argument %s (%s)" % (targ, list(nt_args.__fields__.keys())))
                if not issubclass(nt_step_args[step_name][sarg], nt_args.__fields__[targ].type_):
                    raise TypeError("Step argument %s (%s) is not subtype of tractor argument %s" % (targ, step_name, sarg))

        for result_io, results_map in namespace['RESULTS_MAP'].items():
            if result_io not in nt_results.__fields__:
                raise TypeError("Tractor doesn't have result '%s'" % (result_io))
            if not isinstance(results_map, tuple):
                raise TypeError("RESULTS_MAP value has to be tuple")
            (step_name, step_result) = results_map
            if step_name not in nt_step_results:
                raise TypeError("Tractor doesn't have step %s mentioned in RESULTS_MAP" % step_name)
            if step_result not in nt_step_results[step_name]:
                raise TypeError("Step %s doesn't have result %s" % (step_name, step_result))
            if not issubclass(nt_step_results[step_name][step_result], nt_results.__fields__[result_io].type_):
                raise TypeError("Step result '%s'(%s) (%s) is not subtype of tractor result '%s' (%s)" % (
                        step_result, nt_step_results[step_name][step_result], step_name, result_io, nt_results.__fields__[result_io].type_)
                    )

        for step_name, resources_map in namespace['RESOURCES_MAP'].items():
            if not isinstance(step_name, str):
                raise TypeError("RESOURCES_MAP %s has to be string" % step_name)
            if not isinstance(resources_map, dict):
                raise TypeError("RESOURCES_MAP value for step %s has to be dictionary" % step_name)
            if step_name not in nt_step_resources:
                raise TypeError("Tractor doesn't have step %s mentioned in RESOURCES_MAP" % step_name)
            for sresource, tresource in resources_map.items():
                if sresource not in nt_step_resources[step_name]:
                    raise TypeError("Step %s doesn't have resource %s" % (step_name, sresource))
                if tresource not in nt_resources.__fields__:
                    raise TypeError("Tractor doesn't have resource %s (%s)" % (tresource, list(nt_resources.__fields__.keys())))
                if not issubclass(nt_step_resources[step_name][sresource], nt_resources.__fields__[tresource].type_):
                    raise TypeError("Step resource %s (%s) is not subtype of tractor resource %s" % (tresource, step_name, sresource))


        ret = super().__new__(mcs, name, bases, namespace)
        return ret


class NTInput(pydantic.BaseModel):
    name: str


class NamedTractor(NoCopyModel, metaclass=NamedTractorMeta, nt_steps=[], nt_inputs=StepIOsList(StepIOs)):
    """Class which runs sequence of steps."""

    _step_map: Dict[str, Union[Step[Any, Any, Any, Any, Any],NamedTractor]] = {}
    steps: List[Union[StepNG, Step[Any, Any, Any, Any, Any],NamedTractor]] = []
    _current_step: Union[StepNG, Step[Any, Any, Any, Any, Any],NamedTractor, STMD, None] = None # TODO: fix
    _step_name_by_uid: Dict[str, Union[Step[Any, Any, Any, Any, Any],NamedTractor]]
    uid: str
    state: StepState = StepState.READY
    TYPE: ClassVar[str] = "Tractor"

    INPUTS_MAP: ClassVar[Dict[str, Dict[str, str]]] = {}
    ARGS_MAP: ClassVar[Dict[str, Dict[str, str]]] = {}
    DETAILS_MAP: ClassVar[Dict[str, Dict[str, str]]] = {}
    RESULTS_MAP: ClassVar[Dict[str, Dict[str, str]]] = {}
    RESOURCES_MAP: ClassVar[Dict[str, Dict[str, str]]] = {}
    NAME: ClassVar[str] = "NamedTractor"

    def __init__(self, *args, **kwargs):
        results_model_data = {}
        details_model_data = {}

        results = self.ResultsModel(**results_model_data)
        results.input_mode(mode=False)
        #results.input_mode(mode=False)
        kwargs['results'] = results
        kwargs['details'] = self.DetailsModel()
        super().__init__(**kwargs)
        self._step_name_by_uid = {}
        #self.inputs = kwargs['inputs']

        i = 0
        for step_name, step_type in self._nt_steps:
            i += 1
            step_inputs = self.INPUTS_MAP[step_name]
            step_args_map = self.INPUTS_MAP[step_name]
            inputs_model_data = {}
            args_model_data = {}
            resources_model_data = {}

            for sarg, targ in self.ARGS_MAP[step_name].items():
                args_model_data[sarg] = getattr(self.args, targ)

            for sresource, tresource in self.RESOURCES_MAP[step_name].items():
                resources_model_data[sresource] = getattr(self.resources, tresource)

            for input_, output_from in step_inputs.items():
                if isinstance(output_from, tuple):
                    output_step, output_key = output_from
                    if output_step not in self._step_map:
                        raise ValueError("(%s) Required step '%s' is not in tractor steps" % (self.NAME, output_step))
                    inputs_model_data[input_] = getattr(self._step_map[output_step].results, output_key)
                else: # NTInput
                    inputs_model_data[input_] = getattr(self.inputs, output_from.name)


            inputs_model = self._steps_inputs[step_name](**inputs_model_data)

            args_model = self._steps_args[step_name](**args_model_data)
            resources_model = self._steps_resources[step_name](**resources_model_data)

            step = step_type(uid='%s:%d.%s' % (self.uid, i, step_name),
                          args=args_model,
                          resources=resources_model,
                          inputs=inputs_model)
            self._step_name_by_uid[step.uid] = step_name

            self.steps.append(step)

            self._step_map[step_name] = step
        for res_key, res_map in self.RESULTS_MAP.items():
            (res_step, res_step_key) = res_map
            setattr(self.results, res_key, getattr(self._step_map[res_step].results, res_step_key))


        details = self.DetailsModel(**details_model_data)
        self._current_step = None

        results = self.ResultsModel(**results_model_data)
        # update step reference in individual results
        for f in self.results.__fields__:
            getattr(self.results, f)._step = self

        self.details = details
        self.state = StepState.READY

    @property
    def fullname(self) -> str:
        """Full name of class instance."""
        return "%s[%s]" % (self.NAME, self.uid)

    def _dump_resources(self, full: bool) -> TractorDumpDict:
        """Dump stepper state and shared_results to json compatible dict."""
        resources_dump: Dict[str, Any] = {}
        for step in self.steps:
            if isinstance(step, Step):
                for k in step.resources.__fields__:
                    res = getattr(step.resources, k)
                    resources_dump[res.fullname] = res.dump()
            elif isinstance(step, StepNG):
                for k in step.resources.__fields__:
                    res = getattr(step.resources, k)
                    resources_dump[res.fullname] = res.dump()
            elif isinstance(step, STMD):
                for k in step.resources.__fields__:
                    res = getattr(step.resources, k)
                    resources_dump[res.fullname] = res.dump()
            else: # named tractor
                _resources = step._dump_resources(full=full)
                for res_fullname, res in _resources.items():
                    resources_dump[res_fullname] = res
        return resources_dump

    def dump(self, full: bool) -> TractorDumpDict:
        """Dump stepper state and shared_results to json compatible dict."""
        steps: List[Dict[str, Any]] = []
        out: TractorDumpDict = {
            "state": self.state.value,
            "steps": steps,
            "resources": {},
            "uid": self.uid,
            'results': self.results.dict(),
            "current_step": self._current_step.fullname if self._current_step else None}
        resources_dump: Dict[str, Any] = {}
        for step in self.steps:
            if isinstance(step, Step):
                steps.append({"type": "step", "data": step.dump(full=False)})
            elif isinstance(step, StepNG):
                steps.append({"type": "stepng", "data": step.dump(full=False)})
            elif isinstance(step, STMD):
                steps.append({"type": "stmd", "data": step.dump()})
            else:
                steps.append({"type": "tractor", "data": step.dump(full=False)})
        out["resources"] = self._dump_resources(full=full)
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
            self._current_step = step

            step.run(on_update=on_update, on_error=on_error)
            for res_key, res_map in self.RESULTS_MAP.items():
                (res_step, res_step_key) = res_map
                if self._step_name_by_uid[step.uid] == res_step:
                    setattr(self.results, res_key, getattr(step.results, res_step_key))


            if step.state == StepState.ERROR:
                self.state = StepState.ERROR
                if on_error:
                    on_error(self)
                return self
            if step.state == StepState.FAILED:
                self.state = StepState.FAILED
                if on_update:
                    on_update(self)
                return self

        self.state = StepState.FINISHED
        return self

NamedTractor.update_forward_refs()


class STMDMeta(pydantic.main.ModelMetaclass):
    def __new__(mcs, name, bases, namespace,
                tractor_type: Type[NamedTractor],
                **kwargs):

        results_model = {}
        inputs_model = {}
        inputs_as_results_model = {}
        args_model = {}
        details_model = {}
        stmd_io_res_model = {}
        stmd_io_res_data = {}
        stmd_io_ins_model = {}
        stmd_io_ins_data = {}


        args_model["__base__"] = tractor_type.ArgsModel
        details_model["__base__"] = StepDetails

        args_model["tractors"] = (int, ...)
        ArgsModel = pydantic.create_model("%s_args" % (name,), **args_model)
        setattr(pydantic.main, "%s_args" % (name,), ArgsModel)
        ArgsModel.update_forward_refs(**globals())

        details_model["data"] = (List[tractor_type.DetailsModel], [])
        DetailsModel = pydantic.create_model("%s_details" % (name,), **details_model)
        setattr(pydantic.main, "%s_details" % (name,), DetailsModel)
        DetailsModel.update_forward_refs(**globals())

        namespace.setdefault('__annotations__', {})
        namespace['__annotations__']['results'] = StepIOsList(tractor_type.ResultsModel)
        namespace['__annotations__']['inputs'] = StepIOsList(tractor_type.InputsModel)
        namespace['__annotations__']['args'] = ArgsModel
        namespace['__annotations__']['resources'] = tractor_type.ResourcesModel
        namespace['__annotations__']['details'] = DetailsModel
        namespace['__annotations__']['tractions'] = List[Union[Step, StepNG, NamedTractor, STMD]]

        namespace['ResultsModel'] = StepIOsList(tractor_type.ResultsModel)
        namespace['InputsModel'] = StepIOsList(tractor_type.InputsModel)
        namespace['TractorType'] = tractor_type
        namespace['ArgsModel'] = ArgsModel
        namespace['ResourcesModel'] = tractor_type.ResourcesModel
        #namespace['MultiDataModel'] = MultiDataModel
        namespace['MultiDataInput'] = StepIOsList(tractor_type.InputsModel)
        namespace['DetailsModel'] = DetailsModel
        namespace['results'] = StepIOsList(tractor_type.ResultsModel)()
        ret = super().__new__(mcs, name, bases, namespace)
        return ret


class STMD(
    NoCopyModel, metaclass=STMDMeta, tractor_type=NamedTractor
):
    TYPE: ClassVar[str] = "STMD"
    uid: str
    state: StepState

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
            details=self.DetailsModel(),
            state=StepState.READY,
            tractions=[]
        )

        self.results.input_mode(mode=False)
        # update step reference in individual results
        for f in self.results.__fields__:
            getattr(self.results, f)._step = self


    @property
    def fullname(self) -> str:
        """Full name of class instance."""
        return "%s[%s]" % (self.NAME, self.uid)

    def run(
        self,
        on_error: StepOnErrorCallable = None,
        on_update: StepOnUpdateCallable = None,
    ) -> None:
        _on_update: StepOnUpdateCallable = lambda step: None
        if on_update:
            _on_update = on_update
    
        self.state = StepState.RUNNING
        
        with ProcessPoolExecutor(max_workers=self.args.tractors) as executor:
            ft_results = {}
            nts = []
            for i in range(0, len(self.inputs.data)):
                nt = self.TractorType(uid="%s:%d" % (self.uid, i),
                                 args=self.args,
                                 resources=self.resources,
                                 inputs=self.inputs.data[i])
                self.details.data.append(self.TractorType.DetailsModel())
                self.details.data[i] = nt.details
                self.tractions.append(nt)
                res = executor.submit(nt.run, on_update=on_update, on_error=on_error)
                ft_results[res] = (nt, i)
                self.results.data.append(self.TractorType.ResultsModel())
            _on_update(self)
            for ft in as_completed(ft_results):
                (_, i) = ft_results[ft]
                nt = ft.result()
                self.tractions[i] = nt
                self.results.data[i] = nt.results
                self.details.data[i] = nt.details
                _on_update(self)

        self.state = StepState.FINISHED

    def dump(self, full=False):
        return self.dict()


STMD.update_forward_refs()



class NoArgs(StepArgs):
    pass


class NoResources(ExtResources):
    NAME: ClassVar[str] = "NoResources"



class NoResults(StepIOs):
    pass
