import abc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import datetime
import inspect
import json

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
    GenericAlias,
    Union,
    ForwardRef,
    get_origin,
)
import typing_inspect


import enum

import dataclasses

from .exc import (
    LoadWrongStepError,
    LoadWrongExtResourceError,
    MissingSecretError,
    DuplicateStepError,
    DuplicateTractorError,
)

from .utils import (
    get_type, _get_args, check_type
)

Validator = Callable[Any, Any]

Step = ForwardRef("Step")

Tractor = ForwardRef("Tractor")

NamedTractor = ForwardRef("NamedTractor")

STMD = ForwardRef("STMD")

def empty_on_error_callback() -> None:
    return None


def isodate_now() -> str:
    """Return current datetime in iso8601 format."""
    return "%s%s" % (datetime.datetime.utcnow().isoformat(), "Z")


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

    READY = "ready"
    PREP = "prep"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    ERROR = "error"


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


class NoArgs(StepArgs):
    pass


class ExtResource(pydantic.generics.BaseModel):
    """Step Resource class.

    Use SECRETS class variable to mask attributes at the output.
    """

    NAME: ClassVar[str] = "ExtResource"
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
    NAME: ClassVar[str] = "ExtResources"

    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            if key == "uid":
                continue
            if not issubclass(type(val), ExtResource):
                raise ValueError("%s has to be type ExtResource not %s" % (key, type(val)))

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
            setattr(self, key, getattr(self, key).load(val))
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


class NoResources(ExtResources):
    NAME: ClassVar[str] = "NoResources"


class DefaultsModelMeta(pydantic.main.ModelMetaclass):
    """Metaclass to ensure all defined attributes in StepIOs have default value."""

    NO_ANNOTATION: ClassVar[List[str]] = [
        "dump",
        "load",
        "Config",
        "dict",
        "input_mode",
        "_ios_type",
        "json"
    ]

    def __new__(cls, name, bases, attrs):
        annotations = attrs.get("__annotations__", {})
        NO_ANNOTATION = attrs.get("NO_ANNOTATION", cls.NO_ANNOTATION)
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
    pydantic.generics.BaseModel,
    metaclass=DefaultsModelMeta,
):
    class Config:
        extra = pydantic.Extra.forbid
        validate_assignment = True
        copy_on_model_validation = False


class StepIO(RequiredDefaultsModel):
    """Class to store results of step."""

    _step: Optional[Union[STMD, NamedTractor]] = pydantic.PrivateAttr(default=None)
    _input_mode: bool = pydantic.PrivateAttr(default_factory=lambda: True)
    _ref: Optional["StepIO"] = pydantic.PrivateAttr(default=None)
    _self_name: str = pydantic.PrivateAttr(default="")

    def __setattr__(self, key, value):
        """Override setattr to make sure assigning to step.results doesn't break
        existing references for step.results"""

        if key == "step" and value:
            dict_without_original_value = {k: v for k, v in self.__dict__.items() if k != key}
            self.__fields__["step"].validate(value, dict_without_original_value, loc=key, cls=Step)

            object.__setattr__(self, key, value)
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
            return super().dict(exclude={"step", "_input_mode", "_ref", "dict"})
        else:
            return self._ref.dict(exclude={"step", "_input_mode", "_ref", "dict"})

    def input_mode(self, mode: bool = True):
        self._input_mode = mode


class RequiredDefaultsModelG(
    pydantic.generics.GenericModel,
    metaclass=DefaultsModelMeta,
):
    class Config:
        extra = pydantic.Extra.forbid
        validate_assignment = True
        copy_on_model_validation = False


class StepIOG(RequiredDefaultsModelG):
    """Class to store results of step."""

    _step: Optional[Union[STMD, NamedTractor]] = pydantic.PrivateAttr(default=None)
    _input_mode: bool = pydantic.PrivateAttr(default_factory=lambda: True)
    _ref: Optional["StepIO"] = pydantic.PrivateAttr(default=None)
    _self_name: str = pydantic.PrivateAttr(default="")

    def __setattr__(self, key, value):
        """Override setattr to make sure assigning to step.results doesn't break
        existing references for step.results"""

        if key == "step" and value:
            dict_without_original_value = {k: v for k, v in self.__dict__.items() if k != key}
            self.__fields__["step"].validate(value, dict_without_original_value, loc=key, cls=Step)

            object.__setattr__(self, key, value)
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
            return super().dict(exclude={"step", "_input_mode", "_ref", "__orig_class__"})
        else:
            return self._ref.dict(exclude={"step", "_input_mode", "_ref", "__orig_class__"})

    def input_mode(self, mode: bool = True):
        self._input_mode = mode


class StepIOList(StepIO):

    NO_ANNOTATION: ClassVar[List[str]] = [
        "dump",
        "load",
        "Config",
        "dict",
        "input_mode",
        "_check_type",
        "append",
        "count",
        "index",
        "insert",
        "pop",
        "reverse",
        "sort",
        "_ios_type",
        "json"
    ]

    def _check_type(self, item):
        if hasattr(self, "_ios_type"):
            if not issubclass(item, self._ios_type):
                raise TypeError("Item has to be type of %s, but is %s" % (self._ios_type, item))
        else:
            pass
            #  TODO: fix check for generics

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

    def json(self, *args, **kwargs):
        if "exclude" in kwargs:
            kwargs['exclude'].add("__orig_class__")
        else:
            kwargs['exclude'] = {"__orig_class__"}
        return super().json(*args, **kwargs)

    def dict(self, *args, **kwargs):
        if "exclude" in kwargs and kwargs['exclude']:
            kwargs['exclude'].add("__orig_class__")
        else:
            kwargs['exclude'] = {"__orig_class__"}

        ret = pydantic.BaseModel.dict(self, *args, **kwargs)
        if "__orig_class__" in ret:
            ret.pop("__orig_class__")
        return ret

TV_LIST_TYPE = TypeVar("TV_LIST_TYPE")


class StepIOListG(StepIOList, Generic[TV_LIST_TYPE]):
    list_data: List[TV_LIST_TYPE] = []

    def __class_getitem__(cls, item):
        ret = super().__class_getitem__(item)
        ret.ltype = item
        ret.__torigin__ = cls
        ret.__targs__ = (item,)
        return ret




class StepIOsMeta(pydantic.main.ModelMetaclass):
    """Metaclass to ensure all defined attributes in StepIOs have default value."""

    def __new__(cls, name, bases, attrs):
        annotations = attrs.get("__annotations__", {})
        for attrk, attrv in attrs.items():
            if attrk.startswith("__"):
                continue
            if inspect.ismethoddescriptor(attrv):
                continue
            if attrk in ("Config", "input_mode", "set_step", 'dict'):
                continue
            if attrk not in annotations:
                raise TypeError("%s has to be annotated" % attrk)
        for annotated in annotations:
            ann_cls = annotations[annotated]
            if annotated not in attrs:
                raise TypeError("Attribute %s is missing default value" % annotated)
            if get_origin(ann_cls):
                ann_cls = get_origin(ann_cls)

            if not issubclass(ann_cls, StepIO) and not issubclass(ann_cls, StepIOG):
                raise ValueError(
                    "%s attribute as to be StepIO type not %s" % (annotated, annotations[annotated])
                )

        return super().__new__(cls, name, bases, attrs)


class NoCopyModel(pydantic.generics.BaseModel):
    class Config:
        validate_all = True
        extra = pydantic.Extra.forbid
        validate_assignment = True
        copy_on_model_validation = "none"
        use_enum_values = True
        copy_on_model_validation = False

    def __setattr__(self, key, value):
        """Override setattr to make sure assigning to step.results doesn't break
        existing references for step.results"""

        dict_without_original_value = {k: v for k, v in self.__dict__.items() if k != key}
        try:
            self.__fields__[key].validate(value, dict_without_original_value, loc=key)
        except KeyError as e:
            if not key.startswith("_"):
                raise ValueError(f'"{self.__class__.__name__}" object has no field "{key}"') from e

        try:
            chcls = value.__orig_class__ if hasattr(value, "__orig_class__") else value.__class__
            if not check_type(
                    chcls,
                    self.__fields__[key].outer_type_):
                raise TypeError(
                    f"{key} is not subclass of {chcls} but is ({self.__fields__[key].outer_type_})"
                )
        except KeyError as e:
            if not key.startswith("_"):
                raise ValueError(f'"{self.__class__.__name__}" object has no field "{key}"') from e

        if not isinstance(value, StepIO):
            object.__setattr__(self, key, value)
        else:
            # do set stepIO, rather copy its attributes.
            self_io = getattr(self, key)
            for f in value.__fields__:
                object.__setattr__(self_io, f, getattr(value, f))

    def __init__(self, **data: Any) -> None:
        for k, v in data.items():
            if not check_type(
                    v.__orig_class__ if hasattr(v, "__orig_class__") else v.__class__,
                    self.__fields__[k].outer_type_):


                raise TypeError(
                    f"{k} is not subclass of ({self.__fields__[k].outer_type_}) but is {type(v)}"
                )
        super().__init__(**data)
        for k, v in data.items():
            setattr(self, k, v)


class NoCopyModelG(pydantic.generics.GenericModel):
    class Config:
        validate_all = True
        extra = pydantic.Extra.forbid
        validate_assignment = True
        copy_on_model_validation = "none"
        use_enum_values = True
        copy_on_model_validation = False

    def __setattr__(self, key, value):
        """Override setattr to make sure assigning to step.results doesn't break
        existing references for step.results"""

        dict_without_original_value = {k: v for k, v in self.__dict__.items() if k != key}
        try:
            self.__fields__[key].validate(value, dict_without_original_value, loc=key)
        except KeyError as e:
            if not key.startswith("_"):
                raise ValueError(f'"{self.__class__.__name__}" object has no field "{key}"') from e
        chcls = value.__orig_class__ if hasattr(value, "__orig_class__") else value.__class__
        if not check_type(
                chcls,
                self.__fields__[key].outer_type_):
            raise TypeError(
                f"{key} is not subclass of {chcls} but is ({self.__fields__[key].outer_type_})"
            )

        if not isinstance(value, StepIO):
            object.__setattr__(self, key, value)
        else:
            # do set stepIO, rather copy its attributes.
            self_io = getattr(self, key)
            for f in value.__fields__:
                object.__setattr__(self_io, f, getattr(value, f))

    def __init__(self, **data: Any) -> None:
        for k, v in data.items():
            chcls = v.__orig_class__ if hasattr(v, "__orig_class__") else v.__class__
            if not check_type(
                    chcls,
                    self.__fields__[k].outer_type_):
                raise TypeError(
                    f"{k} is not subclass of {chcls} but is ({self.__fields__[k].outer_type_})"
                )
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
        if not isinstance(v, StepIO) and not isinstance(v, StepIOG):
            raise ValueError("field must be StepIOs subclass, but is %s" % type(v))
        return v

    def __init__(self, **data: Any) -> None:
        for k, v in data.items():
            if k not in self.__fields__:
                raise AttributeError(f"{type(self)} doesn't have field {k}")
            vclass = v.__orig_class__ if hasattr(v, "__orig_class__") else v.__class__
            if not check_type(self.__fields__[k].outer_type_, vclass):
                raise TypeError("{k} is not subclass of {type(vclass)}")
            v._self_name = k

        no_io_data = {}
        for k in data.keys():
            if not isinstance(data[k], StepIO) and not isinstance(data[k], StepIOG):
                no_io_data[k] = data[k]

        super().__init__(**no_io_data)
        for k, v in data.items():
            setattr(self, k, v)

    def input_mode(self, mode: bool = True):
        for f in self.__fields__.keys():
            v = getattr(self, f)
            v.input_mode(mode=mode)

    def set_step(self, step):
        for f in self.__fields__:
            getattr(self, f)._step = self

    def __setattr__(self, key, value):
        """Override setattr to make sure assigning to step.results doesn't break
        existing references for step.results"""

        dict_without_original_value = {k: v for k, v in self.__dict__.items() if k != key}
        try:
            self.__fields__[key].validate(value, dict_without_original_value, loc=key)
        except KeyError as e:
            if not key.startswith("_"):
                raise ValueError(f'"{self.__class__.__name__}" object has no field "{key}"') from e
        
        vclass = value.__orig_class__ if hasattr(value, "__orig_class__") else value.__class__
        if not check_type(self.__fields__[key].outer_type_, vclass):
            raise TypeError("{k} is not subclass of {type(vclass)}")

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

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        if "exclude" in kwargs and kwargs['exclude']:
            kwargs['exclude'].add('__orig_class__')
        else:
            kwargs['exclude'] = {'__orig_class__'}
        return super().dict(*args, **kwargs)



class StepIOsListMeta(pydantic.main.ModelMetaclass):
    def __new__(mcs, name, bases, attrs, ios_type=StepIOs):
        io_list_model_data = {
            "list_data": (List[ios_type], []),
            "_ios_type": ios_type,
            "__base__": StepIOList,
        }
        IOList = pydantic.create_model("%s_list" % (name,), **io_list_model_data)
        setattr(pydantic.main, "%s_list" % (name,), IOList)
        attrs["_ios_type"] = ios_type
        attrs.setdefault("__annotations__", {})
        attrs["__annotations__"]["data"] = IOList
        attrs["data"] = IOList()
        attrs["ListModel"] = IOList
        return super().__new__(mcs, name, bases, attrs)


class StepIOsListCore:
    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
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

        dict_without_original_value = {k: v for k, v in self.__dict__.items() if k != key}
        try:
            self.__fields__[key].validate(value, dict_without_original_value, loc=key)
        except KeyError as e:
            if not key.startswith("_"):
                raise ValueError(f'"{self.__class__.__name__}" object has no field "{key}"') from e
        if not check_type(
                self.__fields__[key].outer_type_,
                value.__orig_class__ if hasattr(value, "__orig_class__") else value.__class__):
            raise TypeError(
                f"{key} is not subclass of {type(value)} but is ({self.__fields__[key].outer_type_})"
            )

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


class StepIOsListBase(StepIOsListCore, NoCopyModel, metaclass=StepIOsListMeta):
    pass


class StepIOsListBaseG(pydantic.generics.GenericModel, NoCopyModel, StepIOsListCore, Generic[TV_LIST_TYPE]):
    data: StepIOListG[TV_LIST_TYPE]# = pydantic.Field(default_factory=StepIOListG[TV_LIST_TYPE])

    def __class_getitem__(cls, item):
        ret = super().__class_getitem__(item)
        ret.ltype = item
        ret.__torigin__ = cls
        ret.__targs__ = (item,)
        return ret

    def __init__(self, **data):
        if data:
            super().__init__(data=cast(StepIOListG[self.ltype], data['data']))
        else:
            super().__init__(data=StepIOListG[self.ltype]())


class StepIOsList(StepIOsListMeta):
    lists = {}

    def __new__(mcs: str, ios_type: StepIOs) -> StepIOsListBase:
        if ios_type not in mcs.lists:
            ret = super().__new__(
                mcs,
                "%sList" % ios_type.__name__,
                (StepIOsListBase,),
                {"_ios_type": ios_type},
                ios_type=ios_type,
            )
            mcs.lists[ios_type] = ret
            setattr(abc, "%sList" % ios_type.__name__, ret)
            return ret
        else:
            return mcs.lists[ios_type]


class NoResults(StepIOs):
    pass


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


class BaseTractionModel(pydantic.generics.GenericModel):
    args: StepArgs
    inputs: StepIOs
    results: StepIOs
    details: StepDetails
    resources: ExtResources

    def __setattr__(self, key, value):
        """Override setattr to make sure assigning to step.results doesn't break
        existing references for step.results"""

        if key in ("inputs", "results"):
            dict_without_original_value = {k: v for k, v in self.__dict__.items() if k != key}
            self.__fields__[key].validate(value, dict_without_original_value, loc=key)
            self_field = getattr(self, key)
            if issubclass(type(self_field), StepIOs):
                for f in self_field.__fields__:
                    setattr(self_field, f, getattr(value, f))
            else:  # StepIOsList
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

        annotations = namespace.get("__annotations__", {})

        results_type = annotations.get("results", NoResults)
        inputs_type = annotations.get("inputs", NoInputs)
        args_type = annotations.get("args", NoArgs)
        resources_type = annotations.get("resources", NoResources)
        details_type = annotations.get("details", NoDetails)


        if isinstance(results_type, pydantic.fields.DeferredType):
            results_type = bases[0].__fields__["results"].outer_type_
            inputs_type = bases[0].__fields__["inputs"].outer_type_
            args_type = bases[0].__fields__["args"].outer_type_
            resources_type = bases[0].__fields__["resources"].outer_type_
            details_type = bases[0].__fields__["details"].outer_type_



        if (
            not check_type(results_type, StepIOs)
            and not check_type(results_type, StepIOsListBase)
            and not check_type_generics(results_type, StepIOsListBaseG)
        ):
            raise TypeError(
                "results_type has to be subclass of StepIOs, StepIOsListBase, StepIOsListBaseG. but is %s"
                % results_type
            )
        
        if (
            not check_type(inputs_type, StepIOs)
            and not check_type(inputs_type, StepIOsListBase)
            and not check_type(inputs_type, StepIOsListBaseG[ANY])
        ):
            raise TypeError(
                "inputs_type has to be subclass of StepIOs or StepIOsListBase, is %s"
                % type(results_type)
            )

        if name != "Step" and not check_type(args_type, StepArgs):
            raise TypeError("args_type has to be subclass of StepArgs")

        if name != "Step" and not check_type(resources_type, ExtResources):
            raise TypeError("args_type has to be subclass of ExtResources")

        if name != "Step" and not check_type(details_type, StepDetails):
            raise TypeError("args_type has to be subclass of StepDetails")

        if name != "Step":
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

        return ret


class Step(
    BaseTractionModel,
    metaclass=StepMeta,
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
    stats: StepStats

    args: StepArgs
    inputs: StepIOs
    results: StepIOs
    details: StepDetails
    resources: ExtResources
    _ResultsModel: StepIOs = pydantic.PrivateAttr()

    def __init__(
        self,
        uid: str,
        args: StepArgs,
        resources: ExtResources,
        inputs: Union[StepIOs, StepIOsListBase],
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

        results = self.__fields__["results"].outer_type_()
        results.input_mode(mode=False)
        details = self.__fields__["details"].outer_type_()

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
            self.errors.errors["exception"] = str(e)
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

    def dump(self, full=True, exclude_results: bool = False) -> dict[str, Any]:
        """Dump step data into json compatible complex dictionary."""
        ret = self.dict(exclude={"inputs", "results"})
        ret["type"] = self.NAME
        ret["inputs"] = {}
        ret["inputs_standalone"] = {}
        ret["results"] = json.loads(self.results.json(exclude={"step", "__orig_class__"})) if not exclude_results else None
        ret["errors"] = self.errors.dict()
        ret["resources"] = self.resources.dump(full=full)
        for f, ftype in self.inputs.__fields__.items():
            field = getattr(self.inputs, f)
            if issubclass(ftype.type_, StepIO) and field._step:
                ret["inputs"][f] = (
                    getattr(self.inputs, f)._step.fullname,
                    getattr(self.inputs, f)._self_name,
                )
            elif issubclass(ftype.type_, StepIOsListBaseG) and field.data.step:
                ret["inputs"][f] = getattr(self.inputs, f).data.step.fullname
            else:
                ret["inputs_standalone"][f] = getattr(self.inputs, f).dict(exclude={"step"})
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

        self.args = self.__fields__["args"].type_.parse_obj(loaded_args)
        self.errors = step_dump["errors"]
        self.stats = step_dump["stats"]
        self.results.set_step(self)
        self.resources.load(step_dump["resources"])  # , secrets=_secrets)

    @classmethod
    def load_cls(
        cls,
        step_dump,
        inputs_map: Dict[str, "Step"],
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
        for f, ftype in cls.__fields__["args"].type_.__fields__.items():
            if ftype.type_ == Secret:
                try:
                    loaded_args[f] = Secret(_secrets["%s:%s" % (cls.NAME, step_dump["uid"])][f])
                except KeyError as e:
                    raise MissingSecretError(f) from e
            else:
                loaded_args[f] = step_dump["args"][f]

        args = cls.__fields__["args"].type_.parse_obj(loaded_args)

        loaded_inputs = {}
        for iname, step_and_field in step_dump["inputs"].items():
            res_step, res_field = step_and_field
            field = getattr(inputs_map[res_step].results, res_field)
            loaded_inputs[iname] = field

        for iname in step_dump["inputs_standalone"]:
            itype = cls.__fields__["inputs"].type_.__fields__[iname]
            loaded_result = itype.type_()
            for rfield in itype.type_.__fields__:
                if rfield == "step":
                    continue
                setattr(loaded_result, rfield, step_dump["inputs_standalone"][iname][rfield])
            loaded_inputs[iname] = loaded_result

        inputs = cls.__fields__["inputs"].type_.parse_obj(loaded_inputs)
        if not resources:
            _resources = cls.__fields__["resources"].type_.load_cls(
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
        ret.results.set_step(ret)

        return ret


Step.update_forward_refs()


class TractorDumpDict(TypedDict):
    steps: List[Dict[str, Any]]


class TractorValidateResult(TypedDict):
    missing_inputs: List[Tuple[str, str, str]]
    valid: bool


class NamedTractorMeta(pydantic.main.ModelMetaclass):
    def __new__(
        mcs,
        name,
        bases,
        namespace,
        nt_steps: List[Tuple[str, Union[Step, NamedTractor]]] = [],
        **kwargs,
    ):

        nt_step_map = {}
        nt_step_args = {}
        nt_step_results = {}
        nt_step_inputs = {}
        nt_step_details = {}
        nt_step_resources = {}
        steps_inputs_model = {}
        steps_args_model = {}
        steps_resources_model = {}

        for (nt_step_name, nt_step) in nt_steps:
            nt_step_map[nt_step_name] = nt_step
            results_type = nt_step.__fields__["results"].outer_type_
            args_type = nt_step.__fields__["args"].outer_type_
            resources_type = nt_step.__fields__["resources"].outer_type_
            inputs_type = nt_step.__fields__["inputs"].outer_type_
            details_type = nt_step.__fields__["details"].outer_type_

            nt_step_args[nt_step_name] = {k: v for k, v in args_type.__fields__.items()}
            nt_step_results[nt_step_name] = {
                k: v.outer_type_ for k, v in results_type.__fields__.items()
            }
            nt_step_resources[nt_step_name] = {
                k: v.outer_type_ for k, v in resources_type.__fields__.items()
            }
            nt_step_inputs[nt_step_name] = {
                k: v.outer_type_ for k, v in inputs_type.__fields__.items()
            }
            nt_step_details[nt_step_name] = details_type

            steps_inputs_model[nt_step_name] = inputs_type
            steps_args_model[nt_step_name] = args_type
            steps_resources_model[nt_step_name] = resources_type

        annotations = namespace.get("__annotations__", {})
        results_type = annotations.get("results", NoResults)
        inputs_type = annotations.get("inputs", NoInputs)
        args_type = annotations.get("args", NoArgs)
        resources_type = annotations.get("resources", NoResources)
        details_type = annotations.get("details", NoDetails)

        namespace.setdefault("__annotations__", {})
        namespace["InputsModel"] = inputs_type
        namespace["ResultsModel"] = results_type
        namespace["ArgsModel"] = args_type
        namespace["DetailsModel"] = details_type
        namespace["ResourcesModel"] = resources_type

        namespace["_steps_inputs"] = steps_inputs_model
        namespace["_steps_args"] = steps_args_model
        namespace["_steps_resources"] = steps_resources_model
        namespace["_nt_steps"] = nt_steps

        if (
            not check_type(results_type, StepIOs)
            and not check_type(results_type, StepIOsListBase)
            and not check_type_generics(results_type, StepIOsListBaseG)
        ):
            raise TypeError(
                "results_type has to be subclass of StepIOs or StepIOsListBase (but is %s)"
                % results_type
            )

        if (
            not check_type(inputs_type, StepIOs)
            and not check_type(inputs_type, StepIOsListBase)
            and not check_type_generics(inputs_type, StepIOsListBaseG)
        ):
            raise TypeError(
                "inputs_type has to be subclass of StepIOs or StepIOsListBase, is %s"
                % type(results_type)
            )

        if not check_type(args_type, StepArgs):
            raise TypeError("args_type has to be subclass of StepArgs")

        if not check_type(resources_type, ExtResources):
            raise TypeError("args_type has to be subclass of ExtResources")

        if not check_type(details_type, StepDetails):
            raise TypeError("args_type has to be subclass of StepDetails")

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

        for step_name, inputs_map in namespace["INPUTS_MAP"].items():
            if not isinstance(step_name, str):
                raise TypeError("INPUTS_MAP %s has to be string" % step_name)
            if not isinstance(inputs_map, dict):
                raise TypeError("INPUTS_MAP value for step %s has to be dictionary" % step_name)
            if step_name not in nt_step_inputs:
                raise TypeError("Tractor doesn't have step %s mentioned in INPUTS_MAP" % step_name)
            for input_name, output_map in inputs_map.items():
                if input_name not in nt_step_inputs[step_name]:
                    raise TypeError(
                        "Step %s doesn't have input '%s' (%s)"
                        % (step_name, input_name, nt_step_inputs[step_name])
                    )
                if not isinstance(output_map, NTInput) and not isinstance(output_map, tuple):
                    raise TypeError(
                        "INPUTS_MAP->%s->%s value has to be tuple" % (step_name, input_name)
                    )
                if isinstance(output_map, NTInput):
                    if output_map.name not in inputs_type.__fields__:
                        raise TypeError(
                            "output step INPUTS_MAP->%s->%s = NTInput(%s), %s is not in %s "
                            % (
                                step_name,
                                input_name,
                                output_map.name,
                                output_map.name,
                                inputs_type.__fields__,
                            )
                        )
                else:
                    output_step, output_key = output_map
                    if output_step not in nt_step_results:
                        raise TypeError(
                            "INPUTS_MAP->%s->%s = (%s, ...) output step not found"
                            % (step_name, input_name, output_step)
                        )
                    if output_key not in nt_step_results[output_step]:
                        raise TypeError(
                            "output '%s' is not in step '%s' (%s)"
                            % (output_key, output_step, nt_step_results[output_step].keys())
                        )
                    if output_step not in nt_step_results:
                        raise TypeError(
                            "output step INPUTS_MAP->%s->%s = (%s, ...) is not in tractor steps"
                            % (step_name, input_name, output_step)
                        )

                    step_results = nt_step_map[output_step].__fields__["results"].outer_type_
                    step_inputs = nt_step_map[step_name].__fields__['inputs'].outer_type_
                    if not check_type(
                        step_inputs.__fields__[input_name].outer_type_,
                        step_results.__fields__[output_key].outer_type_
                    ):
                        # if (
                        #     nt_step_inputs[step_name][input_name]
                        #     != nt_step_results[output_step][output_key]
                        # ):
                        raise TypeError(
                            f"output step INPUTS_MAP->{step_name}->{input_name} = ({output_step}, {output_key}) result type is"
                            "%s but %s is required"
                            % (
                                nt_step_results[output_step][output_key],
                                nt_step_inputs[step_name][input_name],
                            )
                        )

        for step_name, args_map in namespace["ARGS_MAP"].items():
            if not isinstance(step_name, str):
                raise TypeError("ARGS_MAP %s has to be string" % step_name)
            if not isinstance(args_map, dict):
                raise TypeError("ARGS_MAP value for step %s has to be dictionary" % step_name)
            if step_name not in nt_step_args:
                raise TypeError("Tractor doesn't have step %s mentioned in ARGS_MAP" % step_name)
            for sarg, targ in args_map.items():
                if sarg not in nt_step_args[step_name]:
                    raise TypeError("Step %s doesn't have argument %s" % (step_name, sarg))
                if targ not in annotations.get("args", NoArgs).__fields__:
                    raise TypeError(
                        "Tractor doesn't have argument %s (%s)"
                        % (targ, list(annotations["args"].__fields__.keys()))
                    )

                model, errors = (
                    annotations["args"]
                    .__fields__[targ]
                    .validate(nt_step_args[step_name][sarg], {}, loc="")
                )
                if (
                    nt_step_args[step_name][sarg].type_
                    != annotations["args"].__fields__[targ].type_
                ):
                    raise TypeError(
                        "Step argument %s (%s) is not subtype of tractor argument %s"
                        % (targ, step_name, sarg)
                    )

        # need to initialize results here to convert parameters to types
        # e.g. -> Class[~T] to Class[ActuallType]
        results = annotations["results"]()

        for result_io, results_map in namespace["RESULTS_MAP"].items():
            if result_io not in annotations["results"].__fields__:
                raise TypeError("Tractor doesn't have result '%s'" % (result_io))
            if not isinstance(results_map, tuple):
                raise TypeError("RESULTS_MAP value has to be tuple")
            (step_name, step_result) = results_map
            if step_name not in nt_step_results:
                raise TypeError("Tractor doesn't have step %s mentioned in RESULTS_MAP" % step_name)
            if step_result not in nt_step_results[step_name]:
                raise TypeError(
                    "Step %s doesn't have result %s (%s)"
                    % (step_name, step_result, nt_step_results[step_name])
                )

            # here check namespace['results']() because annotations['results'] is not valid field
            step_results = nt_step_map[step_name].__fields__["results"].outer_type_

            if not check_type(
                step_results.__fields__[step_result].outer_type_,
                results.__fields__[result_io].outer_type_,
            ):
                raise TypeError(
                    "Step result '%s->%s' (%s) is not subtype of tractor result '%s' (%s)"
                    % (
                        step_name,
                        step_result,
                        nt_step_map[step_name].__fields__["results"].outer_type_,
                        result_io,
                        results.__fields__[result_io].outer_type_,
                    )
                )

        for step_name, resources_map in namespace["RESOURCES_MAP"].items():
            if not isinstance(step_name, str):
                raise TypeError("RESOURCES_MAP %s has to be string" % step_name)
            if not isinstance(resources_map, dict):
                raise TypeError("RESOURCES_MAP value for step %s has to be dictionary" % step_name)
            if step_name not in nt_step_resources:
                raise TypeError(
                    "Tractor doesn't have step %s mentioned in RESOURCES_MAP" % step_name
                )
            for sresource, tresource in resources_map.items():
                if sresource not in nt_step_resources[step_name]:
                    raise TypeError("Step %s doesn't have resource %s" % (step_name, sresource))
                if tresource not in annotations["resources"].__fields__:
                    raise TypeError(
                        "Tractor doesn't have resource %s (%s)"
                        % (tresource, list(annotations["resources"].__fields__.keys()))
                    )
                if not issubclass(
                    nt_step_resources[step_name][sresource],
                    annotations["resources"].__fields__[tresource].type_,
                ):
                    raise TypeError(
                        "Step resource %s (%s) is not subtype of tractor resource %s"
                        % (tresource, step_name, sresource)
                    )
            step_resources = nt_step_resources[step_name]
            for step_res_field in step_resources:
                if step_res_field not in resources_map:
                    raise TypeError(
                        "Step resource %s is not mapped in RESOURCES_MAP[%s]"
                        % (tresource, step_name)
                    )

        ret = super().__new__(mcs, name, bases, namespace)
        return ret


class NTInput(pydantic.BaseModel):
    name: str


class NamedTractor(
    NoCopyModel, metaclass=NamedTractorMeta, nt_steps=[], nt_inputs=StepIOsListBaseG[StepIOs]
):
    """Class which runs sequence of steps."""

    _step_map: Dict[str, Union[Step, NamedTractor]] = {}
    steps: List[Union[Step, Step, NamedTractor]] = []
    _current_step: Union[Step, Step, NamedTractor, STMD, None] = None  # TODO: fix
    _step_name_by_uid: Dict[str, Union[Step, NamedTractor]]
    uid: str
    state: StepState = StepState.READY

    TYPE: ClassVar[str] = "Tractor"
    INPUTS_MAP: ClassVar[Dict[str, Dict[str, str]]] = {}
    ARGS_MAP: ClassVar[Dict[str, Dict[str, str]]] = {}
    DETAILS_MAP: ClassVar[Dict[str, Dict[str, str]]] = {}
    RESULTS_MAP: ClassVar[Dict[str, Dict[str, str]]] = {}
    RESOURCES_MAP: ClassVar[Dict[str, Dict[str, str]]] = {}
    NAME: ClassVar[str] = "NamedTractor"

    args: StepArgs
    inputs: StepIOs
    results: StepIOs
    details: StepDetails
    resources: ExtResources

    def __init__(self, *args, **kwargs):
        results_model_data = {}
        details_model_data = {}

        results = self.ResultsModel(**results_model_data)
        results.input_mode(mode=False)
        # results.input_mode(mode=False)
        kwargs["results"] = results
        kwargs["details"] = self.DetailsModel()
        super().__init__(**kwargs)
        self._step_name_by_uid = {}

        i = 0
        for step_name, step_type in self._nt_steps:
            i += 1
            if step_type.__fields__["inputs"].type_ not in (StepIOs, NoInputs):
                step_inputs = self.INPUTS_MAP[step_name]
            else:
                step_inputs = {}
            inputs_model_data = {}
            args_model_data = {}
            resources_model_data = {}

            if step_type.__fields__["args"].type_ not in (NoArgs, StepArgs):
                for sarg, targ in self.ARGS_MAP[step_name].items():
                    args_model_data[sarg] = getattr(self.args, targ)

            if step_type.__fields__["resources"].type_ not in (NoResources, ExtResources):
                if step_type.__fields__["resources"].type_ not in (ExtResources, NoResources):
                    for sresource, tresource in self.RESOURCES_MAP[step_name].items():
                        resources_model_data[sresource] = getattr(self.resources, tresource)

            for input_, output_from in step_inputs.items():
                if isinstance(output_from, tuple):
                    output_step, output_key = output_from
                    if output_step not in self._step_map:
                        raise ValueError(
                            "(%s) Required step '%s' is not in tractor steps"
                            % (self.NAME, output_step)
                        )
                    inputs_model_data[input_] = getattr(
                        self._step_map[output_step].results, output_key
                    )
                else:  # NTInput
                    inputs_model_data[input_] = getattr(self.inputs, output_from.name)

            
            inputs_model = self._steps_inputs[step_name](**inputs_model_data)

            args_model = self._steps_args[step_name](**args_model_data)
            resources_model = self._steps_resources[step_name](**resources_model_data)

            step = step_type(
                uid="%s:%d.%s" % (self.uid, i, step_name),
                args=args_model,
                resources=resources_model,
                inputs=inputs_model,
            )
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
            elif isinstance(step, STMD):
                for k in step.resources.__fields__:
                    res = getattr(step.resources, k)
                    resources_dump[res.fullname] = res.dump()
            else:  # named tractor
                _resources = step._dump_resources(full=full)
                for res_fullname, res in _resources.items():
                    resources_dump[res_fullname] = res
        return resources_dump

    def dump(self, full: bool, exclude_results: bool = False) -> TractorDumpDict:
        """Dump stepper state and shared_results to json compatible dict."""
        steps: List[Dict[str, Any]] = []
        out: TractorDumpDict = {
            "state": self.state.value,
            "steps": steps,
            "resources": {},
            "uid": self.uid,
            "results": json.loads(self.results.json()) if not exclude_results else None,
            "current_step": self._current_step.fullname if self._current_step else None,
        }
        resources_dump: Dict[str, Any] = {}
        for step in self.steps:
            if isinstance(step, Step):
                steps.append({"type": "step", "data": step.dump(full=False)})
            elif isinstance(step, STMD):
                steps.append({"type": "stmd", "data": step.dump(full=False)})
            else:
                steps.append({"type": "tractor", "data": step.dump(full=False)})
        out["resources"] = self._dump_resources(full=full)
        return out

    @classmethod
    def load_cls(
        cls,
        dump_obj: TractorDumpDict,
        step_map: Dict[str, Type[Step]],
        resources_map: Dict[str, Type[ExtResource]],
        secrets: Dict[str, Dict[str, str]],
    ) -> None:
        ret = cls(dump_obj["uid"])
        ret.load(dump_obj, step_map, resources_map, secrets)
        return ret

    def load(
        self,
        dump_obj: TractorDumpDict,
        step_map: Dict[str, Type[Step]],
        resources_map: Dict[str, Type[ExtResource]],
        secrets: Dict[str, Dict[str, str]],
    ) -> None:
        """Load and initialize stepper from data produced by dump method."""
        loaded_steps = {}
        loaded_resources = {}
        self.steps = []
        for fullname, resource_dump in dump_obj["resources"].items():
            resource_dump_copy = resource_dump.copy()
            loaded_resources[fullname] = resources_map[resource_dump["type"]].load_cls(
                resource_dump_copy,
                secrets.get("%s:%s" % (resource_dump["type"], resource_dump["uid"]), {}),
            )

        for step_obj in dump_obj["steps"]:
            if step_obj["type"] == "stepng":
                step_resources = {}
                for resource, resource_fullname in step_obj["data"]["resources"].items():
                    if resource == "type":
                        continue
                    step_resources[resource] = loaded_resources[resource_fullname]
                resources_type = step_map[step_obj["data"]["type"]].__fields__["resources"].type_
                resources = resources_type(**step_resources)
                step = step_map[step_obj["data"]["type"]].load_cls(
                    step_obj["data"],
                    loaded_steps,
                    resources=resources,
                    secrets=secrets,
                )  # .get("%s:%s" % (step_obj['type'], step_obj['uid']), {}))
                loaded_steps[step.fullname] = step
                self.steps.append(step)
            elif step_obj["type"] == "tractor":
                tractor = self.load_cls(
                    step_obj["data"],
                    step_map,
                    resources_map,
                    secrets.get(step_obj["data"]["uid"], {}),
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
    def __new__(mcs, name, bases, namespace, tractor_type: Type[NamedTractor], **kwargs):

        results_model = {}
        inputs_model = {}
        inputs_as_results_model = {}
        args_model = {}
        details_model = {}
        stmd_io_res_model = {}
        stmd_io_res_data = {}
        stmd_io_ins_model = {}
        stmd_io_ins_data = {}

        tArgsModel = tractor_type.__fields__["args"].outer_type_
        tInputsModel = tractor_type.__fields__["inputs"].outer_type_
        tResultsModel = tractor_type.__fields__["results"].outer_type_
        tDetailsModel = tractor_type.__fields__["results"].outer_type_
        tResourcesModel = tractor_type.__fields__["resources"].outer_type_

        args_model["__base__"] = tractor_type.__fields__["args"].type_

        details_model["__base__"] = StepDetails

        args_model["tractors"] = (int, ...)
        args_model["use_processes"] = (bool, False)
        ArgsModel = pydantic.create_model("%s_args" % (name,), **args_model)
        setattr(pydantic.main, "%s_args" % (name,), ArgsModel)
        ArgsModel.update_forward_refs(**globals())

        details_model["data"] = (List[tDetailsModel], [])
        DetailsModel = pydantic.create_model("%s_details" % (name,), **details_model)
        setattr(pydantic.main, "%s_details" % (name,), DetailsModel)
        DetailsModel.update_forward_refs(**globals())

        namespace.setdefault("__annotations__", {})
        namespace["__annotations__"]["results"] = StepIOsListBaseG[tResultsModel]
        namespace["__annotations__"]["inputs"] = StepIOsListBaseG[tInputsModel]
        namespace["__annotations__"]["args"] = ArgsModel
        namespace["__annotations__"]["resources"] = tResourcesModel
        namespace["__annotations__"]["details"] = DetailsModel
        namespace["__annotations__"]["tractions"] = List[Union[Step, NamedTractor, STMD]]

        namespace["ResultsModel"] = StepIOsListBaseG[tResultsModel]
        namespace["InputsModel"] = StepIOsListBaseG[tInputsModel]
        namespace["TractorType"] = tractor_type
        namespace["ArgsModel"] = ArgsModel
        namespace["ResourcesModel"] = tResourcesModel
        # namespace['MultiDataModel'] = MultiDataModel
        namespace["MultiDataInput"] = StepIOsListBaseG[tInputsModel]
        namespace["DetailsModel"] = DetailsModel

        namespace["results"] = StepIOsListBaseG[tResultsModel]()
        ret = super().__new__(mcs, name, bases, namespace)
        return ret


class STMD(NoCopyModel, metaclass=STMDMeta, tractor_type=NamedTractor):
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
            tractions=[],
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

        targs = self.args.dict()
        targs.pop("tractors")


        if self.args.use_processes:
            executor_class = ProcessPoolExecutor
        else:
            executor_class = ThreadPoolExecutor
        with executor_class(max_workers=self.args.tractors) as executor:
            ft_results = {}
            nts = []
            for i in range(0, len(self.inputs.data)):

                nt = self.TractorType(
                    uid="%s:%d" % (self.uid, i),
                    args=self.TractorType.__fields__["args"].type_(**targs),
                    resources=self.resources,
                    inputs=self.inputs.data[i],
                )
                self.details.data.append(self.TractorType.__fields__["details"].type_())
                self.details.data[i] = nt.details
                self.tractions.append(nt)
                res = executor.submit(nt.run, on_update=on_update, on_error=on_error)
                ft_results[res] = (nt, i)
                self.results.data.append(self.TractorType.__fields__["results"].type_())
            _on_update(self)
            for ft in as_completed(ft_results):
                (_, i) = ft_results[ft]
                nt = ft.result()
                self.tractions[i] = nt
                self.results.data[i] = nt.results
                self.details.data[i] = nt.details
                _on_update(self)

        self.state = StepState.FINISHED

    def dump(self, full: bool, exclude_results: bool = False) -> TractorDumpDict:
        """Dump stepper state and shared_results to json compatible dict."""
        steps: List[Dict[str, Any]] = []
        out: TractorDumpDict = {
            "state": self.state.value,
            "tractions": [],
            "resources": {},
            "uid": self.uid,
            "inputs": {},
            "inputs_standalone": {},
            "results": json.loads(self.results.json()) if not exclude_results else None,
        }
        for f, ftype in self.inputs.__fields__.items():
            field = getattr(self.inputs, f)
            if issubclass(ftype.type_, StepIO) and field._step:
                out["inputs"][f] = (
                    getattr(self.inputs, f)._step.fullname,
                    getattr(self.inputs, f)._self_name,
                )
            elif issubclass(ftype.type_, StepIOsListBaseG) and field.data.step:
                out["inputs"][f] = getattr(self.inputs, f).data.step.fullname
            else:
                out["inputs_standalone"][f] = getattr(self.inputs, f).dict(exclude={"step"})
        resources_dump: Dict[str, Any] = {}
        for tr in self.tractions:
            out["tractions"].append(tr.dump(full=full, exclude_results=True))
        out["resources"] = self.resources.dump(full=full)
        return out


STMD.update_forward_refs()
