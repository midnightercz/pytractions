import abc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import datetime
import hashlib
import inspect
import json
from types import prepare_class, resolve_bases
import enum
import sys
import uuid

from typing import (
    Dict,
    List,
    Any,
    ClassVar,
    Union,
    get_origin,
    ForwardRef,
    Optional,
    TypeVar,
    Generic,
    GenericAlias,
    Tuple,
    Type,
    Callable,
    get_args
)
from typing_extensions import Self

import dataclasses

from .exc import TractionFailedError
from .utils import ANY
from .types import TypeNode, JSON_COMPATIBLE
from .abase import ABase

_Base = ForwardRef("Base")
TList = ForwardRef("TList")
TDict = ForwardRef("TDict")
_Traction = ForwardRef("Traction")


class NoAnnotationError(TypeError):
    pass


class JSONIncompatibleError(TypeError):
    pass


def find_attr(objects, attr_name):
    for o in objects:
        if hasattr(o, attr_name):
            return getattr(o, attr_name)


def on_update_empty(T):
    pass


def evaluate_forward_ref(ref, frame):
    caller_globals, caller_locals = frame.f_globals, frame.f_locals
    recursive_guard = set()
    return ref._evaluate(caller_globals, caller_locals, recursive_guard)


@dataclasses.dataclass
class BaseConfig:
    validate_set_attr: bool = True
    allow_extra: bool = False


@dataclasses.dataclass
class DefaultOut:
    type_: JSON_COMPATIBLE
    params: List[JSON_COMPATIBLE]

    def copy(self, generic_cache):
        return DefaultOut(
            type_=TypeNode.from_type(self.type_).to_type(types_cache=generic_cache),
            params=(TypeNode.from_type(self.params[0]).to_type(types_cache=generic_cache),)
        )

    def __call__(self):
        # handling Optional
        if get_origin(self.type_) == Union and len(get_args(self.type_)) == 2 and \
           get_args(self.type_)[-1] == type(None):
            ret = Out[self.params]()
        else:
            ret = Out[self.params]()
            ret.data = self.type_()

        return ret

    def replace_params(self, params_map, cache):
        tn = TypeNode.from_type(self.type_)
        tn.replace_params(params_map)
        new_type = tn.to_type(types_cache=cache)
        self.type_ = new_type

        tn = TypeNode.from_type(self.params[0])
        tn.replace_params(params_map)
        new_type = tn.to_type(types_cache=cache)
        self.params = (new_type,)


def _hash(obj):
    return hashlib.sha256(json.dumps(obj.to_json(), sort_keys=True)).hexdigest()


class BaseMeta(type):
    def __repr__(cls):
        qname = cls.__orig_qualname__ if hasattr(cls, "__orig_qualname__") else cls.__qualname__ 
        if cls._params:
            params = []
            for p in cls._params:
                if get_origin(p) is Union:
                    uparams = ",".join([repr(up) if up is not type(None) else 'NoneType' for up in sorted(get_args(p), key=lambda x: repr(x))])
                    params.append(f"Union[{uparams}]")
                elif isinstance(p, cls):
                    params.append(repr(p))
                else:
                    params.append(p.__qualname__ if hasattr(p, "__qualname__") else p.__name__)
            params_str = ",".join(params)
            return f"{qname}[{params_str}]"
        else:
            return f"{qname}"

    @classmethod
    def _before_new(cls, name, attrs, bases):
        pass

    def __new__(cls, name, bases, attrs):
        if '_config' in attrs:
            assert TypeNode.from_type(type(attrs["_config"])) == TypeNode(BaseConfig)
            config = attrs['_config']
        else:
            # if not, provide default config
            config = BaseConfig()
            attrs["_config"] = config

        if config.validate_set_attr:
            # if setter validation is on, use _validate_setattr_
            # or find it in class bases
            if '_validate_setattr_' in attrs:
                _setattr = attrs['_validate_setattr_']
            else:
                _setattr = find_attr(bases, '_validate_setattr_')
            attrs['__setattr__'] = _setattr
        else:
            if '_no_validate_setattr_' in attrs:
                _setattr = attrs['_no_validate_setattr_']
            else:
                _setattr = find_attr(bases, '_no_validate_setattr_')
            attrs['__setattr__'] = attrs['_no_validate_setattr_']

        annotations = attrs.get('__annotations__', {})
        for attr, attrv in attrs.items():
            # skip annotation check for methods and functions
            if inspect.ismethod(attrv) or \
                    inspect.isfunction(attrv) or \
                    isinstance(attrv, classmethod) or \
                    isinstance(attrv, property) or \
                    isinstance(attrv, staticmethod):
                continue
            # attr starting with _ is considered to be private, there no checks
            # are applied
            if attr.startswith("_"):
                continue
            # other attributes has to be annotated
            if attr not in annotations:
                raise NoAnnotationError(f"{attr} has to be annotated")
        defaults = {}

        # check if all attrs are in supported types
        for attr, type_ in annotations.items():
            # skip private attributes
            if attr.startswith("_"):
                continue
            # Check if type of attribute is json-compatible
            if not TypeNode.from_type(annotations[attr]).json_compatible():
                raise JSONIncompatibleError(f"Attribute {attr} is not json compatible {annotations[attr]}")
            if attr in attrs:
                if type(attrs[attr]) == dataclasses.Field:
                    default = dataclasses.MISSING
                    if attrs[attr].default is not dataclasses.MISSING:
                        default = attrs[attr].default
                    if attrs[attr].default_factory is not dataclasses.MISSING:
                        default = attrs[attr].default_factory
                elif type(attrs[attr]) in (str, int, None, float):
                    default = attrs[attr]
                else:
                    default = type(attrs[attr])

                if isinstance(default, DefaultOut):
                    default = type(default())

                defaults[attr] = default
                #TODO: fix
                #if default != dataclasses._MISSING_TYPE\
                #   and not isinstance(default, dataclasses._MISSING_TYPE)\
                #   and TypeNode.from_type(type_) != TypeNode.from_type(default):
                #    raise TypeError(f"Annotation for {attr} is {type_} but default is {default}")

        # record fields to private attribute
        attrs["_attrs"] = attrs
        fields = {}
        all_annotations = {}
        for base in bases:
            for f, ft in getattr(base, "_fields", {}).items():
                fields[f] = ft

        for base in bases:
            for f, ft in fields.items():
                if hasattr(base, "__dataclass_fields__") and f in base.__dataclass_fields__:
                    if base.__dataclass_fields__[f].default is not dataclasses.MISSING:
                        defaults[f] = dataclasses.field(default=base.__dataclass_fields__[f].default)
                    elif base.__dataclass_fields__[f].default_factory is not dataclasses.MISSING:
                        defaults[f] = dataclasses.field(default_factory=base.__dataclass_fields__[f].default_factory)

            for f, ft in getattr(base, "__annotations__", {}).items():
                if f in fields:
                    all_annotations[f] = ft

        fields.update({k: v for k, v in attrs.get('__annotations__', {}).items() if not k.startswith("_")})
        all_annotations.update(annotations)
        attrs["__annotations__"] = all_annotations

        for default, defval in defaults.items():
            if default not in attrs:
                attrs[default] = defval

        attrs['_fields'] = fields
        attrs['__hash__'] = _hash

        cls._before_new(name, attrs, bases)

        ret = super().__new__(cls, name, bases, attrs)
        ret = dataclasses.dataclass(ret, kw_only=True)
        return ret


class Base(ABase, metaclass=BaseMeta):
    _CUSTOM_TYPE_TO_JSON: bool = False

    # dataclasses configuration class
    _config: ClassVar[BaseConfig] = BaseConfig()
    # mapping of class fields
    _fields: ClassVar[Dict[str, Any]] = {}
    # mapping used as lookup dict when creating generic subclasses
    _generic_cache: ClassVar[Dict[str, Type[Any]]] = {}
    # use to store actual parameters when creating generic subclass
    _params: ClassVar[List[Any]] = []
    # used to store original class when creating generic subclass
    _orig_cls: Optional[Type[Any]] = None

    def __post_init__(self):
        pass

    def _no_validate_setattr_(self, name: str, value: Any) -> None:
        return super().__setattr__(name, value)

    def _validate_setattr_(self, name: str, value: Any) -> None:
        if not name.startswith("_"):  # do not check for private attrs
            properties = dict(inspect.getmembers(self.__class__, lambda o: isinstance(o, property)))

            if name not in self._fields and not self._config.allow_extra and name not in properties:
                raise AttributeError(f"{self.__class__} doesn't have attribute {name}")

            if name not in properties:
                vtype = value.__orig_class__ if hasattr(value, "__orig_class__") else value.__class__
                tt1 = TypeNode.from_type(vtype)
                tt2 = TypeNode.from_type(self._fields[name])
                if tt1 != tt2:
                    raise TypeError(f"Cannot set attribute {self.__class__}.{name} to {value}({vtype}), expected {self._fields[name]}")
            else:
                getattr(self.__class__, name).setter(value)
                return

        return super().__setattr__(name, value)

    @staticmethod
    def _make_qualname(cls, params):
        if params:
            stack = [(cls.__qualname__, params)]
        elif cls._params:
            stack = [(cls.__qualname__, cls._params)]
        else:
            stack = [(cls.__qualname__, [])]

        order = []
        while stack:
            current_qname, current_params = stack.pop(0)
            order.append(current_qname)
            if current_params:
                stack.insert(0, ("]", []))
                order.append("[")
            for p in current_params:
                if get_origin(p) is Optional:
                    stack.insert(0, ("Optional", get_args(p)))
                if get_origin(p) is Union:
                    stack.insert(0, ("Union", sorted(get_args(p), key=lambda x: repr(x))))
                elif hasattr(p, "_params"):
                    if hasattr(p, "__orig_qualname__"):
                        stack.insert(0, (p.__orig_qualname__, p._params))
                    else:
                        stack.insert(0, (p.__qualname__, p._params))
                elif hasattr(p, "__orig_qualname__"):
                    order.append(p.__qualname__)
                    order.append(",")
                elif hasattr(p, "__qualname__"):
                    order.append(p.__qualname__)
                    order.append(",")
                elif get_origin(p) is ForwardRef:
                    order.append(p.__forward_arg__)
                    order.append(",")
                elif isinstance(p, TypeVar):
                    # for typevar we need to add id to qualname, otherwise typevar replacement won't work
                    # as class_getitem could return cached version with typevar with different id
                    order.append(f"{p.__name__}[{id(p)}]")
                    order.append(",")
        return "".join(order)

    def __class_getitem__(cls, param, params_map={}):
        _params = param if isinstance(param, tuple) else (param,)

        # param ids for caching as TypeVars are class instances
        # therefore has to compare with id() to get good param replacement
        _param_ids = tuple([id(p) for p in param]) if isinstance(param, tuple) else (id(param),)
        qname = cls._make_qualname(cls, _params)
        # if there's already existing class, return it instead
        if qname in sys.modules[cls.__module__].__dict__:
            ret = sys.modules[cls.__module__].__dict__[qname]
            return ret

        bases = [x for x in resolve_bases([cls] + list(cls.__bases__)) if x is not Generic]
        attrs = {k: v for k, v in cls._attrs.items() if k not in ("_attrs", "_fields")}

        meta, ns, kwds = prepare_class(f"{cls.__name__}[{param}]", bases, attrs)

        _params_map = params_map.copy()
        _params_map.update(dict(zip(cls.__parameters__, _params)))

        # Fields needs to be copied to new subclass, otherwise
        # it's stays shared with base class
        for attr, type_ in cls._fields.items():
            tn = TypeNode.from_type(type_)
            # field params needs to be replaced as field can also reffer to TypeVar
            tn.replace_params(_params_map)
            new_type = tn.to_type(types_cache=cls._generic_cache, params_map=_params_map)

            if not tn.json_compatible():
                raise JSONIncompatibleError(f"Attribute  {attr}: {new_type} is not json compatible")
            if attr not in kwds:
                kwds[attr] = new_type
            kwds["__annotations__"][attr] = new_type

        for k, kf in kwds.items():
            if not isinstance(kf, dataclasses.Field):
                continue
            new_kf = dataclasses.Field(
                default=kf.default,
                default_factory=kf.default_factory,
                init=kf.init,
                repr=kf.repr,
                hash=kf.hash,
                compare=kf.compare,
                metadata=kf.metadata,
                kw_only=kf.kw_only)

            if hasattr(new_kf.default_factory, "replace_params"):
                new_default_factory = new_kf.default_factory.copy(generic_cache=cls._generic_cache)
                new_default_factory.replace_params(_params_map, cls._generic_cache)
                new_kf.default_factory = new_default_factory
            else:
                if new_kf.default_factory is not dataclasses.MISSING:
                    new_default_factory = TypeNode.from_type(new_kf.default_factory)
                    new_default_factory.replace_params(_params_map)
                    new_default_factory = new_default_factory.to_type(types_cache=cls._generic_cache)
                    new_kf.default_factory = new_default_factory

            kwds[k] = new_kf

        kwds["_params"] = _params
        if cls._orig_cls:
            kwds["_orig_cls"] = cls._orig_cls
        else:
            kwds["_orig_cls"] = cls

        kwds['__orig_qualname__'] = kwds.get('__orig_qualname__', kwds['__qualname__'])
        kwds['__qualname__'] = cls._make_qualname(cls, _params)

        ret = meta(kwds['__qualname__'], tuple(bases), kwds)

        sys.modules[ret.__module__].__dict__[ret.__qualname__] = ret

        cls._generic_cache[f"{id(cls)}[{_param_ids}]"] = ret
        return ret

    def to_json(self) -> Dict[str, Any]:
        """Return json representation of Base object. Function is written to dump a base object
        to json from which can be considered as serialized object, therefore it's possible to
        use this representation to load the very same object.
        However compare to python serializer, performance of much slower.
        """
        pre_order: Dict[str, Any] = {}
        stack: List[Tuple[Base, Dict[str, Any], str]] = [(self, pre_order, "root")]
        while stack:
            current, current_parent, parent_key = stack.pop(0)
            if not isinstance(current, (int, str, bool, float, type(None), TList, TDict)):
                current_parent[parent_key] = {"$type": TypeNode.from_type(current.__class__).to_json(), "$data": {}}
                for f in current._fields:
                    stack.append((getattr(current, f), current_parent[parent_key]['$data'], f))
            elif isinstance(current, (TList, TDict)):
                current_parent[parent_key] = current.to_json()
            else:
                current_parent[parent_key] = current
        return pre_order['root']

    def content_to_json(self) -> Dict[str, Any]:
        """Similar to `Base.to_json` method, but doesn't include information of type of the object.
        Only it's content. This is exit only method. Output if this method cannot be used as 
        input for any 'load' method.
        """
        pre_order: Dict[str, Any] = {"root": {}}
        stack: List[Tuple[Base, Dict[str, Any], str]] = []
        for f in self._fields:
            stack.append((getattr(self, f), pre_order['root'], f))
        while stack:
            current, current_parent, parent_key = stack.pop(0)
            if isinstance(current, (TList, TDict, Base)):
                current_parent[parent_key] = current.content_to_json()
            else:
                current_parent[parent_key] = current
        return pre_order['root']

    @classmethod
    def type_to_json(cls) -> Dict[str, Any]:
        """Similar to `Base.to_json` method, but dumps information only of type of the object
        """
        pre_order: Dict[str, Any] = {}
        # stack is list of (current_cls_to_process, current_parent, current_key, current_default)
        stack: List[Tuple[Type[Base], Dict[str, Any], str, Optional[JSON_COMPATIBLE]]] = [(cls, pre_order, "root", None)]
        while stack:
            current, current_parent, parent_key, current_default = stack.pop(0)
            if hasattr(current, "_TYPE"):
                current_parent['_TYPE'] = current._TYPE
            if hasattr(current, "_CUSTOM_TYPE_TO_JSON") and current._CUSTOM_TYPE_TO_JSON and current != cls:
                current_parent[parent_key] = current.type_to_json()
            elif hasattr(current, "_fields"):
                current_parent[parent_key] = {"$type": TypeNode.from_type(current).to_json()}
                for f, ftype in current._fields.items():
                    if type(current.__dataclass_fields__[f].default) in (str, int, float, None):
                        stack.append((ftype, current_parent[parent_key], f, current.__dataclass_fields__[f].default))
                    else:
                        stack.append((ftype, current_parent[parent_key], f, None))
            else:
                current_parent[parent_key] = {"$type": TypeNode.from_type(current).to_json(), "default": current_default}
        return pre_order['root']

    @classmethod
    def from_json(cls, json_data: JSON_COMPATIBLE) -> Self:
        """ Oposite to `Base.to_json` method. Method returns dumped instance of a Base class filled with
        data provided on the input.
        """
        stack: List[Tuple[Dict[str, JSON_COMPATIBLE], str, Dict[str, JSON_COMPATIBLE], Type[Optional[Self]], Dict[str, ANY]]] = []
        post_order = []
        root_args: Dict[str, Any] = {"root": None}
        stack.append((root_args, 'root', json_data, cls, {}))

        while stack:
            parent_args, parent_key, data, type_, type_args = stack.pop(0)
            if hasattr(type_, "__qualname__") and type_.__qualname__ in ("Optional", "Union"):
                if isinstance(data, dict):
                    data_type = data.get("$type", None)
                else:
                    data_type = TypeNode.from_type(data.__class__).to_json()
                for uarg in get_args(type_):
                    if TypeNode.from_type(uarg) == TypeNode.from_json(data_type):
                        stack.append((parent_args, parent_key, data, uarg, type_args))
                        break
                else:
                    stack.append((parent_args, parent_key, data, type(None), type_args))

            elif hasattr(type_, "__qualname__") and type_.__qualname__ == "Union":
                if isinstance(data, dict):
                    data_type = data.get("$type", None)
                else:
                    data_type = TypeNode.from_type(data.__class__).to_json()
                for uarg in get_args(type_):
                    if TypeNode.from_type(uarg).to_json() == TypeNode.from_json(data_type).to_json():
                        stack.append((parent_args, parent_key, data, uarg, type_args))
                        break
            elif TypeNode.from_type(type_) == TypeNode.from_type(TList[ANY]):
                parent_args[parent_key] = type_.from_json(data)
            elif type_ not in (int, str, bool, float, type(None)):
                for key in type_._fields:
                    field_args = {}
                    if '$data' in data:
                        stack.append((type_args, key, data['$data'].get(key), type_._fields[key], field_args))
                    else:
                        stack.append((type_args, key, data.get(key), type_._fields[key], field_args))
                if '$data' in data:
                    extra = data.get('$data', {}).keys() - type_._fields.keys()
                else:
                    extra = data.keys() - type_._fields.keys()
                if extra:
                    raise ValueError(f"There are extra attributes uknown to type {type_}: {extra}")

                post_order.insert(0, (parent_args, parent_key, type_, type_args))
            else:
                parent_args[parent_key] = data

        for (parent_args, parent_key, type_, type_args) in post_order:
            init_fields = {}
            for k, v in type_args.items():
                if type_.__dataclass_fields__[k].init:
                    init_fields[k] = v
            parent_args[parent_key] = type_(**init_fields)
            for k, v in type_args.items():
                if not type_.__dataclass_fields__[k].init:
                    setattr(parent_args[parent_key], k, v)

        return root_args['root']

T = TypeVar("T")
TK = TypeVar("TK")
TV = TypeVar("TV")


class TList(Base, Generic[T]):
    """Similar to python list except this checks type of content and won't allow to add content of
    different type then what is provided as typevar.

    Example usage: TList[str](['foo'])
    """

    _list: List[T]

    def __new__(cls, *args, **kwargs):
        if not cls._params:
            raise TypeError("Cannot create TList without subtype, construct with TList[<type>]")
        return Base.__new__(cls)

    def __init__(self, iterable=[]):
        self._list = []
        for item in iterable:
            if TypeNode.from_type(type(item)) != TypeNode.from_type(self._params[0]):
                raise TypeError(f"Cannot assign item {type(item)} to list of type {self._params[0]}")
        list.__init__(self._list, iterable)

    def __add__(self, value):
        if TypeNode.from_type(type(value)) != TypeNode.from_type(type(self)):
            raise TypeError(f"Cannot extend list {type(self)} with {type(value)}")
        return self.__class__(self._list.__add__(value._list))

    def __contains__(self, value):
        if TypeNode.from_type(type(value)) != TypeNode.from_type(self._params[0]):
            raise TypeError(f"Cannot call __contains__ on {self._params[0]} with {type(value)}")
        return self._list.__contains__(value)

    def __delitem__(self, x):
        return self._list.__delitem__(x)

    def __getitem__(self, x):
        return self._list.__getitem__(x)

    def __iter__(self):
        return self._list.__iter__()

    def __len__(self):
        return self._list.__len__()

    def __reversed__(self):
        return self._list.__reversed__()

    def __setitem__(self, key, value):
        if TypeNode.from_type(type(value)) != TypeNode.from_type(self._params[0]):
            raise TypeError(f"Cannot assign item {type(value)} to list of type {self._params[0]}")
        self._list.__setitem__(key, value)

    def append(self, obj: T) -> None:
        if TypeNode.from_type(type(obj)) != TypeNode.from_type(self._params[0]):
            raise TypeError(f"Cannot assign item {type(obj)} to list of type {self._params[0]}")
        self._list.append(obj)

    def clear(self):
        return self._list.clear()

    def count(self, value):
        return self._list.count(value)

    def extend(self, iterable):
        if TypeNode.from_type(type(iterable)) != TypeNode.from_type(type(self)):
            raise TypeError(f"Cannot extend list {self.__class__.__name__} with {iterable.__class__.__name__}")
        self._list.extend(iterable._list)

    def index(self, value, start=0, stop=-1):
        return self._list.index(value, start, stop)

    def insert(self, index, obj):
        if TypeNode.from_type(type(obj)) != TypeNode.from_type(self._params[0]):
            raise TypeError(f"Cannot assign item {type(obj)} to list of type {type(self._params[0])}")
        self._list.insert(index, obj)

    def pop(self, *args, **kwargs):
        return self._list.pop(*args, **kwargs)

    def remove(self, value):
        return self._list.remove(value)

    def reverse(self):
        return self._list.reverse()

    def sort(self, *args, **kwargs):
        return self._list.sort(*args, **kwargs)

    def to_json(self) -> Dict[str, Any]:
        pre_order: Dict[str, Any] = {}
        pre_order['root'] = {"$type": TypeNode.from_type(self.__class__).to_json(), "$data": []}
        stack: List[Tuple[Base, Dict[str, Any], str]] = []
        for n, item in enumerate(self._list):
            pre_order['root']["$data"].append(None)
            stack.append((item, pre_order['root']["$data"], n))
        while stack:
            current, current_parent, parent_key = stack.pop(0)
            if not isinstance(current, (int, str, bool, float, type(None), TList, TDict)):
                current_parent[parent_key] = {"$type": TypeNode.from_type(current.__class__).to_json(), "$data": {}}
                for f in current._fields:
                    stack.append((getattr(current, f), current_parent[parent_key]['$data'], f))
            elif isinstance(current, (TList, TDict)):
                current_parent[parent_key] = current.to_json()
            else:
                current_parent[parent_key] = current
        return pre_order['root']

    def content_to_json(self) -> Dict[str, Any]:
        pre_order: Dict[str, Any] = {}
        pre_order['root'] = []
        stack: List[Tuple[Base, Dict[str, Any], str]] = []
        for n, item in enumerate(self._list):
            pre_order['root'].append(None)
            stack.append((item, pre_order['root'], n))
        while stack:
            current, current_parent, parent_key = stack.pop(0)
            if isinstance(current, (TList, TDict, Base)):
                current_parent[parent_key] = current.content_to_json()
            else:
                current_parent[parent_key] = current
        return pre_order['root']

    @classmethod
    def from_json(cls, json_data) -> Self:
        stack = []
        post_order = []
        self_type_json = TypeNode.from_type(cls).to_json()
        root_args: Dict[str, Any] = {"root": None}
        if TypeNode.from_json(json_data['$type']) != TypeNode.from_type(cls):
            raise ValueError(f"Cannot load {json_data['$type']} to {self_type_json}")

        root_type_args = {"iterable": []}
        for n, item in enumerate(json_data['$data']):
            root_type_args['iterable'].append(None)
            if not isinstance(item, (int, str, bool, float, type(None))):
                item_type = TypeNode.from_json(item['$type']).to_type(types_cache=cls._generic_cache)
            else:
                item_type = cls._params[0]
            stack.append((root_type_args["iterable"], n, item, item_type, {}))

        while stack:
            parent_args, parent_key, data, type_, type_args = stack.pop(0)
            if hasattr(type_, "__qualname__") and type_.__qualname__ in ("Optional", "Union"):
                if isinstance(data, dict):
                    data_type = data.get("$type", None)
                else:
                    data_type = TypeNode.from_type(data.__class__).to_json()
                for uarg in get_args(type_):
                    if TypeNode.from_json(data_type) == TypeNode.from_type(uarg):
                        stack.append((parent_args, parent_key, data, uarg, type_args))
                        break
                    if data_type == TypeNode.from_type(uarg).to_json():
                        stack.append((parent_args, parent_key, data, uarg, type_args))
                        break
                else:
                    stack.append((parent_args, parent_key, data, type(None), type_args))
            elif TypeNode.from_type(type_) == TypeNode.from_type(TList[ANY]) or\
                 TypeNode.from_type(type_) == TypeNode.from_type(TDict[ANY]):
                parent_args[parent_key] = type_.from_json(data)
            elif type_ not in (int, str, bool, float, type(None)):
                for key in type_._fields:
                    field_args = {}
                    if '$data' in data:
                        stack.append((type_args, key, data['$data'].get(key), type_._fields[key], field_args))
                    else:
                        stack.append((type_args, key, data.get(key), type_._fields[key], field_args))

                post_order.insert(0, (parent_args, parent_key, type_, type_args))
            else:
                parent_args[parent_key] = data

        for (parent_args, parent_key, type_, type_args) in post_order:
            init_fields = {}
            for k, v in type_args.items():
                if type_.__dataclass_fields__[k].init:
                    init_fields[k] = v
            parent_args[parent_key] = type_(**init_fields)
            for k, v in type_args.items():
                if not type_.__dataclass_fields__[k].init:
                    setattr(parent_args[parent_key], k, v)

        root_args['root'] = cls(root_type_args['iterable'])
        return root_args['root']


class TDict(Base, Generic[TK, TV]):
    """Similar to python dict except this checks type of content and won't allow to add content of
    different type then what is provided as typevar.

    Example usage: TDict[str, int]({'foo': 1})
    """

    _dict: Dict[TK, TV]

    def __new__(cls, *args, **kwargs):
        if not cls._params:
            raise TypeError("Cannot create TDict without subtype, construct with TDict[<type>]")
        return Base.__new__(cls)

    def __contains__(self, key: TK) -> bool:
        _tk = self._params[0]
        _tv = self._params[1]
        if TypeNode.from_type(type(key)) != TypeNode.from_type(_tk):
            raise TypeError(f"Cannot check key {key} of type {type(key)} in dict of type {Dict[_tk, _tv]}")
        return self._dict.__contains__(key)

    def __delitem__(self, key: TK):
        _tk = self._params[0]
        _tv = self._params[1]
        if TypeNode.from_type(type(key)) != TypeNode.from_type(_tk):
            raise TypeError(f"Cannot remove key {key} of type {type(key)} in dict of type {Dict[_tk, _tv]}")
        self._dict.__delitem__(key)

    def __getitem__(self, key: TK) -> TV:
        _tk = self._params[0]
        _tv = self._params[1]
        if TypeNode.from_type(type(key)) != TypeNode.from_type(_tk):
            raise TypeError(f"Cannot get item by key {key} of type {type(key)} in dict of type {Dict[_tk, _tv]}")
        return self._dict.__getitem__(key)

    def __init__(self, d={}):
        self._dict = {}
        for k, v in d.items():
            self.__setitem__(k, v)

    def __iter__(self):
        return self._dict.__iter__()

    def __len__(self):
        return self._dict.__len__()

    def __reversed__(self):
        return self._dict.__reversed__()

    def __setitem__(self, k: TK, v: TV):
        _tk = self._params[0]
        _tv = self._params[1]
        if TypeNode.from_type(type(k)) != TypeNode.from_type(_tk):
            raise TypeError(f"Cannot set item by key {k} of type {type(k)} in dict of type {Dict[_tk, _tv]}")
        if TypeNode.from_type(type(v)) != TypeNode.from_type(_tv):
            raise TypeError(f"Cannot set item {v} of type {type(v)} in dict of type {Dict[_tk, _tv]}")
        self._dict.__setitem__(k, v)

    def clear(self):
        self._dict.clear()

    def fromkeys(self, iterable, value):
        _tk = self._params[0]
        _tv = self._params[1]
        for key in iterable:
            if TypeNode.from_type(type(key)) != TypeNode.from_type(_tk):
                raise TypeError(f"Cannot set item by key {key} of type {type(key)} in dict of type {Dict[_tk, _tv]}")
        if TypeNode.from_type(type(value)) != TypeNode.from_type(_tv):
            raise TypeError(f"Cannot set item {value} of type {type(value)} in dict of type {Dict[_tk, _tv]}")
        new_d = self._dict.fromkeys(iterable, value)
        return self.__class__(new_d)

    def get(self, key: TK, default=None):
        if TypeNode.from_type(type(key)) != TypeNode.from_type(TK):
            raise TypeError(f"Cannot get item by key {key} of type {type(key)} in dict of type TDict[{self._params}]")
        return self._dict.get(key, default=default)

    def items(self):
        return self._dict.items()

    def keys(self):
        return self._dict.keys()

    def pop(self, k: TK, d=None):
        _tk = self._params[0]
        _tv = self._params[1]
        if TypeNode.from_type(type(k)) != TypeNode.from_type(_tk):
            raise TypeError(f"Cannot pop item by key {k} of type {type(k)} in dict of type {Dict[_tk, _tv]}")
        return self._dict.pop(k, d)

    def popitem(self) -> Tuple[TK, TV]:
        return self._dict.popitem()

    def setdefault(self, key, default):
        _tk = self._params[0]
        _tv = self._params[1]
        if TypeNode.from_type(type(key)) != TypeNode.from_type(_tk):
            raise TypeError(f"Cannot setdefault for key {key} of type {type(key)} in dict of type {Dict[_tk, _tv]}")
        return self._dict.setdefault(key, default)

    def update(self, other):
        if TypeNode.from_type(type(other)) != TypeNode.from_type(type(self)):
            raise TypeError(f"Cannot update dict {Dict[TK, TV]} with type {type(other)}")
        self._dict.update(other)

    def values(self):
        return self._dict.values()

    def to_json(self) -> Dict[str, Any]:
        pre_order: Dict[str, Any] = {}
        pre_order['root'] = {"$type": TypeNode.from_type(self.__class__).to_json(), "$data": {}}
        stack: List[Tuple[Base, Dict[str, Any], str]] = []
        for k, v in self._dict.items():
            pre_order['root']["$data"][k] = None
            stack.append((v, pre_order['root']["$data"], k))
        while stack:
            current, current_parent, parent_key = stack.pop(0)
            if not isinstance(current, (int, str, bool, float, type(None), TDict, TList)):
                current_parent[parent_key] = {"$type": TypeNode.from_type(current.__class__).to_json(), "$data": {}}
                for f in current._fields:
                    stack.append((getattr(current, f), current_parent[parent_key]['$data'], f))
            elif isinstance(current, (TList, TDict)):
                current_parent[parent_key] = current.to_json()
            else:
                current_parent[parent_key] = current
        return pre_order['root']

    def content_to_json(self) -> Dict[str, Any]:
        pre_order: Dict[str, Any] = {}
        pre_order['root'] = {}
        stack: List[Tuple[Base, Dict[str, Any], str]] = []
        for k, v in self._dict.items():
            pre_order['root'][k] = None
            stack.append((v, pre_order['root'], k))
        while stack:
            current, current_parent, parent_key = stack.pop(0)
            if isinstance(current, (TList, TDict, Base)):
                current_parent[parent_key] = current.content_to_json()
            else:
                current_parent[parent_key] = current
        return pre_order['root']

    @classmethod
    def from_json(cls, json_data) -> Self:
        stack = []
        post_order = []
        self_type_json = TypeNode.from_type(cls).to_json()
        root_args: Dict[str, Any] = {"root": None}
        if TypeNode.from_json(json_data['$type']) != TypeNode.from_type(cls):
            raise ValueError(f"Cannot load {json_data['$type']} to {self_type_json}")

        root_type_args = {"iterable": {}}
        for k, v in json_data['$data'].items():
            root_type_args['iterable'][k] = None
            if not isinstance(v, (int, str, bool, float, type(None))):
                item_type = TypeNode.from_json(item['$type']).to_type(types_cache=cls._generic_cache)
            else:
                item_type = cls._params[0]
            stack.append((root_type_args["iterable"], k, v, item_type, {}))

        while stack:
            parent_args, parent_key, data, type_, type_args = stack.pop(0)
            if hasattr(type_, "__qualname__") and type_.__qualname__ in ("Optional", "Union"):
                if isinstance(data, dict):
                    data_type = data.get("$type", None)
                else:
                    data_type = TypeNode.from_type(data.__class__).to_json()
                for uarg in get_args(type_):
                    if TypeNode.from_json(data_type) == TypeNode.from_type(uarg):
                        stack.append((parent_args, parent_key, data, uarg, type_args))
                        break
                    if data_type == TypeNode.from_type(uarg).to_json():
                        stack.append((parent_args, parent_key, data, uarg, type_args))
                        break
                else:
                    stack.append((parent_args, parent_key, data, type(None), type_args))
            elif TypeNode.from_type(type_) == TypeNode.from_type(TList[ANY]) or\
                 TypeNode.from_type(type_) == TypeNode.from_type(TDict[ANY]):
                parent_args[parent_key] = type_.from_json(data)
            elif type_ not in (int, str, bool, float, type(None)):
                for key in type_._fields:
                    field_args = {}
                    if '$data' in data:
                        stack.append((type_args, key, data['$data'].get(key), type_._fields[key], field_args))
                    else:
                        stack.append((type_args, key, data.get(key), type_._fields[key], field_args))

                post_order.insert(0, (parent_args, parent_key, type_, type_args))
            else:
                parent_args[parent_key] = data

        for (parent_args, parent_key, type_, type_args) in post_order:
            init_fields = {}
            for k, v in type_args.items():
                if type_.__dataclass_fields__[k].init:
                    init_fields[k] = v
            parent_args[parent_key] = type_(**init_fields)
            for k, v in type_args.items():
                if not type_.__dataclass_fields__[k].init:
                    setattr(parent_args[parent_key], k, v)

        root_args['root'] = cls(root_type_args['iterable'])
        return root_args['root']

class IOStore:
    def data(self, key: str) -> Any:
        pass

    def set_data(self, key: str, val: Any):
        pass


class MemoryIOStore(IOStore):
    def __init__(self):
        self._data = {}

    def data(self, key: str) -> Any:
        return self._data.get(key, None)

    def move_data(self, old_key: str, new_key: str) -> Any:
        data = self._data.pop(old_key, None)
        self._data[new_key] = data

    def set_data(self, key: str, val: Any):
        self._data[key] = val


class _DefaultIOStore:
    def __init__(self):
        self.io_store = MemoryIOStore()


DefaultIOStore = _DefaultIOStore()


class In(Base, Generic[T]):
    """Class used for input of a Traction instance. Once connected it redirect all
    calls to internal _ref variable which is connected output, therefore it's not possible to
    access the original class.
    Class needs to be used with specific type as generic Typevar

    Example: In[str](data='foo')
    """

    _TYPE: str = "IN"

    _ref: Optional[T] = None
    # data here are actually not used after input is assigned to some output
    # it's just to deceive mypy
    data: Optional[T]
    _name: str = dataclasses.field(repr=False, init=False, default=None, compare=False)
    _owner: Optional[_Traction] = dataclasses.field(repr=False, init=False, default=None, compare=False)
    _io_store: IOStore = dataclasses.field(repr=False, init=False, compare=False)
    _uid: str = dataclasses.field(repr=False, init=False, default=None, compare=False)

    def __post_init__(self):
        self._io_store = DefaultIOStore.io_store

    def _validate_setattr_(self, name: str, value: Any) -> None:
        if name in ("_name", "_owner"):
            old_uid = self._uid
            self._uid = None
            object.__setattr__(self, name, value)
            new_uid = self.uid
            self._io_store.move_data(old_uid, new_uid)
            return

        if not name.startswith("_"):  # do not check for private attrs
            properties = dict(inspect.getmembers(self.__class__, lambda o: isinstance(o, property)))
            if name not in self._fields and not self._config.allow_extra and name not in properties:
                raise AttributeError(f"{self.__class__} doesn't have attribute {name}")
            if name == "data":
                if not hasattr(self, "_io_store"):
                    self._io_store = DefaultIOStore.io_store
                return self._io_store.set_data(self.uid, value)
            elif name not in properties:
                vtype = value.__orig_class__ if hasattr(value, "__orig_class__") else value.__class__
                tt1 = TypeNode.from_type(vtype)
                tt2 = TypeNode.from_type(self._fields[name])
                if tt1 != tt2:
                    raise TypeError(f"Cannot set attribute {self.__class__}.{name} to type {value}({vtype}), expected {self._fields[name]}")
            else:
                getattr(self.__class__, name).setter(value)
                return
        return object.__setattr__(self, name, value)

    def __getattribute__(self, name) -> Any:
        if name == 'data':
            return object.__getattribute__(self, '_io_store').data(self.uid)
        else:
            return object.__getattribute__(self, name)

    @property
    def uid(self):
        if not self._name:
            self._name = str(uuid.uuid4())
        if self._uid is None:
            self._uid = (self._owner.fullname if self._owner else "") + "::" + self._name
        return self._uid

    def content_to_json(self) -> Dict[str, Any]:
        if isinstance(self.data, (TList, TDict, Base)):
            return self.data.content_to_json()
        else:
            return self.data


class STMDSingleIn(In, Generic[T]):
    """Special input class which works like regular `In` class, however when
    used in `STMD` class definition. It's not processes as list of inputs but as
    single input used as constant input over STMD Tractions.
    """

    data: Optional[T]


class Out(STMDSingleIn, Generic[T]):
    """Class used to define Traction output. Output can be connected only to a Traction input `In` or
    `STMDSingleIn`
    """

    _TYPE: str = "OUT"

    _io_store: IOStore = dataclasses.field(repr=False, init=False, compare=False)
    data: Optional[T]
    _uid: str = dataclasses.field(repr=False, init=False, default=None, compare=False)

    def __post_init__(self):
        self._io_store = DefaultIOStore.io_store

    @property
    def uid(self):
        if not self._name:
            self._name = str(uuid.uuid4())
        if self._uid is None:
            self._uid = (self._owner.fullname if self._owner else "") + "::" + self._name
        return self._uid


class NoOut(Out, Generic[T]):
    """Special type of output indicating output hasn't been set yet.
    """

    data: Optional[T] = None
    _owner: Optional[_Traction] = dataclasses.field(repr=False, init=False, default=None, compare=False)
    _name: str = dataclasses.field(repr=False, init=False, default=None, compare=False)


class NoData(In, Generic[T]):
    """Special type of input indicating input hasn't been connected to any output.
    """

    data: Optional[T] = None


class Res(Base, Generic[T]):
    """Class represeting Traction resources.

    Usage: Res[GithubClient](r=gh_client)
    """

    _TYPE: str = "RES"
    r: T


class Arg(Base, Generic[T]):
    """Class represeting Traction argument.

    Usage: Arg[int](a=10)
    """

    _TYPE: str = "ARG"
    a: T

    def content_to_json(self) -> Dict[str, Any]:
        if isinstance(self.a, (TList, TDict, Base)):
            return self.a.content_to_json()
        else:
            return self.a


class MultiArgMeta(BaseMeta):
    @classmethod
    def _attribute_check(cls, attr, type_, all_attrs):
        if attr.startswith("a_"):
            if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(Arg[ANY]):
                raise TypeError(f"Attribute {attr} has to be type Arg[ANY], but is {type_}")
        else:
            raise TypeError(f"Attribute {attr} has start with i_, o_, a_ or r_")


class MultiArg(Base, metaclass=MultiArgMeta):
    pass


class TractionMeta(BaseMeta):
    @classmethod
    def _attribute_check(cls, attr, type_, all_attrs):
        if attr not in ('uid', 'state', 'skip', 'skip_reason', 'errors', 'stats', 'details'):
            if attr.startswith("i_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(In[ANY]):
                    raise TypeError(f"Attribute {attr} has to be type In[ANY], but is {type_}")
            elif attr.startswith("o_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(Out[ANY]):
                    raise TypeError(f"Attribute {attr} has to be type Out[ANY], but is {type_}")
            elif attr.startswith("a_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(Arg[ANY]) and\
                   TypeNode.from_type(type_, subclass_check=True) != TypeNode.from_type(MultiArg):
                    raise TypeError(f"Attribute {attr} has to be type Arg[ANY] or MultiArg, but is {type_}")
            elif attr.startswith("r_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(Res[ANY]):
                    raise TypeError(f"Attribute {attr} has to be type Res[ANY], but is {type_}")
            elif attr.startswith("d_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(str):
                    raise TypeError(f"Attribute {attr} has to be type str, but is {type_}")
                if attr != 'd_' and attr.replace("d_", "", 1) not in all_attrs['__annotations__']:
                    raise TypeError(f"Attribute {attr.replace('d_', '', 1)} is not defined for description {attr}: {all_attrs}")
            else:
                raise TypeError(f"Attribute {attr} has start with i_, o_, a_, r_ or d_")

    def __new__(cls, name, bases, attrs):
        annotations = attrs.get('__annotations__', {})
        # check if all attrs are in supported types
        for attr, type_ in annotations.items():
            # skip private attributes
            if attr.startswith("_"):
                continue
            cls._attribute_check(attr, type_, attrs)

        # record fields to private attribute
        attrs["_attrs"] = attrs
        attrs['_fields'] = {k: v for k, v in attrs.get('__annotations__', {}).items() if not k.startswith("_")}

        for f, ftype in attrs["_fields"].items():
            # Do not include outputs in init
            if f.startswith("o_") and f not in attrs:
                attrs[f] = dataclasses.field(
                    init=False,
                    default_factory=DefaultOut(type_=ftype._params[0],
                                               params=(ftype._params)))
            # Set all inputs to NoData after as default
            if f.startswith("i_") and f not in attrs:
                attrs[f] = dataclasses.field(default_factory=NoData[ftype._params])

        attrs['_fields'] = {k: v for k, v in attrs.get('__annotations__', {}).items() if not k.startswith("_")}

        cls._before_new(name, attrs, bases)

        ret = super().__new__(cls, name, bases, attrs)
        return ret


def isodate_now() -> str:
    """Return current datetime in iso8601 format."""
    return "%s%s" % (datetime.datetime.utcnow().isoformat(), "Z")


class TractionStats(Base):
    started: str = ""
    finished: str = ""
    skipped: bool = False


class TractionState(str, enum.Enum):
    """Enum-like class to store step state."""

    READY = "ready"
    PREP = "prep"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    ERROR = "error"


OnUpdateCallable = Callable[[_Traction], None]
OnErrorCallable = Callable[[_Traction], None]


class Traction(Base, metaclass=TractionMeta):
    """Class represeting basic processing element.
Traction works with data provided on defined inputs, using provided resources and arguments and
store output data to defined outputs.

Traction subclasses can have defined only 5 type of user attributes:
inputs
    every input name needs to start with ``i_``
outputs
    every output name needs to start with ``o_``
resources
    every resource name needs to start with ``r_``
arguments
    every argument name needs to start with ``a_``
documentation
    every documentation argument needs to start with ``d_``, also rest of the 
    name must be already defined field. For example `i_in1` can be described in
   ``d_i_in1``. With only ``d_`` is used as the field name, it should be used as
   description of whole traction.

example of Traction subclass

.. code-block::

    class AVG(Traction):
        a_len: Arg[int]
        a_timeframe: Arg[str]
        r_ticker_client: Res[TickerClient]
        o_avg: Out[float]

        d_a_len: str = "Size of the window for calculation."
        d_a_timeframe: str = "Timeframe which used for calculation"
        d_r_ticker_client: str = "Ticker client which provides market data"
        d_o_avg: str = "Average value of fetched candles for selected timeframe and window"
        d_: str = "Traction used to fetch last spx500 candles and calculates average of their close values"

        def run(self, on_update: Optional[OnUpdateCallable]=None):
            ret = self.r_ticker_client.r.fetch_spx_data(self.a_timeframe.a)
            closes = [x['close'] for x in ret[:self.a_len.a]]
            self.o_avg.data = sum(closes)/self.a_len.a

    tc = TickerClient(...)
    avg = AVG(uid='spx-avg',
              a_len=Arg[int](a=10),
              a_timeframe=Arg[str](a='1H'),
              r_ticker_client=Res[TickerClient](r=tc)
    )
    avg.run()
    print(avg.o_avg.data)

In the following example, output is set to Out member data. However it's also
possible to set output like this:

.. code-block::

    self.o_avg = Out[float](data=1.0)

Traction class will internally set only data of the output, reference to the output itself will not
be overwritten

"""

    _TYPE: str = "TRACTION"
    _CUSTOM_TYPE_TO_JSON: bool = False
    
    uid: str
    "Unique identifier of the current traction."
    state: TractionState = TractionState.READY
    "Indicator of current state of the traction."
    skip: bool = False
    "Flag indicating if execution of the traction was skipped."
    skip_reason: Optional[str] = ""
    "Can be se to explain why the execution of the traction was skipped."
    errors: TList[str] = dataclasses.field(default_factory=TList[str])
    """List of errors which occured during the traction execution. Inherited class should add errors
    here manually"""
    stats: TractionStats = dataclasses.field(default_factory=TractionStats)
    "Collection of traction stats"
    details: TList[str] = dataclasses.field(default_factory=TList[str])
    "List of details of the execution of the Traction. Inherited class can add details here manually"

    def __post_init__(self):
        for f in self._fields:
            if not f.startswith("o_") and not f.startswith("i_"):
                continue
            if not getattr(self, f)._owner:
                getattr(self, f)._name = f
                getattr(self, f)._owner = self

    def __getattribute_orig__(self, name: str) -> Any:
        return super().__getattribute__(name)

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("i_"):
            if name not in super().__getattribute__("_fields"):
                _class = super().__getattribute__("__class__")
                raise AttributeError(f"{_class} doesn't have attribute {name}")
            if super().__getattribute__(name)._ref:
                return super().__getattribute__(name)._ref
            #else:
            #    return NoData[super().__getattribute__(name)._params]()
        return super().__getattribute__(name)

    def _validate_setattr_(self, name: str, value: Any) -> None:
        if not name.startswith("_"):  # do not check for private attrs
            if name not in self._fields and not self._config.allow_extra:
                raise AttributeError(f"{self.__class__} doesn't have attribute {name}")

        if name.startswith("i_") or name.startswith("o_") or name.startswith("a_"):
            vtype = value.__class__
            tt1 = TypeNode.from_type(vtype)
            tt2 = TypeNode.from_type(self._fields[name])
            if tt1 != tt2:
                raise TypeError(f"Cannot set attribute {self.__class__}.{name} to type {vtype}, expected  {self._fields[name]}")

        if name.startswith("i_"):
            # Need to check with hasattr first to make sure inputs can be initialized
            if hasattr(self, name):

                # Allow overwrite default input values
                if getattr(self, name) == self.__dataclass_fields__[name].default:
                    self._no_validate_setattr_(name, value)
                    return
                connected = (
                    TypeNode.from_type(type(getattr(self, name)), subclass_check=False) != TypeNode.from_type(NoData[ANY]) and
                    TypeNode.from_type(type(getattr(self, name)), subclass_check=False) != TypeNode.from_type(In[ANY])
                )
                if connected:
                    raise AttributeError(f"Input {name} is already connected")

            # in the case input is not set, initialize it
            elif not hasattr(self, name):
                super().__setattr__(name, value)
            self.__getattribute_orig__(name)._ref = value
            return

        elif name.startswith("o_"):
            if not hasattr(self, name):
                # output is set for the first time
                super().__setattr__(name, value)

            self.__getattribute_orig__(name)._owner = self
            self.__getattribute_orig__(name)._name = name
            # Do not overwrite whole output container, rather just copy update data
            self.__getattribute_orig__(name).data = value.data
            return

        super().__setattr__(name, value)

    @property
    def fullname(self) -> str:
        """Full name of traction instance. It's composition of class name and instance uid."""
        return f"{self.__class__.__name__}[{self.uid}]"

    def to_json(self) -> Dict[str, Any]:
        ret = {}
        for f in self._fields:
            if f.startswith("i_"):
                if hasattr(getattr(self, f), "_owner") and getattr(self, f)._owner and getattr(self, f)._owner != self:
                    ret[f] = getattr(self, f)._owner.fullname + "#" + getattr(self, f)._name
                else:
                    i_json = getattr(self, f).to_json()
                    ret[f] = i_json
            elif isinstance(getattr(self, f), (int, str, bool, float, type(None))):
                ret[f] = getattr(self, f)
            else:
                ret[f] = getattr(self, f).to_json()
        ret['name'] = self.__class__.__name__
        ret['type'] = self._TYPE
        return ret

    def _getstate_to_json(self) -> Dict[str, Any]:
        ret = {}
        for f in self._fields:
            if isinstance(getattr(self, f), (int, str, bool, float, type(None))):
                ret[f] = getattr(self, f)
            else:
                ret[f] = getattr(self, f).to_json()
        return ret

    def run(
        self,
        on_update: Optional[OnUpdateCallable] = None,
        on_error: Optional[OnErrorCallable] = None,
    ) -> Self:
        """Start execution of the Traction.
* When traction is in `TractionState.READY` it runs the
user defined _pre_run method where user can do some
preparation before the run itself, potentially set `skip`
attribute to True to skip the execution. After that, traction
state is set to TractionState.PREP

* When traction is in TractionState.PREP or TractionState.ERROR, if skip is set to True 
  skipped attribute is set to True, and execution is finished.

* When skip is not set to True, state is set to TractionState.RUNNING
  and user defined _run method is executed.
If an exception is raised during the execution:
  * If exception is TractionFailedError, state is set to FAILED. This means
    traction failed with defined failure and it's not possible to rerun it

  * If unexpected exception is raised, traction state is set to ERROR which is
    state from which it's possible to rerun the traction.

 At the end of the execution traction stats are updated.
"""


        _on_update: OnUpdateCallable = on_update or on_update_empty
        _on_error: OnErrorCallable = on_error or on_update_empty
        self._reset_stats()
        if self.state == TractionState.READY:
            self.stats.started = isodate_now()

            self.state = TractionState.PREP
            self._pre_run()
            _on_update(self)  # type: ignore
        try:
            if self.state not in (TractionState.PREP, TractionState.ERROR):
                return self
            if not self.skip:
                self.state = TractionState.RUNNING
                _on_update(self)  # type: ignore
                self._run(on_update=_on_update)
        except TractionFailedError:
            self.state = TractionState.FAILED
        except Exception as e:
            self.state = TractionState.ERROR
            self.errors.append(str(e))
            _on_error(self)
            raise
        else:
            self.state = TractionState.FINISHED
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
        self.stats = TractionStats(
            started="",
            finished="",
            skipped=False,
        )

    def _finish_stats(self) -> None:
        self.stats.finished = isodate_now()
        self.stats.skipped = self.skip

    @abc.abstractmethod
    def _run(self, on_update: Optional[OnUpdateCallable] = None) -> None:  # pragma: no cover
        """Run code of the step.

        Method expects raise StepFailedError if step code fails due data error
        (incorrect configuration or missing/wrong data). That ends with step
        state set to failed.
        If error occurs due to uncaught exception in this method, step state
        will be set to error
        """
        raise NotImplementedError

    @classmethod
    def from_json(cls, json_data) -> Self:
        args = {}
        outs = {}
        for f, ftype in cls._fields.items():
            if f.startswith("i_") and isinstance(json_data[f], str):
                continue
            elif f.startswith("a_") or f.startswith("i_") or f.startswith("r_") or f in ("errors", "stats", "details"):
                args[f] = ftype.from_json(json_data[f])
            elif f.startswith("i_"):
                args[f] = ftype.from_json(json_data[f])
            elif f.startswith("o_"):
                outs[f] = ftype.from_json(json_data[f])
            else:
                args[f] = json_data[f]
        ret = cls(**args)
        for o, oval in outs.items():
            setattr(ret, o, oval)
        return ret




class STMDMeta(TractionMeta):
    _SELF_ARGS = ['a_pool_size', 'a_executor_type', 'a_delete_after_finished']

    @classmethod
    def _attribute_check(cls, attr, type_, all_attrs):
        if attr not in ('uid', 'state', 'skip', 'skip_reason', 'errors', 'stats', 'details', 'traction', 'tractions'):
            if attr.startswith("i_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(In[TList[In[ANY]]]) and \
                        TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(STMDSingleIn[ANY]):
                    raise TypeError(f"Attribute {attr} has to be type In[TList[In[ANY]]] or STMDSingleIn[ANY], but is {type_}")
            elif attr.startswith("o_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(Out[TList[Out[ANY]]]):
                    raise TypeError(f"Attribute {attr} has to be type Out[TList[Out[ANY]]], but is {type_}")
            elif attr.startswith("a_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(Arg[ANY]):
                    raise TypeError(f"Attribute {attr} has to be type Arg[ANY], but is {type_}")
            elif attr.startswith("r_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(Res[ANY]):
                    raise TypeError(f"Attribute {attr} has to be type Res[ANY], but is {type_}")
            else:
                raise TypeError(f"Attribute {attr} has start with i_, o_, a_, r_ or d_")

    def __new__(cls, name, bases, attrs):
        annotations = attrs.get('__annotations__', {})
        # check if all attrs are in supported types
        for attr, type_ in annotations.items():
            # skip private attributes
            if attr.startswith("_"):
                continue
            cls._attribute_check(attr, type_, attrs)
            if attr.startswith("i_") and attr not in attrs['_traction']._fields:
                raise ValueError(f"STMD {cls}{name} has attribute {attr} but traction doesn't have input with the same name")
            if attr.startswith("o_") and attr not in attrs['_traction']._fields:
                raise ValueError(f"STMD {cls}{name} has attribute {attr} but traction doesn't have input with the same name")
            if attr.startswith("r_") and attr not in attrs['_traction']._fields:
                raise ValueError(f"STMD {cls}{name} has attribute {attr} but traction doesn't have resource with the same name")
            if attr.startswith("a_") and attr not in cls._SELF_ARGS and attr not in attrs['_traction']._fields:
                raise ValueError(f"STMD {cls}{name} has attribute {attr} but traction doesn't have argument with the same name")

        if '_traction' not in attrs:
            raise ValueError("Missing _traction: Type[<Traction>] = <Traction> definition")

        # record fields to private attribute
        attrs["_attrs"] = attrs
        attrs['_fields'] = {k: v for k, v in attrs.get('__annotations__', {}).items() if not k.startswith("_")}

        for f, ftype in attrs["_fields"].items():
            # Do not include outputs in init
            if f.startswith("o_") and f not in attrs:
                attrs[f] = dataclasses.field(
                    init=False,
                    default_factory=DefaultOut(type_=ftype._params[0],
                                               params=(ftype._params)))

            # Set all inputs to NoData after as default
            if f.startswith("i_") and f not in attrs:
                attrs[f] = dataclasses.field(default_factory=NoData[ftype._params])

        attrs['_fields'] = {k: v for k, v in attrs.get('__annotations__', {}).items() if not k.startswith("_")}

        cls._before_new(name, attrs, bases)

        ret = super().__new__(cls, name, bases, attrs)
        return ret

    @classmethod
    def _before_new(cls, name, attrs, bases):
        outputs_map = []
        inputs_map = {}
        resources_map = {}
        args_map = {}
        for f, fo  in attrs.items():
            if f.startswith("i_"):
                inputs_map[f]= id(fo)
                outputs_map.append(id(fo))
            if f.startswith("r_"):
                resources_map[f] = id(fo)
            if f.startswith("a_"):
                args_map[f] = id(fo)

        attrs['_inputs_map'] = inputs_map
        attrs['_resources_map'] = resources_map
        attrs['_args_map'] = args_map


class STMDExecutorType(str, enum.Enum):
    LOCAL = "LOCAL"
    THREAD = "THREAD"
    PROCESS = "PROCESS"


class STMD(Traction, metaclass=STMDMeta):
    _TYPE: str = "STMD"
    uid: str
    state: str = "ready"
    skip: bool = False
    skip_reason: Optional[str] = ""
    errors: TList[str] = dataclasses.field(default_factory=TList[str])
    stats: TractionStats = dataclasses.field(default_factory=TractionStats)
    details: TList[str] = dataclasses.field(default_factory=TList[str])
    _traction: Type[Traction] = Traction
    a_pool_size: Arg[int]
    a_delete_after_finished: Arg[bool] = Arg[bool](a=True)
    a_executor_type: Arg[STMDExecutorType] = Arg[STMDExecutorType](a=STMDExecutorType.LOCAL)
    tractions: TList[Union[Traction, None]] = TList[Optional[Traction]]([])

    def _traction_runner(self, index, on_update=None):
        traction = self._copy_traction(index)
        traction.run(on_update=on_update)
        outputs = {}
        for o in traction._fields:
            if o.startswith("o_"):
                outputs[o] = getattr(traction, o)
        return outputs

    def _copy_traction(self, index, connect_inputs=True):
        """ After tractor is created, all tractions needs to be copied so it's
            possible to use same Tractor class multiple times
        """
        traction = self._traction
        init_fields = {}
        for ft, field in traction.__dataclass_fields__.items():
            # set all inputs for the traction to outputs of traction copy
            # created bellow

            if ft.startswith("r_"):
                init_fields[ft] = getattr(self, ft)

            elif ft.startswith("a_"):
                init_fields[ft] = getattr(self, ft)

            elif ft.startswith("i_") and connect_inputs:
                if TypeNode.from_type(self._fields[ft], subclass_check=False) == TypeNode.from_type(STMDSingleIn[ANY]):
                    init_fields[ft] = getattr(self, ft)
                else:
                    init_fields[ft] = getattr(self, ft).data[index]

        init_fields['uid'] = "%s:%d" % (self.fullname, index)
        # create copy of existing traction
        ret = traction(**init_fields)

        for ft in traction._fields:
            if ft.startswith("o_"):
                getattr(self, ft).data[index] = getattr(ret, ft)

        if not connect_inputs:
            return ret

        return ret

    def run(
        self,
        on_update: Optional[OnUpdateCallable] = None,
    ) -> Self:
        _on_update: OnUpdateCallable = lambda step: None
        dt = datetime.datetime.now()

        if on_update:
            _on_update = on_update

        self.state = TractionState.RUNNING

        if self.a_executor_type.a == STMDExecutorType.PROCESS:
            executor_class = ProcessPoolExecutor
        elif self.a_executor_type.a == STMDExecutorType.THREAD:
            executor_class = ThreadPoolExecutor
        else:
            executor_class = None

        inputs = {}
        for f, ftype in self._fields.items():
            if f.startswith("i_"):
                inputs[f] = getattr(self, f)
        outputs = {}
        for f in self._fields:
            if f.startswith("o_"):
                outputs[f] = getattr(self, f)

        first_in = None
        for fname in inputs.keys():
            infield = self._fields[fname]
            if TypeNode.from_type(infield, subclass_check=False) != TypeNode.from_type(STMDSingleIn[ANY]):
                first_in = inputs[fname]
                break

        if not first_in:
            raise RuntimeError("Cannot have STMD with only SingleIn inputs")

        for key in inputs:
            if inputs[key].data is None:
                raise ValueError(f"{self.fullname}: No input data for {key}")
            if TypeNode.from_type(self._fields[key], subclass_check=False) != TypeNode.from_type(STMDSingleIn[ANY]) and len(inputs[key].data) != len(first_in.data):
                raise ValueError(f"{self.__class__}: Input {key} has length {len(inputs[key].data)} but others have length {len(first_in.data)} ({list(inputs.keys())[0]})")

        for o in outputs:
            o_type = getattr(self, o).data._params[0]._params[0]
            for _ in range(len(first_in.data)):
                getattr(self, o).data.append(NoOut[o_type]())

        if executor_class:
            with executor_class(max_workers=self.a_pool_size.a) as executor:
                ft_results = {}
                self.tractions.extend(TList[Optional[Traction]]([None]*len(first_in.data)))
                for i in range(0, len(first_in.data)):
                    res = executor.submit(self._traction_runner, i, on_update=on_update)
                    ft_results[res] = i
                _on_update(self)
                for ft in as_completed(ft_results):
                    i = ft_results[ft]
                    t_outputs = ft.result()
                    for o in outputs:
                        #getattr(self, o).data[i].data = getattr(nt, o).data
                        getattr(self, o).data[i].data = t_outputs[o].data
                    if self.a_delete_after_finished.a:
                        self.tractions[i] = None
                _on_update(self)
        else:
            for i in range(0, len(first_in.data)):
                res = self._traction_runner(i, on_update=on_update)
                #self.tractions.append(self._copy_traction(i, connect_inputs=False))
                for o in outputs:
                    #getattr(self, o).data[i].data = getattr(res, o).data
                    getattr(self, o).data[i].data = res[o].data

        self.state = TractionState.FINISHED
        return self
