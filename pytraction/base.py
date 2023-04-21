import abc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import datetime
import inspect
import json
from types import prepare_class, resolve_bases
import enum

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


@dataclasses.dataclass
class BaseConfig:
    validate_set_attr: bool = True
    allow_extra: bool = False


class CMPNode:
    def __init__(self, n1, n2, op, ch_op):
        self.n1 = n1
        self.n2 = n2
        self.children = []
        self.op = op
        self.eq = None
        self.ch_op = ch_op

    def __str__(self):
        return "<CMPNode n1=%s n2=%s op=%s eq=%s>" % (self.n1, self.n2, self.op, self.eq)

    def __repr__(self):
        return "<CMPNode n1=%s n2=%s op=%s eq=%s>" % (self.n1, self.n2, self.op, self.eq)


class TypeNode:
    def __str__(self):
        return "<TypeNode type=%s>" % self.type_

    def __init__(self, type_, subclass_check=True):
        self.type_ = type_
        self.subclass_check = subclass_check
        self.children = []

    @classmethod
    def from_type(cls, type_, subclass_check=True):
        """Create TypeNode from provided type)."""

        root = cls(type_=type_, subclass_check=subclass_check)
        current = root
        stack = []
        while True:
            if hasattr(current.type_, "__args__"):
                for arg in current.type_.__args__:
                    n = cls(type_=arg)
                    stack.append(n)
                    current.children.append(n)
            elif hasattr(current.type_, "_params"):
                for arg in current.type_._params:
                    n = cls(type_=arg)
                    stack.append(n)
                    current.children.append(n)
            if not stack:
                break
            current = stack.pop()
        return root

    def post_order(self):
        stack = [(self, 0, None)]
        post_order = []
        while stack:
            current, parent_index, current_parent = stack.pop(0)
            for n, ch in enumerate(current.children):
                stack.insert(0, (ch, n, current))
            post_order.insert(0, (current, parent_index, current_parent))
        return post_order

    def replace_params(self, params_map):
        """Replace Typevars in TypeNode structure with values from provided mapping."""

        stack = [(self, 0, None)]
        while stack:
            current, parent_index, current_parent = stack.pop(0)
            for n, ch in enumerate(current.children):
                stack.insert(0, (ch, n, current))
            if type(current.type_) == TypeVar:
                if current.type_ in params_map:
                    current.type_ = params_map[current.type_]

    def to_type(self, types_cache={}):
        """Return new type for TypeNode or already existing type from cache."""

        stack = [(self, 0, None)]
        post_order = []
        while stack:
            current, parent_index, current_parent = stack.pop(0)
            for n, ch in enumerate(current.children):
                stack.insert(0, (ch, n, current))
            post_order.insert(0, (current, parent_index, current_parent))

        for item in post_order:
            node, parent_index, parent = item

            if node.children:
                if hasattr(node.type_, "__origin__"):
                    type_ = node.type_.__origin__
                else:
                    type_ = type(node.type_)

                children_types = tuple([x.type_ for x in node.children])
                children_type_ids = tuple([id(x.type_) for x in node.children])

                if f"{type_.__qualname__}[{children_type_ids}]" in types_cache:
                    node.type_ = types_cache[f"{type_.__qualname__}[{children_type_ids}]"][0]
                else:
                    if type_ == Union:
                        node.type_ = Union[tuple([x.type_ for x in node.children])]
                    else:
                        type_.__class_getitem__(tuple([x.type_ for x in node.children]))
                        node.type_ = types_cache[f"{type_.__qualname__}[{children_type_ids}]"][0]

            if not parent:
                continue
            parent.children[parent_index] = node
        return post_order[-1][0].type_

    @staticmethod
    def __determine_op(ch1, ch2) -> str:
        op = "all"
        if (get_origin(ch1.type_) == Union and get_origin(ch2.type_) == Union) or (
            get_origin(ch1.type_) != Union and get_origin(ch2.type_) != Union
        ):
            op = "all"
        elif (get_origin(ch1.type_) != Union and get_origin(ch2.type_) == Union) or (
            get_origin(ch1.type_) == Union and get_origin(ch2.type_) != Union
        ):
            op = "any"
        return op

    def __eq_post_order(self, root_node):
        stack = [root_node]
        post_order = []
        post_order.insert(0, root_node)
        while stack:
            current_node = stack.pop()
            if get_origin(current_node.n1.type_) == Union and get_origin(current_node.n2.type_) != Union:
                for ch1 in current_node.n1.children:
                    node = CMPNode(ch1, current_node.n2, "all", "all")
                    stack.insert(0, node)
                    post_order.insert(0, node)
                    current_node.children.append(node)

            elif get_origin(current_node.n1.type_) != Union and get_origin(current_node.n2.type_) == Union:
                for ch2 in current_node.n2.children:
                    node = CMPNode(current_node.n1, ch2, "all", "all")
                    stack.insert(0, node)
                    post_order.insert(0, node)
                    current_node.children.append(node)

            elif current_node.op == "all":
                for ch1, ch2 in zip(current_node.n1.children, current_node.n2.children):
                    op = self.__determine_op(ch1, ch2)
                    node = CMPNode(ch1, ch2, op, op)
                    stack.insert(0, node)
                    post_order.insert(0, node)
                    current_node.children.append(node)
            else:
                if current_node.n1.type_ == Union:
                    for ch in current_node.n1.children:
                        op = self.__determine_op(ch, current_node.n2.type_)
                        node = CMPNode(ch, current_node.n2, op, op)
                        stack.insert(0, node)
                        post_order.insert(0, node)
                        current_node.children.append(node)
                else:
                    for ch in current_node.n2.children:
                        op = self.__determine_op(ch, current_node.n1.type_)
                        node = CMPNode(ch, current_node.n1, op, op)
                        stack.insert(0, node)
                        post_order.insert(0, node)
                        current_node.children.append(node)
        return post_order

    def __eq__(self, other):
        if type(other) != TypeNode:
            return False

        op = self.__determine_op(self, other)
        node = CMPNode(self, other, op, op)
        post_order = self.__eq_post_order(node)

        for cmp_node in post_order:
            if cmp_node.op == "any":
                if cmp_node.children:
                    ch_eq = any([ch.eq for ch in cmp_node.children])
                else:
                    ch_eq = True
            else:
                ch_eq = all([ch.eq for ch in cmp_node.children])

            n1_type = get_origin(cmp_node.n1.type_) or cmp_node.n1.type_
            n2_type = get_origin(cmp_node.n2.type_) or cmp_node.n2.type_

            # check types only of both types are not union
            # otherwise equality was already decided by check above

            orig_cls2 = getattr(n2_type, "_orig_cls", True)
            orig_cls1 = getattr(n1_type, "_orig_cls", False)
            #print(n1_type, n2_type, orig_cls1, orig_cls2)
            if n1_type != Union and n2_type != Union:
                ch_eq &= (n1_type == n2_type or
                          orig_cls1 == orig_cls2 or
                          (self.subclass_check and (inspect.isclass(n1_type) and inspect.isclass(n2_type) and issubclass(n1_type, n2_type))) or
                          bool(self.subclass_check and inspect.isclass(orig_cls1) and inspect.isclass(orig_cls2) and issubclass(orig_cls1, orig_cls2)))
            cmp_node.eq = ch_eq

        return node.eq

    def json_compatible(self):
        if self.children:
            op = "all"
        else:
            op = "any"

        root_node = CMPNode(self, TypeNode(Union[int, float, str, bool, float, Base, type(None)]), op, op)
        stack = [root_node]
        post_order = []
        post_order.insert(0, root_node)
        while stack:
            current_node = stack.pop()
            if current_node.n1.children:
                for ch1 in current_node.n1.children:
                    if ch1.children:
                        op = "all"
                    else:
                        op = "any"
                    node = CMPNode(ch1, TypeNode(Union[int, float, str, bool, Base, type(None), TypeVar("X")]), op, op)
                    stack.insert(0, node)
                    post_order.insert(0, node)
                    current_node.children.append(node)
            else:
                for u in (int, float, str, bool, Base, type(None), TypeVar("X")):
                    node = CMPNode(current_node.n1, TypeNode(u), "any", "any")
                    post_order.insert(0, node)
                    current_node.children.append(node)

        for cmp_node in post_order:
            n1_type = get_origin(cmp_node.n1.type_) or cmp_node.n1.type_
            n2_type = get_origin(cmp_node.n2.type_) or cmp_node.n2.type_
            if cmp_node.children:
                if cmp_node.ch_op == "any":
                    ch_eq = any([ch.eq for ch in cmp_node.children])
                else:
                    ch_eq = all([ch.eq for ch in cmp_node.children] or [True])
            else:
                ch_eq = True
            # check types only of both types are not union
            # otherwise equality was already decided by check above

            if type(n1_type) == TypeVar and type(n2_type) == TypeVar:
                ch_eq &= n1_type == n1_type
            elif type(n1_type) == TypeVar:
                if n2_type == Union:
                    ch_eq = True
                else:
                    ch_eq = False
            elif type(n2_type) == TypeVar:
                if n1_type == Union:
                    ch_eq = True
                else:
                    ch_eq = False
                ch_eq = False

            elif n1_type != Union and n2_type != Union:
                ch_eq &= issubclass(n1_type, n2_type)

            elif n1_type != Union and n2_type == Union:
                ch_eq &= any([issubclass(n1_type, t) for t in [int, float, str, bool, type(None), Base]])
            cmp_node.eq = ch_eq

        return root_node.eq

    def to_json(self) -> Dict[str, Any]:
        pre_order: Dict[str, Any] = {"root": {}}
        stack: List[Tuple[Base, Dict[str, Any], str]] = [(self, pre_order, "root")]
        while stack:
            current, current_parent, parent_key = stack.pop(0)
            type_name = current.type_.__qualname__ if hasattr(current.type_, "__qualname__") else current.type_.__name__
            current_parent[parent_key] = {"type": type_name, "args": [None,] * len(current.children)}
            for n, arg in enumerate(current.children):
                stack.append((arg, current_parent[parent_key]['args'], n))
        return pre_order['root']


class BaseMeta(type):

    @classmethod
    def _before_new(cls, attrs):
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
            if inspect.ismethod(attrv) or inspect.isfunction(attrv) or isinstance(attrv, classmethod) or isinstance(attrv, property):
                continue
            # attr starting with _ is considered to be private, there no checks
            # are applied
            if attr.startswith("_"):
                continue
            # other attributes has to be annotated
            if attr not in annotations:
                raise NoAnnotationError(f"{attr} has to be annotated")

        # check if all attrs are in supported types
        for attr, type_ in annotations.items():
            # skip private attributes
            if attr.startswith("_"):
                continue
            # Check if type of attribute is json-compatible
            if not TypeNode.from_type(annotations[attr]).json_compatible():
                raise JSONIncompatibleError(f"Attribute {attr} is not json compatible")

        # record fields to private attribute
        attrs["_attrs"] = attrs
        fields = {}
        all_annotations = {}
        defaults = {}
        for base in bases:
            for f, ft in getattr(base, "_fields", {}).items():
                fields[f] = ft

        for base in bases:
            for f, ft in fields.items():
                if hasattr(base, "__dataclass_fields__") and f in base.__dataclass_fields__:
                    if type(base.__dataclass_fields__[f].default) != dataclasses._MISSING_TYPE:
                        defaults[f] = dataclasses.field(default=base.__dataclass_fields__[f].default)
                    elif type(base.__dataclass_fields__[f].default_factory) != dataclasses._MISSING_TYPE:
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

        cls._before_new(attrs)

        ret = super().__new__(cls, name, bases, attrs)
        ret = dataclasses.dataclass(ret, kw_only=True)
        return ret


class Base(metaclass=BaseMeta):
    _config: ClassVar[BaseConfig] = BaseConfig()
    _fields: ClassVar[Dict[str, Any]] = {}
    _generic_cache: ClassVar[Dict[str, Type[Any]]] = {}
    _params: ClassVar[List[Any]] = []
    _orig_cls: Optional[Type[Any]] = None

    def __post_init__(self):
        pass

    def __getstate__(self,):
        #print("get state")
        return self.to_json()

    def __setstate__(self, state):
        #print("set state")
        new = self.from_json(state)
        for f in self._fields:
            setattr(self, f, getattr(new, f))


    def _no_validate_setattr_(self, name: str, value: Any) -> None:
        return super().__setattr__(name, value)

    def _validate_setattr_(self, name: str, value: Any) -> None:

        if not name.startswith("_"):  # do not check for private attrs
            if name not in self._fields and not self._config.allow_extra:
                raise AttributeError(f"{self.__class__} doesn't have attribute {name}")

            vtype = value.__orig_class__ if hasattr(value, "__orig_class__") else value.__class__
            tt1 = TypeNode.from_type(vtype)
            tt2 = TypeNode.from_type(self._fields[name])
            if tt1 != tt2:
                raise TypeError(f"Cannot set attribute {self.__class__}.{name} to type {vtype}, expected {self._fields[name]}")
        return super().__setattr__(name, value)

    def __class_getitem__(cls, param):
        _params = param if isinstance(param, tuple) else (param,)

        # param ids for caching as TypeVars are class instances
        # therefore has to compare with id() to get good param replacement
        #
        _param_ids = tuple([id(p) for p in param]) if isinstance(param, tuple) else (id(param),)
        # Create new class to have its own parametrized fields
        if f"{cls.__qualname__}[{_param_ids}]" in cls._generic_cache:
            ret, alias = cls._generic_cache[f"{cls.__qualname__}[{_param_ids}]"]
            return alias
        bases = [x for x in resolve_bases([cls] + list(cls.__bases__)) if x is not Generic]
        attrs = {k: v for k, v in cls._attrs.items() if k not in ("_attrs", "_fields")}
        meta, ns, kwds = prepare_class(f"{cls.__name__}[{param}]", bases, attrs)

        _params_map = dict(zip(cls.__parameters__, _params))

        # Fields needs to be copied to specific subclass, otherwise
        # it's stays shared with base class
        new_fields = {}
        for attr, type_ in cls._fields.items():
            tn = TypeNode.from_type(type_)
            tn.replace_params(_params_map)
            new_type = tn.to_type(types_cache=cls._generic_cache)
            if hasattr(new_type, "_params") and new_type._params:
                cache_key = f"{new_type.__qualname__}[{new_type._params}]"
            if new_type != type_:
                if hasattr(new_type, "_params") and new_type._params:
                    cls._generic_cache[cache_key] = (new_type, GenericAlias(new_type, new_type._params))
            if hasattr(new_type, "_params") and new_type._params:
                new_type = cls._generic_cache[cache_key][1]

            if not tn.json_compatible():
                raise JSONIncompatibleError(f"Attribute  {attr}: {new_type} is not json compatible")
            new_fields[attr] = new_type
            kwds["__annotations__"][attr] = new_type

        kwds["_params"] = _params
        kwds["_orig_cls"] = cls
        ret = meta(f"{cls.__qualname__}[{param}]", tuple(bases), kwds)
        alias = GenericAlias(ret, param)

        cls._generic_cache[f"{cls.__qualname__}[{_param_ids}]"] = (ret, alias)
        return alias

    def to_json(self) -> Dict[str, Any]:
        pre_order: Dict[str, Any] = {}
        stack: List[Tuple[Base, Dict[str, Any], str]] = [(self, pre_order, "root")]
        while stack:
            current, current_parent, parent_key = stack.pop(0)
            if not isinstance(current, (int, str, bool, float, type(None), TList)):
                current_parent[parent_key] = {"$type": TypeNode.from_type(current.__class__).to_json(), "$data": {}}
                for f in current._fields:
                    stack.append((getattr(current, f), current_parent[parent_key]['$data'], f))
            elif isinstance(current, TList):
                current_parent[parent_key] = current.to_json()
            else:
                current_parent[parent_key] = current
        return pre_order['root']

    @classmethod
    def from_json(cls, json_data) -> Self:
        stack = []
        post_order = []
        root_args: Dict[str, Any] = {"root": None}
        stack.append((root_args, 'root', json_data, cls, {}))

        while stack:
            parent_args, parent_key, data, type_, type_args = stack.pop(0)
            if hasattr(type_, "__qualname__") and type_.__qualname__ == "Optional":
                if isinstance(data, dict):
                    data_type = data.get("$type", None)
                else:
                    data_type = TypeNode.from_type(data.__class__).to_json()
                for uarg in get_args(type_):
                    if TypeNode.from_type(uarg).to_json() == data_type:
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
                    if TypeNode.from_type(uarg).to_json() == data_type:
                        stack.append((parent_args, parent_key, data, uarg, type_args))
                        break
            elif type_ == TList[ANY]:
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

        return root_args['root']


T = TypeVar("T")
TK = TypeVar("TK")
TV = TypeVar("TV")


class TList(Base, Generic[T]):
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
            raise TypeError(f"Cannot assign item {type(obj)} to list of type {type(self._params[0])}")
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
            if not isinstance(current, (int, str, bool, float, type(None))):
                current_parent[parent_key] = {"$type": TypeNode.from_type(current.__class__).to_json(), "$data": {}}
                for f in current._fields:
                    stack.append((getattr(current, f), current_parent[parent_key]['$data'], f))
            else:
                current_parent[parent_key] = current
        return pre_order['root']

    @classmethod
    def from_json(cls, json_data) -> Self:
        stack = []
        post_order = []
        root_args: Dict[str, Any] = {"root": None}
        self_type_json = TypeNode.from_type(cls).to_json()
        if json_data['$type'] != self_type_json:
            raise ValueError(f"Cannot load {json_data['$type']} to {self_type_json}")

        type_args={"iterable":[]}
        post_order.insert(0, (root_args, 'root', cls, type_args))
        for n, item in enumerate(json_data['$data']):
            type_args['iterable'].append(None)
            stack.append((type_args["iterable"], n, item, cls._params[0], {}))

        while stack:
            parent_args, parent_key, data, type_, type_args = stack.pop(0)
            if hasattr(type_, "__qualname__") and type_.__qualname__ == "Optional":
                if isinstance(data, dict):
                    data_type = data.get("$type", None)
                else:
                    data_type = TypeNode.from_type(data.__class__).to_json()
                for uarg in get_args(type_):
                    if TypeNode.from_type(uarg).to_json() == data_type:
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
                    if TypeNode.from_type(uarg).to_json() == data_type:
                        stack.append((parent_args, parent_key, data, uarg, type_args))
                        break

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

        return root_args['root']


class TDict(Base, Generic[TK, TV]):
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

    def __init__(self, d):
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
            raise TypeError(f"Cannot get item by key {key} of type {type(key)} in dict of type {Dict[TK, TV]}")
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


class In(Base, Generic[T]):
    _ref: Optional[T] = None
    # data here are actually not used after input is assigned to some output
    # it's just to deceive mypy
    data: Optional[T] = None


class Out(In, Generic[T]):
    _owner: Optional[_Traction] = dataclasses.field(repr=False, init=False, default=None, compare=False)


class NoData(In, Generic[T]):
    pass


class Res(Base, Generic[T]):
    r: T


class Arg(Base, Generic[T]):
    a: T


class TractionMeta(BaseMeta):


    @classmethod
    def _attribute_check(cls, attr, type_):
        if attr not in ('uid', 'state', 'skip', 'skip_reason', 'errors', 'stats', 'details'):
            if attr.startswith("i_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(In[ANY]):
                    raise TypeError(f"Attribute {attr} has to be type In[ANY], but is {type_}")
            elif attr.startswith("o_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(Out[ANY]):
                    raise TypeError(f"Attribute {attr} has to be type Out[ANY], but is {type_}")
            elif attr.startswith("a_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(Arg[ANY]):
                    raise TypeError(f"Attribute {attr} has to be type Arg[ANY], but is {type_}")
            elif attr.startswith("r_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(Res[ANY]):
                    raise TypeError(f"Attribute {attr} has to be type Res[ANY], but is {type_}")
            else:
                raise TypeError(f"Attribute {attr} has start with i_, o_, a_ or r_")

    def __new__(cls, name, bases, attrs):
        annotations = attrs.get('__annotations__', {})
        # check if all attrs are in supported types
        for attr, type_ in annotations.items():
            # skip private attributes
            if attr.startswith("_"):
                continue
            cls._attribute_check(attr, type_)

        # record fields to private attribute
        attrs["_attrs"] = attrs
        attrs['_fields'] = {k: v for k, v in attrs.get('__annotations__', {}).items() if not k.startswith("_")}

        for f, ftype in attrs["_fields"].items():
            # Do not include outputs in init
            if f.startswith("o_") and f not in attrs:
                attrs[f] = dataclasses.field(init=False, default_factory=Out[ftype._params])
            # Set all inputs to NoData after as default
            if f.startswith("i_") and f not in attrs:
                attrs[f] = dataclasses.field(default_factory=NoData[ftype._params])

        attrs['_fields'] = {k: v for k, v in attrs.get('__annotations__', {}).items() if not k.startswith("_")}

        cls._before_new(attrs)

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
    uid: str
    state: str = "ready"
    skip: bool = False
    skip_reason: Optional[str] = ""
    errors: TList[str] = dataclasses.field(default_factory=TList[str])
    stats: TractionStats = dataclasses.field(default_factory=TractionStats)
    details: TList[str] = dataclasses.field(default_factory=TList[str])

    def __getattribute_orig__(self, name: str) -> Any:
        return super().__getattribute__(name)

    def __getattribute__(self, name: str) -> Any:
        if name.startswith("i_"):
            if name not in super().__getattribute__("_fields"):
                _class = super().__getattribute__("__class__")
                raise AttributeError(f"{_class} doesn't have attribute {name}")
            if super().__getattribute__(name)._ref:
                return super().__getattribute__(name)._ref
            else:
                return NoData[super().__getattribute__(name)._params]()
        return super().__getattribute__(name)

    def _validate_setattr_(self, name: str, value: Any) -> None:
        if not name.startswith("_"):  # do not check for private attrs
            if name not in self._fields and not self._config.allow_extra:
                raise AttributeError(f"{self.__class__} doesn't have attribute {name}")

        if name.startswith("i_") or name.startswith("o_") or name.startswith("a_"):
            vtype = value.__orig_class__ if hasattr(value, "__orig_class__") else value.__class__
            tt1 = TypeNode.from_type(vtype)
            tt2 = TypeNode.from_type(self._fields[name])
            if tt1 != tt2:
                raise TypeError(f"Cannot set attribute {self.__class__}.{name} to type {vtype}, expected {self._fields[name]}")

        if name.startswith("i_"):
            # Need to check with hasattr first to make sure inputs can be initialized
            if hasattr(self, name) and TypeNode.from_type(type(getattr(self, name)), subclass_check=False) != TypeNode.from_type(NoData[ANY]):
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
            # Do not overwrite whole output container, rather just copy update data
            self.__getattribute_orig__(name).data = value.data
            return

        elif name.startswith("a_"):
            if not hasattr(self, name):
                # output is set for the first time
                super().__setattr__(name, value)
            # Do not overwrite whole output container, rather just copy update data
            self.__getattribute_orig__(name).a = value.a
            return

        super().__setattr__(name, value)

    @property
    def fullname(self) -> str:
        """Full name of class instance."""
        return f"{self.__class__.__name__}[{self.uid}]"
    

    def to_json(self) -> Dict[str, Any]:
        ret = {}
        for f in self._fields:
            if f.startswith("i_"):
                if hasattr(getattr(self, f), "_owner") and getattr(self, f)._owner:
                    ret[f] = getattr(self, f)._owner.fullname
                else:
                    ret[f] = getattr(self, f).to_json()
            elif isinstance(getattr(self, f), (int, str, bool, float, type(None))):
                ret[f] = getattr(self, f)
            else:
                ret[f] = getattr(self, f).to_json()
        return ret

    def run(
        self,
        on_update: Optional[OnUpdateCallable] = None,
        on_error: Optional[OnErrorCallable] = None,
    ) -> Self:
        _on_update: OnUpdateCallable = on_update or (lambda tr: None)
        _on_error: OnErrorCallable = on_error or (lambda tr: None)
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


class TractorMeta(TractionMeta):
    @classmethod
    def _attribute_check(cls, attr, type_):
        if attr not in ('uid', 'state', 'skip', 'skip_reason', 'errors', 'stats', 'details', 'tractions'):
            if attr.startswith("i_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(In[ANY]):
                    raise TypeError(f"Attribute {attr} has to be type In[ANY], but is {type_}")
            elif attr.startswith("o_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(Out[ANY]):
                    raise TypeError(f"Attribute {attr} has to be type Out[ANY], but is {type_}")
            elif attr.startswith("a_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(Arg[ANY]):
                    raise TypeError(f"Attribute {attr} has to be type Arg[ANY], but is {type_}")
            elif attr.startswith("r_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(Res[ANY]):
                    raise TypeError(f"Attribute {attr} has to be type Res[ANY], but is {type_}")
            elif attr.startswith("t_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(Traction):
                    raise TypeError(f"Attribute {attr} has to be type Traction, but is {type_}")
            else:
                raise TypeError(f"Attribute {attr} has start with i_, o_, a_, r_ or t_")

    @classmethod
    def _before_new(cls, attrs):
        outputs_map = []
        inputs_map = {}
        resources_map = {}
        for f, fo  in attrs.items():
            if f.startswith("i_"):
                inputs_map[f]= id(fo)
                outputs_map.append(id(fo))
            if f.startswith("r_"):
                resources_map[f] = id(fo)

        attrs['_inputs_map'] = inputs_map
        attrs['_resources_map'] = resources_map

        for f in attrs['_fields']:
            if not f.startswith("t_"):
                continue
            traction = attrs[f]
            for tf in traction._fields:
                tfo = getattr(traction, tf)
                if tf.startswith("o_"):
                    outputs_map.append(id(tfo))
                if tf.startswith("i_"):
                    if TypeNode.from_type(type(tfo), subclass_check=False) != TypeNode.from_type(NoData[ANY]):
                        if id(getattr(traction, tf)) not in outputs_map:
                            raise ValueError(f"Input {traction.__class__}[{traction.uid}]->{tf} is mapped to output which is not known yet")


class Tractor(Traction, metaclass=TractorMeta):
    uid: str
    state: str = "ready"
    skip: bool = False
    skip_reason: Optional[str] = ""
    errors: TList[str] = dataclasses.field(default_factory=TList[str])
    stats: TractionStats = dataclasses.field(default_factory=TractionStats)
    details: TList[str] = dataclasses.field(default_factory=TList[str])
    tractions: TList[Traction] = dataclasses.field(default_factory=TList[Traction], init=False)

    def __post_init__(self):
        """ After tractor is created, all tractions needs to be copied so it's
            possible to use same Tractor class multiple times
        """
        output_map = {}
        resources_map = {}

        # record all outputs of the tractor
        for f in self._fields:
            if f.startswith("o_"):
                output_map[id(getattr(self, f))] = getattr(self, f)

        # In the case inputs are overwritten by user provided to __init__
        for f in self._inputs_map:
            if f.startswith("i_"):
                output_map[self._inputs_map[f]] = getattr(self, f)

        # In the case resources are overwritten by user provided to __init__
        for f in self._resources_map:
            if f.startswith("r_"):
                resources_map[self._resources_map[f]] = getattr(self, f)

        for f in self._fields:
            # Copy all tractions
            if f.startswith("t_"):
                traction = getattr(self, f)
                init_fields = {}
                for ft, field in traction.__dataclass_fields__.items():
                    # set all inputs for the traction to outputs of traction copy
                    # created bellow
                    if ft.startswith("i_"):
                        init_fields[ft] = output_map[id(getattr(traction, ft))]

                    elif ft.startswith("r_"):
                        init_fields[ft] = resources_map[id(getattr(traction, ft))]

                    # if field doesn't start with _ include it in init_fields to
                    # initialize the traction copy
                    elif field.init:
                        if ft.startswith("_"):
                            continue
                        init_fields[ft] = getattr(traction, ft)

                # create copy of existing traction
                new_traction = traction.__class__(**init_fields)

                # also put new traction in tractions list used in run
                self.tractions.append(new_traction)

                # map outputs of traction copy to outputs of original
                for ft in new_traction._fields:
                    if ft.startswith("o_"):
                        output_map[id(getattr(traction, ft))] = getattr(new_traction, ft)
                setattr(self, f, new_traction)

        # update all Tractor outputs to outputs of copies of original tractions
        for f in  self._fields:
            if f.startswith("o_"):
                # regular __setattr__ don't overwrite whole output model but just 
                # data in it to keep connection, so need to use _no_validate_setattr_
                self._no_validate_setattr_(f, output_map[id(getattr(self, f))])

    def _run(self, on_update: Optional[OnUpdateCallable] = None) -> Self:  # pragma: no cover
        for traction in self.tractions:
            traction.run(on_update=on_update)
            if on_update:
                on_update(self)
            if traction.state == TractionState.ERROR:
                self.state = TractionState.ERROR
                return self
            if traction.state == TractionState.FAILED:
                self.state = TractionState.FAILED
                return self
        return self

    def run(
        self,
        on_update: Optional[OnUpdateCallable] = None,
        on_error: Optional[OnErrorCallable] = None,
    ) -> Self:
        _on_update: OnUpdateCallable = on_update or (lambda tr: None)
        _on_error: OnErrorCallable = on_error or (lambda tr: None)
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


class STMDMeta(TractionMeta):
    @classmethod
    def _attribute_check(cls, attr, type_):
        if attr not in ('uid', 'state', 'skip', 'skip_reason', 'errors', 'stats', 'details', 'traction', 'tractions'):
            if attr.startswith("i_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(In[TList[In[ANY]]]):
                    raise TypeError(f"Attribute {attr} has to be type In[TList[In[ANY]]], but is {type_}")
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
                raise TypeError(f"Attribute {attr} has start with i_, o_, a_ or r_")

    def __new__(cls, name, bases, attrs):
        annotations = attrs.get('__annotations__', {})
        # check if all attrs are in supported types
        for attr, type_ in annotations.items():
            # skip private attributes
            if attr.startswith("_"):
                continue
            cls._attribute_check(attr, type_)
            if attr.startswith("i_") and attr not in attrs['traction']._fields:
                raise ValueError("STMD {cls}{name} has attribute {attr} but traction doesn't have input with the same name")
            if attr.startswith("o_") and attr not in attrs['traction']._fields:
                raise ValueError("STMD {cls}{name} has attribute {attr} but traction doesn't have input with the same name")

        # record fields to private attribute
        attrs["_attrs"] = attrs
        attrs['_fields'] = {k: v for k, v in attrs.get('__annotations__', {}).items() if not k.startswith("_")}

        for f, ftype in attrs["_fields"].items():
            # Do not include outputs in init
            if f.startswith("o_") and f not in attrs:
                attrs[f] = dataclasses.field(init=False, default_factory=Out[TList[Out[ftype._params]]])
            # Set all inputs to NoData after as default
            if f.startswith("i_") and f not in attrs:
                attrs[f] = dataclasses.field(default_factory=In[TList[In[ftype._params]]])

        attrs['_fields'] = {k: v for k, v in attrs.get('__annotations__', {}).items() if not k.startswith("_")}

        cls._before_new(attrs)

        ret = super().__new__(cls, name, bases, attrs)
        return ret

    @classmethod
    def _before_new(cls, attrs):
        outputs_map = []
        inputs_map = {}
        resources_map = {}
        for f, fo  in attrs.items():
            if f.startswith("i_"):
                inputs_map[f]= id(fo)
                outputs_map.append(id(fo))
            if f.startswith("r_"):
                resources_map[f] = id(fo)

        attrs['_inputs_map'] = inputs_map
        attrs['_resources_map'] = resources_map


class STMD(Traction, metaclass=STMDMeta):
    uid: str
    state: str = "ready"
    skip: bool = False
    skip_reason: Optional[str] = ""
    errors: TList[str] = dataclasses.field(default_factory=TList[str])
    stats: TractionStats = dataclasses.field(default_factory=TractionStats)
    details: TList[str] = dataclasses.field(default_factory=TList[str])
    traction: Traction
    a_pool_size: Arg[int]
    a_use_processes: Arg[bool] = Arg[bool](a=False)
    tractions: TList[Traction] = TList[Traction]([])

    def _copy_traction(self, index):
        """ After tractor is created, all tractions needs to be copied so it's
            possible to use same Tractor class multiple times
        """
        #resources_map = {}
        # In the case resources are overwritten by user provided to __init__
        for f in self._resources_map:
            if f.startswith("r_"):
                resources_map[self._resources_map[f]] = getattr(self, f)

        traction = self.traction
        init_fields = {}
        for ft, field in traction.__dataclass_fields__.items():
            # set all inputs for the traction to outputs of traction copy
            # created bellow
            if ft.startswith("i_"):
                init_fields[ft] = getattr(self, ft).data[index]

            elif ft.startswith("r_"):
                init_fields[ft] = resources_map[id(getattr(traction, ft))]

            # if field doesn't start with _ include it in init_fields to
            # initialize the traction copy
            elif field.init:
                if ft.startswith("_"):
                    continue
                init_fields[ft] = getattr(traction, ft)

        init_fields['uid'] = "%s:%d" % (self.uid, index)
        # create copy of existing traction
        ret = traction.__class__(**init_fields)

        for ft in traction._fields:
            if ft.startswith("o_"):
                getattr(self, ft).data.append(getattr(traction, ft))


        return ret

    def run(
        self,
        on_update: Optional[OnUpdateCallable] = None,
    ) -> Self:
        _on_update: OnUpdateCallable = lambda step: None
        if on_update:
            _on_update = on_update

        self.state = TractionState.RUNNING

        if self.a_use_processes:
            executor_class = ProcessPoolExecutor
        else:
            executor_class = ThreadPoolExecutor

        inputs = {}
        for f in self._fields:
            if f.startswith("i_"):
                inputs[f] = getattr(self, f)

        first_in = inputs[list(inputs.keys())[0]]
        for key in inputs:
            if len(inputs[key].data) != len(first_in.data):
                raise ValueError(f"Input {key} has length {len(inputs[key].data)} but others have length len(first_in.data)")

        with executor_class(max_workers=self.a_pool_size.a) as executor:
            ft_results = {}
            for i in range(0, len(list(inputs.values())[0].data)):
                print("RUN", i)
                t = self._copy_traction(i)
                self.tractions.append(t)
                res = executor.submit(t.run, on_update=on_update)
                ft_results[res] = (t, i)
                #self.results.data.append(self.TractorType.__fields__["results"].type_())
            _on_update(self)
            for ft in as_completed(ft_results):
                (_, i) = ft_results[ft]
                nt = ft.result()
                print("T", nt)
                self.tractions[i] = nt
                #self.results.data[i] = nt.results
                #self.details.data[i] = nt.details
                _on_update(self)
            print(self)

        self.state = TractionState.FINISHED
        return self
