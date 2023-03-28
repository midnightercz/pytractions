import abc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import datetime
import inspect
import json
from types import prepare_class, resolve_bases
import enum

import datetime

from typing import (
    Dict,
    List,
    Any,
    ClassVar,
    Union,
    Union,
    get_origin,
    ForwardRef,
    Optional,
    TypeVar,
    Generic,
    GenericAlias,
    Tuple,
    Type,
    Callable
)

import dataclasses

from .exc import TractionFailedError

from .utils import ANY

Base = ForwardRef("Base")
TList = ForwardRef("TList")
TDict = ForwardRef("TDict")
Traction = ForwardRef("Traction")


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
            #print("FROM TYPE", current, dir(current.type_))
            if hasattr(current.type_, "__args__"):
                for arg in current.type_.__args__:
                    #print("ARG", arg)
                    n = cls(type_=arg)
                    stack.append(n)
                    current.children.append(n)
            elif hasattr(current.type_, "_params"):
                for arg in current.type_._params:
                    #print("ARG", arg)
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
                    print("TYPE", type_)
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
            if n1_type != Union and n2_type != Union:
                ch_eq &= (n1_type == n2_type or
                          orig_cls1 == orig_cls2 or
                          (self.subclass_check and issubclass(n1_type, n2_type)) or
                          (self.subclass_check and orig_cls1 and orig_cls2 and issubclass(orig_cls1, orig_cls2)))
            cmp_node.eq = ch_eq

        return node.eq

    def json_compatible(self):
        if self.children:
            op = "all"
        else:
            op = "any"

        root_node = CMPNode(self, TypeNode(Union[int, str, bool, Base, TList, TDict, type(None)]), op, op)
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
                    node = CMPNode(ch1, TypeNode(Union[int, str, bool, Base, TList, TDict, type(None), TypeVar("X")]), op, op)
                    stack.insert(0, node)
                    post_order.insert(0, node)
                    current_node.children.append(node)
            else:
                for u in (int, str, TList, TDict, bool, Base, type(None), TypeVar("X")):
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
                ch_eq &= issubclass(n1_type, n1_type)

            elif n1_type != Union and n2_type == Union:
                ch_eq &= any([issubclass(n1_type, t) for t in [int, str, bool, type(None), Base, TList, TDict]])
            cmp_node.eq = ch_eq

        return root_node.eq


class BaseMeta(type):
    def __new__(cls, name, bases, attrs):

        #print("BASE META", name)
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
            #print(attr, attrv)
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

        #print("----")
        #print("attrs", attrs)
        #print("----")
        #print("fields", fields)
        for base in bases:
            #print(dir(base))
            for f, ft in fields.items():
                if hasattr(base, "__dataclass_fields__") and f in base.__dataclass_fields__:
                    #print(base.__dataclass_fields__[f])
                    if type(base.__dataclass_fields__[f].default) != dataclasses._MISSING_TYPE:
                        #print("1 default")
                        defaults[f] = dataclasses.field(default=base.__dataclass_fields__[f].default)
                    elif type(base.__dataclass_fields__[f].default_factory) != dataclasses._MISSING_TYPE:
                        #print("2 default")
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
        #print("----")
        #for k,v in attrs.items():
        #    #print(k," ",v)
        #    #print("--")
        #print("----")

        #print("all annotations", all_annotations)
        #print("attrs", attrs)
        #print("---")
        #print("defaults", defaults)

        attrs['_fields'] = fields
        #print("ATTRS", attrs)

        ret = super().__new__(cls, name, bases, attrs)
        #print("META ret dir", dir(ret))
        ret = dataclasses.dataclass(ret, kw_only=True)
        #print("META ret match_args", ret.__match_args__)
        #print("META ret parameters", ret.__parameters__ if hasattr(ret, "__parameters__") else None)
        return ret


class Base(metaclass=BaseMeta):
    _config: ClassVar[BaseConfig] = BaseConfig()
    _fields: ClassVar[Dict[str, Any]] = {}
    _generic_cache: ClassVar[Dict[str, Type[Any]]] = {}
    _params: ClassVar[List[Any]] = []
    _orig_cls: Type[Any] = None

    def _no_validate_setattr_(self, name: str, value: Any) -> None:
        return super().__setattr__(name, value)

    def _validate_setattr_(self, name: str, value: Any) -> None:
        print("base validate setattr", name, value)

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
        #print("GET ITEM", cls, param)
        _params = param if isinstance(param, tuple) else (param,)

        # param ids for caching as TypeVars are class instances
        # therefore has to compare with id() to get good param replacement
        #
        _param_ids = tuple([id(p) for p in param]) if isinstance(param, tuple) else (id(param),)
        # Create new class to have its own parametrized fields
        if f"{cls.__qualname__}[{_param_ids}]" in cls._generic_cache:
            ret, alias = cls._generic_cache[f"{cls.__qualname__}[{_param_ids}]"]
            #print("CLASS ITEM CACHED", alias)
            return alias
        bases = [x for x in resolve_bases([cls] + list(cls.__bases__)) if x is not Generic]
        attrs = {k: v for k, v in cls._attrs.items() if k not in ("_attrs", "_fields")}
        meta, ns, kwds = prepare_class(f"{cls.__name__}[{param}]", bases, attrs)

        _params_map = dict(zip(cls.__parameters__, _params))

        # Fields needs to be copied to specific subclass, otherwise
        # it's stays shared with base class
        new_fields = {}
        #print("CLASS GET ITEM FIELDS")
        for attr, type_ in cls._fields.items():
            print("FIELD", attr, type_)
            tn = TypeNode.from_type(type_)
            tn.replace_params(_params_map)
            print("NEW TYPE")
            new_type = tn.to_type(types_cache=cls._generic_cache)
            print(new_type)
            if hasattr(new_type, "_params"):
                cache_key = f"{new_type.__qualname__}[{new_type._params}]"
            if new_type != type_:
                if hasattr(new_type, "_params"):
                    cls._generic_cache[cache_key] = (new_type, GenericAlias(new_type, new_type._params))
            if hasattr(new_type, "_params"):
                new_type = cls._generic_cache[cache_key][1]

            if not tn.json_compatible():
                raise JSONIncompatibleError(f"Attribute  {attr}: {new_type} is not json compatible")
            new_fields[attr] = new_type
            #print("KWDS", kwds)
            kwds["__annotations__"][attr] = new_type

        kwds["_params"] = _params
        kwds["_orig_cls"] = cls
        ret = meta(f"{cls.__qualname__}[{param}]", tuple(bases), kwds)
        #print("GETITEM DIR RET", dir(ret))
        alias = GenericAlias(ret, param)
        #print("GETITEM DIR ALIAS", dir(alias))

        cls._generic_cache[f"{cls.__qualname__}[{_param_ids}]"] = (ret, alias)
        #print("CLASS ITEM", alias)
        return alias

    def to_json(self) -> Dict[str, Any]:
        pre_order: Dict[str, Any] = {}
        stack: List[Tuple[Type[Base], Dict[str, Any], str]] = [(self, pre_order, "root")]
        while stack:
            current, current_parent, parent_key = stack.pop(0)
            if not isinstance(current, (int, str, bool, type(None))):
                current_parent[parent_key] = {}
                for f in current._fields:
                    stack.append((getattr(current, f), current_parent[parent_key], f))
            else:
                current_parent[parent_key] = current
        return pre_order['root']

    @classmethod
    def from_json(cls, json_data) -> Base:
        stack = []
        post_order = []
        #print(json_data)

        cls_args = {}
        for key in json_data:
            if key not in cls._fields:
                raise TypeError(f"'{key}' doesn't exist in {cls.__qualname__}")
            field_args = {}
            stack.append((cls_args, key, json_data.get(key), cls._fields[key], field_args))
            if  cls._fields[key] not in (int, str, bool, type(None)):
                post_order.append((cls_args, key, cls._fields[key], field_args))

        while stack:
            parent_args, parent_key, data, type_, type_args = stack.pop(0)
            #print("DATA", data)
            if type_ not in (int, str, bool):
                for key in type_._fields:
                    field_args = {}
                    stack.append((type_args, key, data.get(key), type_._fields[key], field_args))
                    if type_._fields[key] not in (int, str, bool, type(None)):
                        post_order.append((type_args, field_args, key, type_._fields[key], field_args))
            else:
                parent_args[parent_key] = data
            #print("PARENT ARGS", parent_args)

        #print("--", post_order)
        for (parent_args, parent_key, type_, type_args) in post_order:
            #print(type_args)
            parent_args[parent_key]=type_(**type_args)
            #print("parent_args", parent_args)
        #print(cls_args)
        return cls(**cls_args)




T = TypeVar("T")
TK = TypeVar("TK")
TV = TypeVar("TV")


class TList(Base, Generic[T]):
    _list: List[T]

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
        #print("--EQ--")
        if TypeNode.from_type(type(iterable)) != TypeNode.from_type(type(self)):
            #print(self.__class__.__name__)
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


class TDict(Base, Generic[TK, TV]):
    _dict: Dict[TK, TV]

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


class Out(In, Generic[T]):
    data: Optional[T] = None
    _owner: Optional[Traction] = dataclasses.field(repr=False, init=False, default=None)


class NoData(In, Generic[T]):
    pass


class Res(Base, Generic[T]):
    pass


class Arg(Base, Generic[T]):
    pass


class TractionMeta(BaseMeta):
    def __new__(cls, name, bases, attrs):
        annotations = attrs.get('__annotations__', {})
        # check if all attrs are in supported types
        for attr, type_ in annotations.items():
            # skip private attributes
            if attr.startswith("_"):
                continue
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

        # record fields to private attribute
        print(attrs)
        attrs["_attrs"] = attrs
        attrs['_fields'] = {k: v for k, v in attrs.get('__annotations__', {}).items() if not k.startswith("_")}

        for f, ftype in attrs["_fields"].items():
            #if not f.startswith("i_"):
            #    continue
            #if f not in attrs:
            #    attrs[f] = NoData[ftype._params]()

            # Do not include outputs in init
            if f.startswith("o_") and f not in attrs:
                attrs[f] = dataclasses.field(init=False, default_factory=Out[ftype._params])
            # Set all inputs to NoData after as default
            if f.startswith("i_") and f not in attrs:
                attrs[f] = NoData[ftype._params]()

        attrs['_fields'] = {k: v for k, v in attrs.get('__annotations__', {}).items() if not k.startswith("_")}


        ret = super().__new__(cls, name, bases, attrs)
        #ret = dataclasses.dataclass(ret, kw_only=True)
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


OnUpdateCallable = Callable[[Traction], None]
OnErrorCallable = Callable[[Traction], None]


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
            print("GET ATTR", name, super().__getattribute__(name))
            if super().__getattribute__(name)._ref:
                print("RETURN ref", super().__getattribute__(name)._ref)
                return super().__getattribute__(name)._ref
            else:
                return NoData[super().__getattribute__(name)._params]()
        return super().__getattribute__(name)

    def _validate_setattr_(self, name: str, value: Any) -> None:
        if not name.startswith("_"):  # do not check for private attrs
            if name not in self._fields and not self._config.allow_extra:
                raise AttributeError(f"{self.__class__} doesn't have attribute {name}")
        if name.startswith("i_"):
            vtype = value.__orig_class__ if hasattr(value, "__orig_class__") else value.__class__
            tt1 = TypeNode.from_type(vtype)
            tt2 = TypeNode.from_type(self._fields[name])
            if tt1 != tt2:
                raise TypeError(f"Cannot set attribute {self.__class__}.{name} to type {vtype}, expected {self._fields[name]}")
            #print("---> set attr", type(getattr(self, name)))
            #print("--- ", TypeNode.from_type(type(getattr(self, name)), subclass_check=False) == TypeNode.from_type(NoData[ANY]))
            #print("--- ", TypeNode.from_type(type(getattr(self, name)), subclass_check=False) == TypeNode.from_type(NoData[ANY]))

            if TypeNode.from_type(type(getattr(self, name)), subclass_check=False) != TypeNode.from_type(NoData[ANY]):
                raise AttributeError(f"Input {name} is already connected")
            self.__getattribute_orig__(name)._ref = value
            return

        # elif name.startswith("o_"):
        #     vtype = value.__orig_class__ if hasattr(value, "__orig_class__") else value.__class__
        #     tt1 = TypeNode.from_type(vtype)
        #     tt2 = TypeNode.from_type(self._fields[name])
        #     if tt1 != tt2:
        #         raise TypeError(f"Cannot set attribute {self.__class__}.{name} to type {vtype}, expected {self._fields[name]}")
        #
        #     if TypeNode.from_type(type(getattr(self, name)), subclass_check=False) != TypeNode.from_type(NoData[ANY]):
        #         raise AttributeError(f"Input {name} is already connected")
        #     self.__getattribute_orig__(name)._ref = value

        super().__setattr__(name, value)

    @property
    def fullname(self) -> str:
        """Full name of class instance."""
        return f"{self.__class__.__qualname__}[{self.uid}]"

    def run(
        self,
        on_update: Optional[OnUpdateCallable] = None,
        on_error: Optional[OnErrorCallable] = None,
    ) -> Traction:
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
            self.errors.errors["exception"] = str(e)
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
            started=None,
            finished=None,
            skipped=False,
        )

    def _finish_stats(self) -> None:
        self.stats.finished = isodate_now()
        self.stats.skipped = self.skip

    @abc.abstractmethod
    def _run(self, on_update: OnUpdateCallable = None) -> None:  # pragma: no cover
        """Run code of the step.

        Method expects raise StepFailedError if step code fails due data error
        (incorrect configuration or missing/wrong data). That ends with step
        state set to failed.
        If error occurs due to uncaught exception in this method, step state
        will be set to error
        """
        raise NotImplementedError
