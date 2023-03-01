import abc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import datetime
import inspect
import json
from types import prepare_class, resolve_bases

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
    Type
)

import dataclasses


Base = ForwardRef("Base")
TList = ForwardRef("TList")
TDict = ForwardRef("TDict")


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

    def __init__(self, type_):
        self.type_ = type_
        self.children = []

    @classmethod
    def from_type(cls, type_):
        """Create TypeNode from provided type)."""

        root = cls(type_=type_)
        current = root
        stack = []
        while True:
            if hasattr(current.type_, "__args__"):
                for arg in current.type_.__args__:
                    n = cls(type_=arg)
                    stack.append(n)
                    current.children.append(n)
            if not stack:
                break
            current = stack.pop()
        return root

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

                if f"{type_.__qualname__}[{children_types}]" in types_cache:
                    node.type_ = types_cache[f"{type_.__qualname__}[{children_types}]"][0]
                else:
                    type_.__class_getitem__(tuple([x.type_ for x in node.children]))
                    node.type_ = types_cache[f"{type_.__qualname__}[{children_types}]"][0]

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

            print("EQ", n1_type, n2_type)
            if n1_type != Union and n2_type != Union:
                ch_eq &= n1_type == n2_type or issubclass(
                    n1_type, n2_type
                )
            cmp_node.eq = ch_eq

        return node.eq

    def json_compatible(self):
        if self.children:
            op = "all"
        else:
            op = "any"

        root_node = CMPNode(self, TypeNode(Union[int, str, bool, Base, TList, Dict, type(None)]), op, op)
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
                    node = CMPNode(ch1, TypeNode(Union[int, str, Dict, bool, Base, TList, type(None), TypeVar("X")]), op, op)
                    stack.insert(0, node)
                    post_order.insert(0, node)
                    current_node.children.append(node)
            else:
                for u in (int, str, TList, Dict, bool, Base, type(None), TypeVar("X")):
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
                ch_eq &= any([issubclass(n1_type, t) for t in [int, str, bool, type(None), Base, TList]])
            cmp_node.eq = ch_eq

        return root_node.eq


class BaseMeta(type):
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
            if inspect.ismethod(attrv) or inspect.isfunction(attrv):
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
        attrs['_fields'] = {k: v for k, v in attrs.get('__annotations__', {}).items() if not k.startswith("_")}

        ret = super().__new__(cls, name, bases, attrs)
        ret = dataclasses.dataclass(ret, kw_only=True)
        return ret


class Base(metaclass=BaseMeta):
    _config: ClassVar[BaseConfig] = BaseConfig()
    _fields: ClassVar[Dict[str, Any]]
    _generic_cache: ClassVar[Dict[str, Type[Any]]] = {}
    _params: ClassVar[List[Any]] = []

    def _no_validate_setattr_(self, name: str, value: Any) -> None:
        return super().__setattr__(name, value)

    def _validate_setattr_(self, name: str, value: Any) -> None:

        if not name.startswith("_"): # do not check for private attrs
            if name not in self._fields and not self._config.allow_extra:
                raise AttributeError(f"{self.__class__} doesn't have attribute name")

            vtype = value.__orig_class__ if hasattr(value, "__orig_class__") else value.__class__
            tt1 = TypeNode.from_type(vtype)
            tt2 = TypeNode.from_type(self._fields[name])
            if tt1 != tt2:
                raise TypeError(f"Cannot set attribute {self.__class__}.{name} to type {vtype}, expected {self._fields[name]}")
        return super().__setattr__(name, value)

    def __class_getitem__(cls, param):
        _params = param if isinstance(param, tuple) else (param,)
        # Create new class to have its own parametrized fields
        if f"{cls.__qualname__}[{_params}]" in cls._generic_cache:
            ret, alias = cls._generic_cache[f"{cls.__qualname__}[{_params}]"]
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
            kwds["__annotations__"][attr] = new_type

        kwds["_params"] = _params
        ret = meta(f"{cls.__qualname__}[{param}]", tuple(bases), kwds)
        alias = GenericAlias(ret, param)

        cls._generic_cache[f"{cls.__qualname__}[{_params}]"] = (ret, alias)
        return alias


T = TypeVar("T")
TK = TypeVar("TK")
TV = TypeVar("TV")


class TList(Base, Generic[T]):
    _list: List[T]

    def __init__(self, iterable):
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
            raise TypeError(f"Cannot extend list {type(self)} with {type(iterable)}")
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
        return self._dict.__contains__(TK)

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
        pass

    def get(self, key: TK, default=None):
        if TypeNode.from_type(type(key)) != TypeNode.from_type(TK):
            raise TypeError(f"Cannot get item by key {key} of type {type(key)} in dict of type {Dict[TK, TV]}")
        return self._dict.get(key, default=default)

    def items(self):
        return self._dict.items()

    def keys(self):
        return self._dict.keys()

    def pop(self, k:TK, d=None):
        if TypeNode.from_type(type(k)) != TypeNode.from_type(TK):
            raise TypeError(f"Cannot pop item by key {k} of type {type(k)} in dict of type {Dict[TK, TV]}")
        return self._dict.pop(k, d)

    def popitem(self) -> Tuple[TK, TV]:
        return self._dict.popitem()

    def setdefault(self, key, default):
        if TypeNode.from_type(type(key)) != TypeNode.from_type(TK):
            raise TypeError(f"Cannot setdefault for key {key} of type {type(key)} in dict of type {Dict[TK, TV]}")
        return self._dict.setdefault(key, default)

    def update(self, other):
        if TypeNode.from_type(type(other)) != TypeNode.from_type(type(self)):
            raise TypeError(f"Cannot update dict {Dict[TK, TV]} with type {type(other)}")
        self._dict.update(other)

    def values(self):
        return self._dict.values()

