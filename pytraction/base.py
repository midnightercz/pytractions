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
    ForwardRef
)

import dataclasses

from utils import (
    get_type, _get_args, check_type, type_tree
)

Base = ForwardRef("Base")

def find_attr(objects, attr_name):
    for o in objects:
        if hasattr(o, attr_name):
            return getattr(o, attr_name)


@dataclasses.dataclass
class BaseConfig:
    validate_set_attr: bool = True
    allow_extra: bool = False


class CMPNode:
    def __init__(self, n1, n2, ch_op):
        self.n1 = n1
        self.n2 = n2
        self.children = []
        self.op = ch_op
        self.eq = None

    def __str__(self):
        return "<CMPNode n1=%s n2=%s op=%s eq=%s>" % (self.n1, self.n2, self.op, self.eq)

    def __repr__(self):
        return "<CMPNode n1=%s n2=%s op=%s eq=%s>" % (self.n1, self.n2, self.op, self.eq)


class Node:
    def __str__(self):
        return "<Node type=%s>" % self.type_

    def __init__(self, type_):
        self.type_ = type_
        self.children = []

    @staticmethod
    def __determine_op(self, ch1, ch2) -> str:
        op = "all"
        if (ch1.type_ == Union and ch2.type_ == Union) or (
            ch1.type_ == Union and ch2.type_ != Union
        ):
            op = "all"
        elif (ch1.type_ != Union and ch2.type_ == Union) or (
            ch1.type_ == Union and ch2.type_ != Union
        ):
            op = "any"
        return op

    def __eq_post_order(self, root_node):
        stack = [root_node]
        post_order = []
        post_order.insert(0, root_node)
        while stack:
            current_node = stack.pop()
            if current_node.op == "all":
                for ch1, ch2 in zip(current_node.n1.children, current_node.n2.children):
                    op = self.__determine_op(ch1, ch2)
                    node = CMPNode(ch1, ch2, op)
                    stack.insert(0, node)
                    post_order.insert(0, node)
                    current_node.children.append(node)
            else:
                if current_node.n1.type_ == Union:
                    for ch in current_node.n1.children:
                        op = self.__determine_op(ch, current_node.n2.type_)
                        node = CMPNode(ch, current_node.n2, op)
                        stack.insert(0, node)
                        post_order.insert(0, node)
                        current_node.children.append(node)
                else:
                    for ch in current_node.n2.children:
                        op = self.__determine_op(ch, current_node.n1.type_)
                        node = CMPNode(ch, current_node.n1, op)
                        stack.insert(0, node)
                        post_order.insert(0, node)
                        current_node.children.append(node)
        return post_order

    def __eq__(self, other):
        if type(other) != Node:
            return False

        op = self.__determine_op(self, other)
        node = CMPNode(self, other, op)
        post_order = self.__eq_post_order(node)

        for cmp_node in post_order:
            if cmp_node.op == "any":
                if cmp_node.children:
                    ch_eq = any([ch.eq for ch in cmp_node.children])
                else:
                    ch_eq = True
            else:
                ch_eq = all([ch.eq for ch in cmp_node.children])

            if hasattr(cmp_node.n1.type_, "__args__"):
                n1_type = cmp_node.n1.type_.__origin__
            else:
                n1_type = cmp_node.n1.type_
            if hasattr(cmp_node.n2.type_, "__args__"):
                n2_type = cmp_node.n2.type_.__origin__
            else:
                n2_type = cmp_node.n2.type_

            ch_eq &= n1_type == n2_type or issubclass(
                n1_type, n2_type
            )
            if not ch_eq:
                return False
        return True

    def json_compatible(self):
        root_node = CMPNode(self, Node(Union[int, str, List, Dict, bool, Base]), "any")
        stack = [root_node]
        post_order = []
        post_order.insert(0, root_node)
        while stack:
            current_node = stack.pop()
            print("CURRENT", current_node, "###", current_node.n1.children)
            if current_node.n1.children:
                for ch1 in current_node.n1.children:
                    node = CMPNode(ch1, Node(Union[int, str, List, Dict, bool, Base]), "any")
                    stack.insert(0, node)
                    post_order.insert(0, node)
                    current_node.children.append(node)
            else:
                for u in (int, str, List, Dict, bool, Base):
                    node = CMPNode(current_node.n1, Node(u), "all")
                    #stack.insert(0, node)
                    post_order.insert(0, node)
                    current_node.children.append(node)

        print("POST ORDER", post_order)
        for cmp_node in post_order:
            if hasattr(cmp_node.n1.type_, "__args__"):
                n1_type = cmp_node.n1.type_.__origin__
            else:
                n1_type = cmp_node.n1.type_
            if hasattr(cmp_node.n2.type_, "__args__"):
                n2_type = cmp_node.n2.type_.__origin__
            else:
                n2_type = cmp_node.n2.type_

            print("CMP", n1_type, n2_type, cmp_node.children)
            if cmp_node.children:
                ch_eq = any([ch.eq for ch in cmp_node.children])
                print("ch_eq any", ch_eq, [ch.eq for ch in cmp_node.children])
            else:
                ch_eq = True


            # check types only of both types are not union
            # otherwise equality was already decided by check above
            if n1_type != Union and n2_type != Union:
                ch_eq &= n1_type == n2_type or issubclass(
                    n1_type, n2_type
                )
            cmp_node.eq = ch_eq
            print(ch_eq)

        return root_node.eq 


def type_tree(type_):
    root = Node(type_=type_)
    current = root
    stack = []
    result = []
    while True:
        if hasattr(current.type_, "__args__"):
            for arg in current.type_.__args__:
                n = Node(type_=arg)
                stack.append(n)
                current.children.append(n)
        else:
            result.append(current)
        if not stack:
            break
        current = stack.pop()
    return root


class BaseMeta(type):
    def __new__(cls, name, bases, attrs):
        if '_config' in attrs:
            assert check_type(type(attrs['_config']), BaseConfig)
            config = attrs['_config']
        else:
            config = BaseConfig()
            attrs["_config"] = config

        if config.validate_set_attr:
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
        for attr in attrs:
            if attr.startswith("_"):
                continue
            if attr not in annotations:
                raise TypeError(f"{attr} has to be annotated")

        for attr, type_ in annotations.items():
            if attr.startswith("_"):
                continue
            if not type_tree(annotations[attr]).json_compatible():
                raise TypeError(f"{attr} is not json compatible")

        attrs['_fields'] = attrs.get('__annotations__', {})
        annotations['_fields'] = ClassVar[Dict[str, Any]]

        ret = super().__new__(cls, name, bases, attrs)
        ret = dataclasses.dataclass(ret, kw_only=True)
        return ret


class Base(metaclass=BaseMeta):
    _config: ClassVar[BaseConfig] = BaseConfig()
    _fields: ClassVar[Dict[str, Any]]

    def _no_validate_setattr_(self, name: str, value: Any) -> None:
        return super().__setattr__(name, value)

    def _validate_setattr_(self, name: str, value: Any) -> None:
        if name not in self._fields and not self._config.allow_extra:
            raise AttributeError("{self.__class__} doesn't have attribute name")

        tt1 = type_tree(type(value))
        tt2 = type_tree(self._fields[name])
        if tt1 != tt2:
            raise TypeError(f"Cannot set attribute {self.__class__}.{name} to type {type(value)}, expected {self._fields[name]}")
        return super().__setattr__(name, value)


@dataclasses.dataclass
class Param:
    type_: Base


class In:
    def __class_getitem__(cls, param):
        return Param(type_=param)


#class IntIO(Base):
#    x: int = 0


class StepMeta(BaseMeta):
    def __new__(cls, name, bases, attrs):
        ret = super().__new__(cls, name, bases, attrs)
        return ret


class X(Base):
    pass


class F:
    pass


class Step(Base, metaclass=StepMeta):
    #x: X
    z: List[Dict[str, F]]
    #a: List[Dict[str, int]]







