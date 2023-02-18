import abc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import datetime
import inspect
import json

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
    TypeVar
)

import dataclasses


Base = ForwardRef("Base")


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
        root = cls(type_=type_)
        current = root
        stack = []
        while True:
            print("TN current", current.type_)
            if hasattr(current.type_, "__args__"):
                print("TN current args", current.type_.__args__)
                for arg in current.type_.__args__:
                    print("TN arg", arg)
                    n = cls(type_=arg)
                    stack.append(n)
                    current.children.append(n)
            if not stack:
                break
            current = stack.pop()
        print("root", root, root.children)
        return root

    def replace_params(self, params_map):
        stack = [(self, 0, None)]
        post_order = []
        while stack:
            current, parent_index, current_parent = stack.pop(0)
            for n, ch in enumerate(current.children):
                print("ch", ch)
                stack.insert(0, (ch, n, current))
            if type(current.type_) == TypeVar:
                if current.type_ in params_map:
                    current.type_ = params_map[current.type_]
            post_order.insert(0, (current, parent_index, current_parent))
        for item in post_order:
            node, parent_index, parent = item
            if not parent:
                continue
            parent.children[parent_index] = node

    def to_type(self):
        stack = [(self, 0, None)]
        post_order = []
        while stack:
            current, parent_index, current_parent = stack.pop(0)
            for n, ch in enumerate(current.children):
                stack.insert(0, (ch, n, current))
            post_order.insert(0, (current, parent_index, current_parent))

        for item in post_order:
            node, parent_index, parent = item
            if not parent:
                continue
            if node.children:
                print("node ch", tuple([x.type_ for x in node.children]))
                node.type_ = node.type_[tuple([x.type_ for x in node.children])]
            parent.children[parent_index] = node
        print("TYPE", post_order[-1][0].type_)
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
            print("current", current_node.op, current_node.n1, current_node.n2)
            if get_origin(current_node.n1.type_) == Union and get_origin(current_node.n2.type_) != Union:
                print("1")
                for ch1 in current_node.n1.children:
                    node = CMPNode(ch1, current_node.n2, "all", "all")
                    stack.insert(0, node)
                    post_order.insert(0, node)
                    current_node.children.append(node)

            elif get_origin(current_node.n1.type_) != Union and get_origin(current_node.n2.type_) == Union:
                print("2")
                print(current_node.n2.type_)
                print(current_node.n2.children)
                for ch2 in current_node.n2.children:
                    print("ch2", ch2)
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

        #print("--- EQ ---")
        for cmp_node in post_order:
            #print("cmp node", cmp_node.op, cmp_node.n1.type_, cmp_node.n2.type_)
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

        root_node = CMPNode(self, TypeNode(Union[int, str, List, Dict, bool, Base, type(None)]), op, op)
        stack = [root_node]
        post_order = []
        post_order.insert(0, root_node)
        while stack:
            current_node = stack.pop()
            if current_node.n1.children: # and current_node.n1.type_ != Union:
                for ch1 in current_node.n1.children:
                    if ch1.children: # and ch1.type_ != Union:
                        op = "all"
                    else:
                        op = "any"
                    node = CMPNode(ch1, TypeNode(Union[int, str, List, Dict, bool, Base, type(None), TypeVar("X")]), op, op)
                    stack.insert(0, node)
                    post_order.insert(0, node)
                    current_node.children.append(node)
            else:
                for u in (int, str, List, Dict, bool, Base, type(None), TypeVar("X")):
                    node = CMPNode(current_node.n1, TypeNode(u), "any", "any")
                    post_order.insert(0, node)
                    current_node.children.append(node)

        for cmp_node in post_order:
            n1_type = get_origin(cmp_node.n1.type_) or cmp_node.n1.type_
            n2_type = get_origin(cmp_node.n2.type_) or cmp_node.n2.type_
            if cmp_node.children:
                if cmp_node.ch_op == "any":
                    #print("any", [ch.eq for ch in cmp_node.children])
                    ch_eq = any([ch.eq for ch in cmp_node.children])
                else:
                    #print("all", [ch.eq for ch in cmp_node.children])
                    ch_eq = all([ch.eq for ch in cmp_node.children] or [True])
            else:
                ch_eq = True
            # check types only of both types are not union
            # otherwise equality was already decided by check above
            #print("get origin", n1_type, n2_type)
            print(n1_type, n2_type)

            if type(n1_type) == TypeVar and type(n2_type) == TypeVar:
                #print("TV", type(n1_type))
                ch_eq &= n1_type == n1_type
            elif type(n1_type) == TypeVar:
                #print("TV1", type(n1_type))
                if n2_type == Union:
                    ch_eq = True
                else:
                    ch_eq = False
            elif type(n2_type) == TypeVar:
                #print("TV2", type(n1_type))
                if n1_type == Union:
                    ch_eq = True
                else:
                    ch_eq = False
                ch_eq = False

            elif n1_type != Union and n2_type != Union:
                ch_eq &= n1_type == n2_type or issubclass(
                    n1_type, n2_type
                )
            cmp_node.eq = ch_eq

        return root_node.eq


class BaseMeta(type):
    def __new__(cls, name, bases, attrs):
        print("META NEW")

        # if config is provided, make sure it's subsclass of BaseConfig
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
        for attr in attrs:
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
        attrs['_fields'] = attrs.get('__annotations__', {})
        #annotations['_fields'] = ClassVar[Dict[str, Any]]

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

        tt1 = TypeNode.from_type(type(value))
        tt2 = TypeNode.from_type(self._fields[name])
        if tt1 != tt2:
            raise TypeError(f"Cannot set attribute {self.__class__}.{name} to type {type(value)}, expected {self._fields[name]}")
        return super().__setattr__(name, value)

    def __class_getitem__(cls, *params):
        ret = super().__class_getitem__(*params)
        ret._params_map = dict(zip(cls.__parameters__, params))

        print("-----", dir(ret), cls.__parameters__)
        print("F", ret._fields)
        for attr, type_ in ret._fields.items():
            print('getitem', attr, type_)
            tn = TypeNode.from_type(type_)
            tn.replace_params(ret._params_map)
            new_type = tn.to_type()
            if not tn.json_compatible():
                raise JSONIncompatibleError(f"Attribute  {attr}: {new_type} is not json compatible")
            ret._fields[attr] = new_type

        ret._params = params
        return ret

