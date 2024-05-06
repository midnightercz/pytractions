from __future__ import annotations
import abc

from typing import Any, Dict, List, get_origin, get_args, Union


class ANYMeta(abc.ABCMeta):
    "A helper object that compares equal to everything."

    def __eq__(mcs, other):
        return True

    def __ne__(mcs, other):
        return False

    def __repr__(mcs):
        return "<ANY>"

    def __hash__(mcs):
        return id(mcs)


class ANY(metaclass=ANYMeta):
    def __eq__(cls, other):
        return True

    def __ne__(cls, other):
        return False

    def __repr__(cls):
        return "<ANY>"

    def __hash__(cls):
        return id(mcs)


class OType:
    origins: Any
    args: List["OType" | Any]
    parameters: List["OType" | Any]

    def __init__(self):
        self.origins = []
        self.args = []
        self.parameters = []

    def __str__(self):
        return f"OType({self.origins}, {self.args}, {self.parameters})"

    def __eq__(self, other):
        print("OType.__eq__")
        return (
            self.origins == other.origins
            and self.args == other.args
            and self.parameters == other.parameters
        )

    def __le__(self, other):
        if self.origins in ([list], [List]) and other.origins in ([list], [List[ANY]]):
            # TODO: fix list check
            olte = True
            return True
        else:
            olte = self.origins == other.origins or any(
                [issubclass(o1, o2) for o1, o2 in zip(self.origins, other.origins)]
            )
        alte = True
        if len(self.args) != len(other.args):
            alte = False
        else:
            for (a1, a2) in zip(self.args, other.args):
                if isinstance(a1, OType) and isinstance(a2, OType):
                    alte &= a1 <= a2
                elif not isinstance(a1, OType) and not isinstance(a2, OType):
                    alte &= any([issubclass(o1, o2) for o1, o2 in zip(self.origins, other.origins)])
                else:
                    alte = False
                    break
        return olte and alte

    def __check_generics__(self, other):
        olte = self.origins == other.origins or any(
            [issubclass(o1, o2) for o1, o2 in zip(self.origins, other.origins)]
        )
        alte = True
        if len(self.args) != len(other.parameters):
            alte = False
        else:
            alte = True
        return olte and alte


def _get_args(v):
    return get_args(v) or v.__targs__ if hasattr(v, "__targs__") else []


def _get_origin(v):
    return get_origin(v) or v.__torigin__ if hasattr(v, "__torigin__") else None


def get_type(var):
    root = OType()
    if _get_origin(var) == Union:
        root.origins = _get_origin(get_args(var))
    elif _get_origin(var):
        root.origins = [_get_origin(var)]
    else:
        root.origins = [var]

    root.parameters = getattr(var, "__parameters__", [])

    to_process = []
    if _get_args(var):
        for arg in _get_args(var):
            to_process.append((root, arg))

    while to_process:
        croot, arg = to_process.pop(0)
        child = OType()
        child.parameters = arg.__parameters__ if hasattr(arg, "__parameters__") else []
        if _get_origin(arg) == Union:
            child.origins = _get_args(_get_origin(arg))
        elif _get_origin(arg):
            child.origins = [_get_origin(arg)]
        else:
            child.origins = [arg]

        if _get_args(arg):
            for charg in get_args(arg):
                to_process.append((child, charg))
        croot.args.append(child)

    return root


def check_type(to_check, expected):
    t1 = get_type(to_check)
    t2 = get_type(expected)
    return t1 <= t2


def check_type_generics(to_check, expected_generic):
    return get_type(to_check).__check_generics__(get_type(expected_generic))


class CMPNode:
    def __init__(self, n1, n2, ch_op):
        self.n1 = n1
        self.n2 = n2
        self.children = []
        self.op = ch_op
        self.eq = None

    def __str__(self):
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
            else:
                if current_node.n1.type_ == Union:
                    for ch in current_node.n1.children:
                        op = self.__determine_op(ch, current_node.n2.type_)
                        node = CMPNode(ch, current_node.n2, op)
                        stack.insert(0, node)
                        post_order.insert(0, node)
                else:
                    for ch in current_node.n2.children:
                        op = self.__determine_op(ch, current_node.n1.type_)
                        node = CMPNode(ch, current_node.n1, op)
                        stack.insert(0, node)
                        post_order.insert(0, node)
        return post_order

    def __eq__(self, other):
        print("Node __eq__")
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

            ch_eq &= cmp_node.n1.type_ == cmp_node.n2.type_ or issubclass(
                cmp_node.n1.type_, cmp_node.n2.type_
            )
            if not ch_eq:
                return False
        return True

    def json_compatible(self):
        node = CMPNode(self, Node(Union[int, str, List, Dict, bool, "Base"]), "any")
        stack = [node]
        post_order = []
        post_order.insert(0, node)
        while stack:
            current_node = stack.pop()
            for ch1 in current_node.n1.children:
                node = CMPNode(ch1, Node(Union[int, str, List, Dict, bool]), "any")
                stack.insert(0, node)
                post_order.insert(0, node)

        for cmp_node in post_order:
            if cmp_node.children:
                ch_eq = any([ch.eq for ch in cmp_node.children])
            else:
                ch_eq = True

            ch_eq &= cmp_node.n1.type_ == cmp_node.n2.type_ or issubclass(
                cmp_node.n1.type_, cmp_node.n2.type_
            )
            if not ch_eq:
                return False
        return True


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
