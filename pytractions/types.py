import inspect
import importlib
import json
from typing import get_origin, TypeVar, Union, ForwardRef, Dict, Any, Tuple, List
import sys


from .abase import ABase
from .utils import ANY

X = TypeVar("X")

JSON_COMPATIBLE = Union[int, float, str, bool, ABase, type(None), X]


def evaluate_forward_ref(ref, frame):
    """Evaluate ForwardRef."""
    caller_globals, caller_locals = frame.f_globals, frame.f_locals
    recursive_guard = set()
    return ref._evaluate(caller_globals, caller_locals, recursive_guard)


class CMPNode:
    """CMPNode class for TypeNode comparison."""

    def __init__(self, n1, n2, op, ch_op):
        """Initialize CMPNode."""
        self.n1 = n1
        self.n2 = n2
        self.children = []
        self.op = op
        self.eq = None
        self.ch_op = ch_op

    def __str__(self):
        """Return string representation of CMPNode."""
        return "<CMPNode n1=%s n2=%s op=%s eq=%s>" % (self.n1, self.n2, self.op, self.eq)

    def __repr__(self):
        """Return repr representation of CMPNode."""
        return "<CMPNode n1=%s n2=%s op=%s eq=%s>" % (self.n1, self.n2, self.op, self.eq)


class TypeNode:
    """TypeNode class for type comparison."""

    _json_cache = []
    _to_type_cache = {}

    def __str__(self):
        """Return string representation of TypeNode."""
        return "<TypeNode type=%s>" % self.type_

    def __init__(self, type_, subclass_check=True):
        """Initialize TypeNode."""
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
            if get_origin(current.type_):
                current.type_ = get_origin(current.type_)
            elif hasattr(current.type_, "_orig_cls") and current.type_._orig_cls:
                current.type_ = current.type_._orig_cls
            current = stack.pop()
        return root

    # def post_order(self):
    #     stack = [(self, 0, None)]
    #     post_order = []
    #     while stack:
    #         current, parent_index, current_parent = stack.pop(0)
    #         for n, ch in enumerate(current.children):
    #             stack.insert(0, (ch, n, current))
    #         post_order.insert(0, (current, parent_index, current_parent))
    #     return post_order

    def replace_params(self, params_map):
        """Replace Typevars in TypeNode structure with values from provided mapping."""
        stack = [(self, 0, None)]
        while stack:
            current, parent_index, current_parent = stack.pop(0)
            for n, ch in enumerate(current.children):
                stack.insert(0, (ch, n, current))
            if type(current.type_) is TypeVar:
                if current.type_ in params_map:
                    if hasattr(params_map[current.type_], "__args__"):
                        replaced_tn = TypeNode.from_type(params_map[current.type_])
                        current.type_ = replaced_tn.type_
                        current.children = replaced_tn.children

                    else:
                        current.type_ = params_map[current.type_]

    def to_type(self, types_cache={}, params_map={}):
        """Return new type for TypeNode or already existing type from cache."""
        if json.dumps(self.to_json(), sort_keys=True) in self._to_type_cache:
            return self._to_type_cache[json.dumps(self.to_json(), sort_keys=True)]

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
                type_ = node.type_

                # children_types = tuple([x.type_ for x in node.children])
                children_type_ids = tuple([id(x.type_) for x in node.children])

                if f"{type_.__qualname__}[{children_type_ids}]" in types_cache:
                    node.type_ = types_cache[f"{type_.__qualname__}[{children_type_ids}]"]
                else:
                    if type_ == Union:
                        node.type_ = Union[tuple([x.type_ for x in node.children])]
                    else:
                        new_type = type_.__class_getitem__(
                            tuple([x.type_ for x in node.children]), params_map=params_map
                        )
                        node.type_ = new_type

            if not parent:
                continue
            parent.children[parent_index] = node
        self._to_type_cache[json.dumps(self.to_json(), sort_keys=True)] = post_order[-1][0].type_
        return post_order[-1][0].type_

    @staticmethod
    def __determine_op(ch1, ch2) -> str:
        op = "all"
        if (ch1.type_ is Union and ch2.type_ is Union) or (
            ch1.type_ is not Union and ch2.type_ is not Union
        ):
            op = "all"
        elif (ch1.type_ is not Union and ch2.type_ is Union) or (
            ch1.type_ is Union and ch2.type_ is not Union
        ):
            op = "any"
        return op

    def __eq_post_order(self, root_node):
        stack = [root_node]
        post_order = []
        post_order.insert(0, root_node)
        while stack:
            current_node = stack.pop()

            if current_node.n1.type_ is Union and current_node.n2.type_ is not Union:
                for ch1 in current_node.n1.children:
                    node = CMPNode(ch1, current_node.n2, "all", "all")
                    stack.insert(0, node)
                    post_order.insert(0, node)
                    current_node.children.append(node)

            elif current_node.n1.type_ is not Union and current_node.n2.type_ is Union:
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
                if current_node.n1.type_ is Union:
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
        """Test if TypeNode is Equal to other TypeNode."""
        if type(other) is not TypeNode:
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

            if isinstance(n1_type, ForwardRef):
                frame = sys._getframe(1)
                n1_type = evaluate_forward_ref(n1_type, frame)
            if isinstance(n2_type, ForwardRef):
                frame = sys._getframe(1)
                n2_type = evaluate_forward_ref(n2_type, frame)
            if n2_type is None:
                n2_type = type(None)
            if n1_type is None:
                n1_type = type(None)

            # check types only of both types are not union
            # otherwise equality was already decided by check above

            orig_cls1 = getattr(n1_type, "_orig_cls", False)
            orig_cls2 = getattr(n2_type, "_orig_cls", True)

            has_params1 = hasattr(n1_type, "_params") and len(n1_type._params) != 0
            has_params2 = hasattr(n2_type, "_params") and len(n2_type._params) != 0

            if n1_type != Union and n2_type != Union:
                ch_eq &= (
                    n1_type == n2_type
                    or (orig_cls1 == orig_cls2 and has_params1 and has_params2)
                    or orig_cls1 == n2_type
                    or orig_cls2 == n1_type
                    or (
                        self.subclass_check
                        and (
                            inspect.isclass(n1_type)
                            and inspect.isclass(n2_type)
                            and issubclass(n1_type, n2_type)
                        )
                    )
                    or (
                        self.subclass_check
                        and (
                            inspect.isclass(n1_type)
                            and inspect.isclass(orig_cls2)
                            and issubclass(n1_type, orig_cls2)
                        )
                    )
                    or bool(
                        self.subclass_check
                        and inspect.isclass(orig_cls1)
                        and inspect.isclass(orig_cls2)
                        and issubclass(orig_cls1, orig_cls2)
                    )
                )
            cmp_node.eq = ch_eq

        return node.eq

    def __eq_cached_(self, other):
        if type(other) is not TypeNode:
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

            n1_type = cmp_node.n1.type_
            n2_type = cmp_node.n2.type_

            if isinstance(n1_type, ForwardRef) or isinstance(n2_type, ForwardRef):
                frame = sys._getframe(1)

            if isinstance(n1_type, ForwardRef):
                n1_type = evaluate_forward_ref(n1_type, frame)
            if isinstance(n2_type, ForwardRef):
                n2_type = evaluate_forward_ref(n2_type, frame)

            if n2_type is None:
                n2_type = type(None)
            if n1_type is None:
                n1_type = type(None)

            # check types only of both types are not union
            # otherwise equality was already decided by check above

            orig_cls1 = getattr(n1_type, "_orig_cls", False)
            orig_cls2 = getattr(n2_type, "_orig_cls", True)

            has_params1 = hasattr(n1_type, "_params") and len(n1_type._params) != 0
            has_params2 = hasattr(n2_type, "_params") and len(n2_type._params) != 0

            is_any1 = n1_type is ANY
            is_any2 = n2_type is ANY

            if is_any1 or is_any2:
                ch_eq &= is_any1 and is_any2
            elif n1_type is not Union and n2_type is not Union:
                ch_eq &= (
                    n1_type == n2_type
                    or (orig_cls1 == orig_cls2 and has_params1 and has_params2)
                    or orig_cls1 == n2_type
                    or orig_cls2 == n1_type
                    or (
                        self.subclass_check
                        and (
                            inspect.isclass(n1_type)
                            and inspect.isclass(n2_type)
                            and issubclass(n1_type, n2_type)
                        )
                    )
                    or (
                        self.subclass_check
                        and (
                            inspect.isclass(n1_type)
                            and inspect.isclass(orig_cls2)
                            and issubclass(n1_type, orig_cls2)
                        )
                    )
                    or bool(
                        self.subclass_check
                        and inspect.isclass(orig_cls1)
                        and inspect.isclass(orig_cls2)
                        and issubclass(orig_cls1, orig_cls2)
                    )
                )
            else:
                ch_eq = False
            cmp_node.eq = ch_eq

        return node.eq

    def json_compatible(self):
        """Check if type in TypeNode is json compatible."""
        if self.children:
            op = "all"
        else:
            op = "any"

        root_node = CMPNode(self, TypeNode(JSON_COMPATIBLE), op, op)
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
                    node = CMPNode(ch1, TypeNode(JSON_COMPATIBLE), op, op)
                    stack.insert(0, node)
                    post_order.insert(0, node)
                    current_node.children.append(node)
            else:
                for u in (int, float, str, bool, ABase, type(None), TypeVar("X")):
                    node = CMPNode(current_node.n1, TypeNode(u), "any", "any")
                    post_order.insert(0, node)
                    current_node.children.append(node)

        for cmp_node in post_order:
            n1_type = get_origin(cmp_node.n1.type_) or cmp_node.n1.type_
            n2_type = get_origin(cmp_node.n2.type_) or cmp_node.n2.type_
            if isinstance(n1_type, ForwardRef):
                frame = sys._getframe(1)
                n1_type = evaluate_forward_ref(n1_type, frame)

            if isinstance(n1_type, ForwardRef):
                frame = sys._getframe(1)
                n2_type = evaluate_forward_ref(n2_type, frame)

            if cmp_node.children:
                if cmp_node.ch_op == "any":
                    ch_eq = any([ch.eq for ch in cmp_node.children])
                else:
                    ch_eq = all([ch.eq for ch in cmp_node.children] or [True])
            else:
                ch_eq = True
            # check types only of both types are not union
            # otherwise equality was already decided by check above

            if type(n1_type) is TypeVar and type(n2_type) is TypeVar:
                ch_eq &= n1_type == n1_type
            elif type(n1_type) is TypeVar:
                if n2_type == Union:
                    ch_eq = True
                else:
                    ch_eq = False
            elif type(n2_type) is TypeVar:
                if n1_type == Union:
                    ch_eq = True
                else:
                    ch_eq = False
                ch_eq = False

            elif n1_type != Union and n2_type != Union:
                if not inspect.isclass(n1_type):
                    raise TypeError(f"{n1_type} is not class")
                if not inspect.isclass(n2_type):
                    raise TypeError(f"{n2_type} is not class")
                ch_eq &= issubclass(n1_type, n2_type)

            elif n1_type != Union and n2_type == Union:
                if not inspect.isclass(n1_type):
                    raise TypeError(f"{n1_type} is not class")
                ch_eq &= any(
                    [issubclass(n1_type, t) for t in [int, float, str, bool, type(None), ABase]]
                )
            cmp_node.eq = ch_eq

        return root_node.eq

    def to_json(self) -> Dict[str, Any]:
        """Dump TypeNode to json."""
        pre_order: Dict[str, Any] = {"root": {}}
        stack: List[Tuple[ABase, Dict[str, Any], str]] = [(self, pre_order, "root")]
        while stack:
            current, current_parent, parent_key = stack.pop(0)

            type_ = current.type_
            if isinstance(type_, ForwardRef):
                type_ = evaluate_forward_ref(type_, sys._getframe(1))

            if hasattr(type_, "__orig_qualname__"):
                type_name = type_.__orig_qualname__
                module = type_.__module__
            elif hasattr(type_, "__qualname__"):
                type_name = type_.__qualname__
                module = type_.__module__
            elif type_ is None:
                type_name = "NoneType"
                module = "types"
            else:
                type_name = type_.__name__
                module = type_.__module__
            current_parent[parent_key] = {
                "type": type_name,
                "args": [
                    None,
                ]
                * len(current.children),
                "module": module,
            }
            for n, arg in enumerate(current.children):
                stack.append((arg, current_parent[parent_key]["args"], n))

        self._json_cache.append((self.copy(subclass_check=False), pre_order["root"]))
        return pre_order["root"]

    @classmethod
    def from_json(cls, json_data, _locals={}) -> "TypeNode":
        """Load TypeNode from json."""
        root_parent = cls(None)
        root_parent.children = [None]

        stack: List[Tuple[TypeNode, str | int, str, str, Dict[str, Any]]] = [
            (root_parent, 0, json_data["type"], json_data["module"], json_data["args"])
        ]
        pre_order = []

        while stack:
            parent, parent_key, _type, _type_mod, _args = stack.pop(0)
            mod = importlib.import_module(_type_mod)
            _type_path = _type.split(".")
            _type_o = mod
            for path_part in _type_path:
                if path_part == "<locals>":
                    _type_o = _locals
                else:
                    if path_part != "NoneType":
                        if isinstance(_type_o, dict):
                            _type_o = _type_o[path_part]
                        else:
                            _type_o = getattr(_type_o, path_part)
                    else:
                        _type_o = None

            type_node = cls(_type_o)
            type_node.children = [None] * len(_args)
            pre_order.insert(0, (parent, parent_key, type_node))
            for n, arg in enumerate(_args):
                stack.insert(0, (type_node, n, arg["type"], arg["module"], arg["args"]))

        for po_entry in pre_order:
            parent, parent_key, type_node = po_entry
            parent.children[parent_key] = type_node

        return root_parent.children[0]

    def copy(self, subclass_check=True):
        """Return new copy of TypeNode with subclass_check override."""
        root_node = TypeNode(type_=self.type_, subclass_check=subclass_check)
        root_node.children = [None]
        stack = [(self, 0, root_node)]
        while stack:
            current, parent_index, current_parent = stack.pop(0)
            new_current = TypeNode(type_=current.type_, subclass_check=subclass_check)
            new_current.children = [None] * len(current.children)
            for n, ch in enumerate(current.children):
                stack.insert(0, (ch, n, new_current))
            current_parent.children[parent_index] = new_current

        return root_node.children[0]
