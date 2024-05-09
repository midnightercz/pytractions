from __future__ import annotations
import abc

from typing import get_origin, get_args


class ANYMeta(abc.ABCMeta):
    """Metaclass for helper object that compares equal to everything."""

    def __eq__(mcs, other):
        """Return always True."""
        return True

    def __ne__(mcs, other):
        """Return always False."""
        return False

    def __repr__(mcs):
        """Any class string representation."""
        return "<ANY>"

    def __hash__(mcs):
        """Return id of the class."""
        return id(mcs)


class ANY(metaclass=ANYMeta):
    """A helper object that compares equal to everything."""

    def __eq__(cls, other):
        """Return True."""
        return True

    def __ne__(cls, other):
        """Return always False."""
        return False

    def __repr__(cls):
        """Return Any class string representation."""
        return "<ANY>"

    def __hash__(cls):
        """Return id of the class."""
        return id(cls)


# class OType:
#
#     origins: Any
#     args: List["OType" | Any]
#     parameters: List["OType" | Any]
#
#     def __init__(self):
#         self.origins = []
#         self.args = []
#         self.parameters = []
#
#     def __str__(self):
#         return f"OType({self.origins}, {self.args}, {self.parameters})"
#
#     def __eq__(self, other):
#         print("OType.__eq__")
#         return (
#             self.origins == other.origins
#             and self.args == other.args
#             and self.parameters == other.parameters
#         )
#
#     def __le__(self, other):
#         if self.origins in ([list], [List]) and other.origins in ([list], [List[ANY]]):
#             # TODO: fix list check
#             olte = True
#             return True
#         else:
#             olte = self.origins == other.origins or any(
#                 [issubclass(o1, o2) for o1, o2 in zip(self.origins, other.origins)]
#             )
#         alte = True
#         if len(self.args) != len(other.args):
#             alte = False
#         else:
#             for a1, a2 in zip(self.args, other.args):
#                 if isinstance(a1, OType) and isinstance(a2, OType):
#                     alte &= a1 <= a2
#                 elif not isinstance(a1, OType) and not isinstance(a2, OType):
#                     alte &= any([issubclass(o1, o2) for o1, o2 in
#                                  zip(self.origins, other.origins)])
#                 else:
#                     alte = False
#                     break
#         return olte and alte
#
#     def __check_generics__(self, other):
#         olte = self.origins == other.origins or any(
#             [issubclass(o1, o2) for o1, o2 in zip(self.origins, other.origins)]
#         )
#         alte = True
#         if len(self.args) != len(other.parameters):
#             alte = False
#         else:
#             alte = True
#         return olte and alte


def _get_args(v):
    return get_args(v) or v.__targs__ if hasattr(v, "__targs__") else []


def _get_origin(v):
    return get_origin(v) or v.__torigin__ if hasattr(v, "__torigin__") else None


# def get_type(var):
#     root = OType()
#     if _get_origin(var) == Union:
#         root.origins = _get_origin(get_args(var))
#     elif _get_origin(var):
#         root.origins = [_get_origin(var)]
#     else:
#         root.origins = [var]
#
#     root.parameters = getattr(var, "__parameters__", [])
#
#     to_process = []
#     if _get_args(var):
#         for arg in _get_args(var):
#             to_process.append((root, arg))
#
#     while to_process:
#         croot, arg = to_process.pop(0)
#         child = OType()
#         child.parameters = arg.__parameters__ if hasattr(arg, "__parameters__") else []
#         if _get_origin(arg) == Union:
#             child.origins = _get_args(_get_origin(arg))
#         elif _get_origin(arg):
#             child.origins = [_get_origin(arg)]
#         else:
#             child.origins = [arg]
#
#         if _get_args(arg):
#             for charg in get_args(arg):
#                 to_process.append((child, charg))
#         croot.args.append(child)
#
#     return root
#
#
# def check_type(to_check, expected):
#     t1 = get_type(to_check)
#     t2 = get_type(expected)
#     return t1 <= t2
#
#
# def check_type_generics(to_check, expected_generic):
#     return get_type(to_check).__check_generics__(get_type(expected_generic))
