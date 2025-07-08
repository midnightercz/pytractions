from __future__ import annotations
import abc
import datetime

from typing import get_origin, get_args

from pytractions.base_field import Field


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


def doc(docstring: str):
    """Create dataclass field for doctring fields."""
    return Field(init=False, repr=False, default=docstring)


def isodate_now() -> str:
    """Return current datetime in iso8601 format."""
    return "%s" % (datetime.datetime.utcnow().isoformat())


def _get_args(v):
    return get_args(v) or v.__targs__ if hasattr(v, "__targs__") else []


def _get_origin(v):
    return get_origin(v) or v.__torigin__ if hasattr(v, "__torigin__") else None
