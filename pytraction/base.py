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
)

import dataclasses

from utils import (
    get_type, _get_args, check_type, type_tree
)


def find_attr(objects, attr_name):
    for o in objects:
        if hasattr(o, attr_name):
            return getattr(o, attr_name)


@dataclasses.dataclass
class BaseConfig:
    validate_set_attr: bool = True
    allow_extra: bool = False


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
                raise TypeError("{attr} has to be annotated")

        for attr, type_ in annotations.items():
            if attr.startswith("_"):
                continue
            print(attr, type_)

        attrs['_fields'] = attrs.get('__annotations__', {})
        attrs['__annotations__']['_fields'] = ClassVar[Dict[str, Any]]

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


class IntIO(Base):
    x: int = 0


class StepMeta(BaseMeta):
    def __new__(cls, name, bases, attrs):
        print(name, bases, attrs)
        ret = super().__new__(cls, name, bases, attrs)
        return ret


class Step(Base, metaclass=StepMeta):
    in1: In[IntIO]




