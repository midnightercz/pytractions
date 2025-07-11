import hashlib
import inspect
import json
import logging
from types import prepare_class, resolve_bases
import enum
import sys
import uuid

try:
    from typing_extensions import Self
except ImportError:
    from typing import Self

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
    Tuple,
    Type,
    _UnionGenericAlias,
    get_args,
    Literal,
)

import dataclasses

from .exc import (
    JSONIncompatibleError,
    NoAnnotationError,
    SerializationError,
    ItemSerializationError,
)
from .utils import ANY  # noqa: F401
from .types import (
    TypeNode,
    JSON_COMPATIBLE,
    _defaultInt,
    _defaultStr,
    _defaultFloat,
    _defaultBool,
    _defaultNone,
)
from .base_field import Field
from .abase import ABase, ATList, ATDict
from .traversal import Tree, ItemHandler
from .validators import LiteralValidator


LOGGER = logging.getLogger(__name__)

_Base = ForwardRef("Base")
_Traction = ForwardRef("Traction")


def find_attr(objects, attr_name):
    """Find attribute in list of objects."""
    for o in objects:
        if hasattr(o, attr_name):
            return getattr(o, attr_name)


@dataclasses.dataclass
class BaseConfig:
    """Base config model class."""

    validate_set_attr: bool = True
    allow_extra: bool = False
    observers: bool = True


@dataclasses.dataclass
class DefaultOut:
    """Class to provide default output model."""

    type_: JSON_COMPATIBLE
    params: List[JSON_COMPATIBLE]

    def copy(self, generic_cache):
        """Copy DefaultOut object."""
        return DefaultOut(
            type_=TypeNode.from_type(self.type_).to_type(types_cache=generic_cache),
            params=(TypeNode.from_type(self.params[0]).to_type(types_cache=generic_cache),),
        )

    def __call__(self):
        """Return default output value."""
        # handling Optional
        if (
            get_origin(self.type_) == Union
            and len(get_args(self.type_)) == 2
            and get_args(self.type_)[-1] is type(None)
        ):
            ret = Port[self.params]()
        else:
            ret = Port[self.params]()
            if isinstance(self.type_, TypeVar):
                ret.data = None
            else:
                ret.data = self.type_()

        return ret

    def replace_params(self, params_map, cache):
        """Replace generic parameters with real ones."""
        tn = TypeNode.from_type(self.type_)
        tn.replace_params(params_map)
        new_type = tn.to_type(types_cache=cache)
        self.type_ = new_type

        tn = TypeNode.from_type(self.params[0])
        tn.replace_params(params_map)
        new_type = tn.to_type(types_cache=cache)
        self.params = (new_type,)


def _hash(obj):
    return int(
        hashlib.sha256(json.dumps(obj.to_json(), sort_keys=True).encode("utf-8")).hexdigest(),
        base=16,
    )


def extract_from_optional(opt):
    """Extract argument type from Optional."""
    if get_origin(opt) == Union and len(get_args(opt)) == 2 and get_args(opt)[-1] is type(None):
        return get_args(opt)[0]
    return opt


type_to_default_type = {
    int: _defaultInt,
    str: _defaultStr,
    float: _defaultFloat,
    bool: _defaultBool,
    type(None): _defaultNone,
}


class BaseMeta(type):
    """Metaclass for Base class.

    Metaclass does following:
    - It replaces standard __setattr__ with __setattr__ with type validation (if set in config)
    - It forbids user to define public attributes which are not type of int, float, string,
      None, Base
    - It force all public properties to be annotated
    """

    def __repr__(cls):
        """Return string representation of the class."""
        qname = cls.__orig_qualname__ if hasattr(cls, "__orig_qualname__") else cls.__qualname__
        if cls._params:
            params = []
            for p in cls._params:
                if get_origin(p) is Union:
                    uparams = ",".join(
                        [
                            repr(up) if up is not type(None) else "NoneType"
                            for up in sorted(get_args(p), key=lambda x: repr(x))
                        ]
                    )
                    params.append(f"Union[{uparams}]")
                elif isinstance(p, cls):
                    params.append(repr(p))
                else:
                    params.append(p.__qualname__ if hasattr(p, "__qualname__") else p.__name__)
            params_str = ",".join(params)
            return f"{qname}[{params_str}]"
        else:
            return f"{qname}"

    @classmethod
    def _before_new(cls, name, attrs, bases):
        """Adjust constructor attributes before new class is created."""
        pass

    @classmethod
    def _attributes_preparation(cls, name, attrs, bases):
        """Prepare attributes for class creation."""
        return attrs

    @classmethod
    def _determine_config(cls, attrs):
        if "_config" in attrs:
            # assert TypeNode.from_type(type(attrs["_config"])) == TypeNode(BaseConfig)
            config = attrs["_config"]
        else:
            # if not, provide default config
            config = BaseConfig()
            attrs["_config"] = config
        return config

    @classmethod
    def _determine_setattr_(cls, attrs, config, bases):
        if config.validate_set_attr:
            # if setter validation is on, use _validate_setattr_
            # or find it in class bases
            if "_validate_setattr_" in attrs:
                _setattr = attrs["_validate_setattr_"]
            else:
                _setattr = find_attr(bases, "_validate_setattr_")
            attrs["__setattr__"] = _setattr
        else:
            if "_no_validate_setattr_" in attrs:
                _setattr = attrs["_no_validate_setattr_"]
            else:
                _setattr = find_attr(bases, "_no_validate_setattr_")
            attrs["__setattr__"] = attrs["_no_validate_setattr_"]

    @classmethod
    def _determine_observers(cls, attrs, config, bases):
        if config.observers:
            pass
        else:
            if "_no_update_observers" in attrs:
                update_observers = attrs["_no_update_observers"]
            else:
                update_observers = find_attr(bases, "_no_update_observers")
            attrs["_update_observers"] = update_observers
            if "_observerved" in attrs:
                observed = attrs["_no_observed"]
            else:
                observed = find_attr(bases, "_no_observed")
            attrs["_observerved"] = observed

    @classmethod
    def _replace_self(cls, ret):
        # replace Self
        for f, ftype in ret._fields.items():
            if ftype == Self:
                ret._fields[f] = ret
            else:
                tn = TypeNode.from_type(ftype)
                if tn.replace_params({Self: ret}):
                    ret._fields[f] = tn.to_type()

    @classmethod
    def _replace_literal(cls, ret):
        # replace Literal
        for f, ftype in ret._fields.items():
            if get_origin(ftype) == Literal:
                ret._fields[f] = type(get_args(ftype)[0])
                ret.__dataclass_fields__[f] = Field(
                    default=ret.__dataclass_fields__[f].default,
                    default_factory=ret.__dataclass_fields__[f].default_factory,
                    init=ret.__dataclass_fields__[f].init,
                    repr=ret.__dataclass_fields__[f].repr,
                    hash=ret.__dataclass_fields__[f].hash,
                    compare=ret.__dataclass_fields__[f].compare,
                    metadata=ret.__dataclass_fields__[f].metadata,
                    kw_only=ret.__dataclass_fields__[f].kw_only,
                    validator=LiteralValidator(get_args(ftype), f),
                )

    def __new__(cls, name, bases, attrs):
        """Create new class."""
        config = cls._determine_config(attrs)
        cls._determine_setattr_(attrs, config, bases)
        cls._determine_observers(attrs, config, bases)

        properties = []
        annotations = attrs.get("__annotations__", {})
        for attr, attrv in attrs.items():
            if isinstance(attrv, property):
                properties.append(attr)
            # skip annotation check for methods and functions
            if (
                inspect.ismethod(attrv)
                or inspect.isfunction(attrv)
                or isinstance(attrv, classmethod)
                or isinstance(attrv, property)
                or isinstance(attrv, staticmethod)
            ):
                continue
            # attr starting with _ is considered to be private, there no checks
            # are applied
            if attr.startswith("_"):
                continue
            # other attributes has to be annotated
            if attr not in annotations:
                raise NoAnnotationError(f"{attr} has to be annotated")

        defaults = {}
        attrs["_properties"] = properties

        # check if all attrs are in supported types
        for attr, type_ in annotations.items():
            # skip private attributes
            if attr.startswith("_"):
                continue
            # Check if type of attribute is json-compatible
            if not TypeNode.from_type(annotations[attr]).json_compatible():
                raise JSONIncompatibleError(
                    f"Attribute {attr} is not json compatible {annotations[attr]}"
                )
            if attr in attrs:
                if type(attrs[attr]) is Field:
                    default = dataclasses.MISSING
                    if attrs[attr].default is not dataclasses.MISSING:
                        default = attrs[attr].default
                    if attrs[attr].default_factory is not dataclasses.MISSING:
                        default = attrs[attr].default_factory
                elif type(attrs[attr]) in (str, int, None, float):
                    default = type_to_default_type[type(attrs[attr])](attrs[attr])
                    attrs[attr] = default

                elif TypeNode.from_type(attrs[attr]) == TypeNode.from_type(Optional[ABase]):
                    default = None
                elif TypeNode.from_type(attrs[attr].__class__) == TypeNode.from_type(ABase):
                    default = attrs[attr]
                else:
                    default = type(attrs[attr])

                if isinstance(default, DefaultOut):
                    default = type(default())

                defaults[attr] = default
                # TODO: fix
                # if default != dataclasses._MISSING_TYPE\
                #   and not isinstance(default, dataclasses._MISSING_TYPE)\
                #   and TypeNode.from_type(type_) != TypeNode.from_type(default):
                #    raise TypeError(f"Annotation for {attr} is {type_} but default is {default}")

        # record fields to private attribute
        attrs["_attrs"] = attrs
        attrs["_child_classes"] = []
        fields = {}
        all_annotations = {}
        for base in bases:
            for f, ft in getattr(base, "_fields", {}).items():
                fields[f] = ft

        for base in bases:
            for f, ft in fields.items():
                if hasattr(base, "__dataclass_fields__") and f in base.__dataclass_fields__:
                    # Skip if default is already set
                    if f in defaults:
                        continue
                    # Skip if parameter is defined again in child class but has no default,
                    # in that case we don't want to use default from base
                    if f in attrs.get("__annotations__", {}) and f not in defaults:
                        continue

                    if base.__dataclass_fields__[f].default is not dataclasses.MISSING:
                        defaults[f] = Field(
                            default=base.__dataclass_fields__[f].default,
                            init=base.__dataclass_fields__[f].init,
                            repr=base.__dataclass_fields__[f].repr,
                        )
                    elif base.__dataclass_fields__[f].default_factory is not dataclasses.MISSING:
                        defaults[f] = Field(
                            default_factory=base.__dataclass_fields__[f].default_factory,
                            init=base.__dataclass_fields__[f].init,
                            repr=base.__dataclass_fields__[f].repr,
                        )

            for f, ft in getattr(base, "__annotations__", {}).items():
                if f in fields:
                    all_annotations[f] = ft

        fields.update(
            {k: v for k, v in attrs.get("__annotations__", {}).items() if not k.startswith("_")}
        )

        # Override fields if they are set in child class
        for f, ft in attrs.get("_fields", {}).items():
            fields[f] = ft

        all_annotations.update(annotations)
        attrs["__annotations__"] = all_annotations

        for default, defval in defaults.items():
            if default not in attrs:
                attrs[default] = defval

        attrs["_fields"] = fields
        attrs["_field_type_nodes"] = {}
        attrs["__hash__"] = _hash
        attrs["__name__"] = name

        cls._before_new(name, attrs, bases)
        ret = super().__new__(cls, name, bases, attrs)

        cls._replace_self(ret)
        # construct field type nodes to save time in setattr

        # wrap with dataclass
        ret = dataclasses.dataclass(ret, kw_only=attrs.get("_KW_ONLY", True))

        cls._replace_literal(ret)

        for f, ft in ret._fields.items():
            ret._field_type_nodes[f] = TypeNode.from_type(ft)

        ret._type_node = TypeNode.from_type(ret)
        for base in bases:
            if not hasattr(base, "_child_classes"):
                continue
            if ret not in base._child_classes:
                base._child_classes.append(ret)

        return ret


def is_wrapped(objcls):
    """Determine if objcls is Arg, In, Out or Res Wrapper."""
    tt1 = TypeNode.from_type(objcls, subclass_check=True)
    if (
        tt1 == TypeNode.from_type(Port[ANY])
        or tt1 == TypeNode.from_type(STMDSingleIn[ANY])
        or tt1 == TypeNode.from_type(NullPort[ANY])
    ):
        return True
    return False


class ListItemHandler(ItemHandler):
    """Serialization handler for list items."""

    def match(self, item):
        """Match list items."""
        if TypeNode.from_type(item.data_type) == TypeNode.from_type(ATList) or isinstance(
            item.data, TList
        ):
            return True

    def process(self, tree, item):
        """Process list items."""
        item.result[item.parent_index] = {
            "$type": TypeNode.from_type(item.data.__class__).to_json(),
            "$data": [],
        }
        if not item.data:
            return
        for n, litem in enumerate(item.data._list):
            if isinstance(litem, (int, float, str, bool, type(None))):
                item.result[item.parent_index]["$data"].append(litem)
            else:
                item.result[item.parent_index]["$data"].append(None)
                tree.add_to_process(
                    data=litem,
                    data_type=item.data._params[0],
                    parent_index=n,
                    result=item.result[item.parent_index]["$data"],
                )


class ListItemHandlerSchema(ItemHandler):
    """Serialization handler for list items."""

    def match(self, item):
        """Match list items."""
        if TypeNode.from_type(item.data_type) == TypeNode.from_type(ATList):
            return True

    def process(self, tree, item):
        """Process list items."""
        item.result[item.parent_index] = {"title": str(item.data), "type": "array", "items": {}}
        if item.extra:
            item.result[item.parent_index].update(item.extra)
        tree.add_to_process(
            data="list item",
            data_type=item.data_type._params[0],
            parent_index="items",
            result=item.result[item.parent_index],
        )


class ListItemHandlerContent(ItemHandler):
    """Content serialization handler for list items."""

    def match(self, item):
        """Match list items."""
        if TypeNode.from_type(item.data_type) == TypeNode.from_type(ATList) or isinstance(
            item.data, TList
        ):
            return True

    def process(self, tree, item):
        """Process list items."""
        item.result[item.parent_index] = []
        if not item.data:
            return
        for n, litem in enumerate(item.data._list):
            if isinstance(litem, (int, float, str, bool, type(None))):
                item.result[item.parent_index].append(litem)
            else:
                item.result[item.parent_index].append(None)
                tree.add_to_process(
                    data=litem,
                    data_type=item.data._params[0],
                    parent_index=n,
                    result=item.result[item.parent_index],
                )


class DictItemHandler(ItemHandler):
    """Serialization handler for dict items."""

    def match(self, item):
        """Match dict items."""
        if TypeNode.from_type(item.data_type) == TypeNode.from_type(ATDict) or isinstance(
            item.data, TDict
        ):
            return True

    def process(self, tree, item):
        """Process dict items."""
        item.result[item.parent_index] = {
            "$type": TypeNode.from_type(item.data.__class__).to_json(),
            "$data": {},
        }
        for k, v in item.data._dict.items():
            jsonk = k
            if not isinstance(k, (str, int, float, bool, type(None))):
                jsonk = json.dumps(k.to_json(), sort_keys=True)
            item.result[item.parent_index]["$data"][jsonk] = None
            tree.add_to_process(
                data=v,
                data_type=item.data._params[1],
                parent_index=jsonk,
                result=item.result[item.parent_index]["$data"],
            )


class DictItemHandlerSchema(ItemHandler):
    """Serialization handler for dict items."""

    def match(self, item):
        """Match dict items."""
        if TypeNode.from_type(item.data_type) == TypeNode.from_type(ATDict):
            return True

    def process(self, tree, item):
        """Process dict items."""
        item.result[item.parent_index] = {
            "title": str(item.data),
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "value": {}},
            },
        }
        if item.extra:
            item.result[item.parent_index].update(item.extra)
        tree.add_to_process(
            data=None,
            data_type=item.data_type._params[1],
            parent_index="value",
            result=item.result[item.parent_index]["items"]["properties"],
        )


class DictItemHandlerContent(ItemHandler):
    """Content serialization handler for dict items."""

    def match(self, item):
        """Match dict items."""
        if TypeNode.from_type(item.data_type) == TypeNode.from_type(ATDict) or isinstance(
            item.data, TDict
        ):
            return True

    def process(self, tree, item):
        """Process dict items."""
        item.result[item.parent_index] = {}
        for k, v in item.data._dict.items():
            jsonk = k
            if not isinstance(k, (str, int, float, bool, type(None))):
                jsonk = json.dumps(k.to_json(), sort_keys=True)
            item.result[item.parent_index][jsonk] = None
            tree.add_to_process(
                data=v,
                data_type=item.data._params[1],
                parent_index=jsonk,
                result=item.result[item.parent_index],
            )


class BasicItemHandler(ItemHandler):
    """Serialization handler for primitive types items."""

    def match(self, item):
        """Match primitive types items."""
        data_type = item.data_type
        if item.data_type.__class__ == _UnionGenericAlias and item.data_type._name == "Optional":
            data_type = item.data_type.__args__[0]
        uni = Union[
            int, str, bool, float, type(None), _defaultNone, _defaultInt, _defaultStr, _defaultFloat
        ]

        ret1 = TypeNode.from_type(uni) == TypeNode.from_type(data_type)
        ret2 = TypeNode.from_type(uni) == TypeNode.from_type(type(item.data))
        return ret1 or ret2

    def process(self, tree, item):
        """Process primitive types items."""
        if isinstance(
            item.data, (_defaultInt, _defaultStr, _defaultFloat, _defaultBool, _defaultNone)
        ):
            item.result[item.parent_index] = item.data._val
        else:
            item.result[item.parent_index] = item.data


class BasicItemHandlerSchema(ItemHandler):
    """Serialization handler for primitive types items."""

    def match(self, item):
        """Match primitive types items."""
        data_type = item.data_type
        if item.data_type.__class__ == _UnionGenericAlias and item.data_type._name == "Optional":
            data_type = item.data_type.__args__[0]
        uni = Union[
            int, str, bool, float, type(None), _defaultNone, _defaultInt, _defaultStr, _defaultFloat
        ]

        return TypeNode.from_type(uni) == TypeNode.from_type(data_type)

    @staticmethod
    def _to_jschema_type(type_):
        """Convert type to JSON schema type."""
        if type_ == int:
            return "integer"
        elif type_ == str:
            return "string"
        elif type_ == bool:
            return "boolean"
        elif type_ == float:
            return "number"
        elif type_ is type(None):
            return "null"

    def process(self, tree, item):
        """Process primitive types items."""
        item.result[item.parent_index] = {
            "title": str(item.data),
            "type": self._to_jschema_type(item.data_type),
        }
        if item.extra:
            item.result[item.parent_index].update(item.extra)


class EnumItemHandler(ItemHandler):
    """Serialization handler for enum items."""

    def match(self, item):
        """Match enum items."""
        if TypeNode.from_type(item.data_type) == TypeNode.from_type(enum.Enum):
            return True

    def process(self, tree, item):
        """Process enum items."""
        item.result[item.parent_index] = item.data.value


class EnumItemHandlerSchema(ItemHandler):
    """Serialization handler for enum items."""

    def match(self, item):
        """Match enum items."""
        if TypeNode.from_type(item.data_type) == TypeNode.from_type(enum.Enum):
            return True

    def process(self, tree, item):
        """Process enum items."""
        item.result[item.parent_index] = "string"


class PortItemHandler(ItemHandler):
    """Serialization handler for Port items."""

    def match(self, item):
        """Match Port items."""
        if TypeNode.from_type(item.data_type) == TypeNode.from_type(Port[ANY]):
            return True

    def process(self, tree, item):
        """Process Port items."""
        item.result[item.parent_index] = {
            "$type": TypeNode.from_type(item.data_type).to_json(),
            "$data": {},
        }
        for f in item.data_type._fields:
            if isinstance(item.data, (_defaultInt, _defaultStr, _defaultFloat, _defaultBool)):
                item.result[item.parent_index]["$data"][f] = item.data._val
            elif isinstance(item.data, (int, float, str, bool, type(None))):
                item.result[item.parent_index]["$data"][f] = item.data
            else:
                if not is_wrapped(type(item.data)):
                    tree.add_to_process(
                        data=item.data,
                        data_type=type(item.data),
                        parent_index=f,
                        result=item.result[item.parent_index]["$data"],
                    )
                else:
                    tree.add_to_process(
                        data=getattr(item.data, f),
                        data_type=item.data._fields[f],
                        parent_index=f,
                        result=item.result[item.parent_index]["$data"],
                    )


class PortItemHandlerSchema(ItemHandler):
    """Serialization handler for Port items."""

    def match(self, item):
        """Match Port items."""
        if TypeNode.from_type(item.data_type) == TypeNode.from_type(Port[ANY]):
            return True

    def process(self, tree, item):
        """Process Port items."""
        if isinstance(item.extra.get("default"), Port):
            item.extra["default"] = (
                item.extra["default"].data
                if not hasattr(item.extra["default"].data, "content_to_json")
                else item.extra["default"].data.content_to_json()
            )

        if (
            item.data_type._params[0].__class__ == _UnionGenericAlias
            and item.data_type._params[0]._name == "Optional"
        ):
            # print("OPTINONAL PORT", item.parent.result[item.parent.parent_index])
            item.parent.result[item.parent.parent_index]["required"].remove(item.parent_index)

        tree.add_to_process(
            data=item.data,
            data_type=item.data_type._params[0],
            parent_index=item.parent_index,
            result=item.result,
            extra=item.extra,
        )


class PortItemHandlerContent(ItemHandler):
    """Content serialization handler for Port items."""

    def match(self, item):
        """Match Port items."""
        if TypeNode.from_type(item.data_type) == TypeNode.from_type(Port[ANY]):
            return True

    def process(self, tree, item):
        """Process Port items."""
        item.result[item.parent_index] = {}
        for f in item.data_type._fields:
            if isinstance(item.data, (_defaultInt, _defaultStr, _defaultFloat, _defaultBool)):
                item.result[item.parent_index][f] = item.data._val
            elif isinstance(item.data, (int, float, str, bool, type(None))):
                item.result[item.parent_index][f] = item.data
            elif TypeNode.from_type(type(item.data)) == TypeNode.from_type(Port[ANY]):
                # data is wrapped in Port[]
                tree.add_to_process(
                    data=getattr(item.data, f),
                    data_type=item.data._fields[f],
                    parent_index=f,
                    result=item.result[item.parent_index],
                )
            else:
                # data is not wrapped in Port[]
                tree.add_to_process(
                    data=item.data,
                    data_type=item.data,
                    parent_index=f,
                    result=item.result[item.parent_index],
                )


class BaseItemHandler(ItemHandler):
    """Serialization handler for base subclasses."""

    def match(self, item):
        """Match Base subclasses."""
        if TypeNode.from_type(Union[int, str, bool, float, type(None)]) != TypeNode.from_type(
            item.data_type
        ) or TypeNode.from_type(Union[int, str, bool, float, type(None)]) != TypeNode.from_type(
            type(item.data)
        ):
            return True

    def process(self, tree, item):
        """Process Base subclasses."""
        if item.data._CUSTOM_TO_JSON:
            item.result[item.parent_index] = item.data.to_json()
            return

        item.result[item.parent_index] = {
            "$type": TypeNode.from_type(item.data.__class__).to_json(),
            "$data": {},
        }
        for f in item.data._fields:
            _f = item.data._SERIALIZE_REPLACE_FIELDS.get(f, f)
            data = getattr(item.data, f)
            tree.add_to_process(
                data=data,
                data_type=item.data._fields[f],
                parent_index=_f,
                result=item.result[item.parent_index]["$data"],
            )


class BaseItemHandlerSchema(ItemHandler):
    """Serialization handler for base subclasses."""

    def match(self, item):
        """Match Base subclasses."""
        if TypeNode.from_type(Union[int, str, bool, float, type(None)]) != TypeNode.from_type(
            item.data_type
        ):
            return True

    def process(self, tree, item):
        """Process Base subclasses."""
        if hasattr(item.data, "_to_json_schema"):
            item.result[item.parent_index] = item.data._to_json_schema()
            return

        required = []
        item.result[item.parent_index] = {
            "type": "object",
            "title": str(item.data),
            "properties": {},
        }
        if item.extra:
            item.result[item.parent_index].update(item.extra)

        for f, ftype in item.data_type._fields.items():
            if (
                item.data_type.__dataclass_fields__.get(f)
                and item.data_type.__dataclass_fields__[f].init is False
            ):
                # Skip if field is not initialized
                continue
            default = item.data_type.__dataclass_fields__.get(f, {}).default
            extra = {}
            if not isinstance(default, dataclasses._MISSING_TYPE):
                extra["default"] = (
                    default
                    if not hasattr(default, "content_to_json")
                    else default.content_to_json()
                )
            _f = item.data_type._SERIALIZE_REPLACE_FIELDS.get(f, f)
            datacls_field = item.data_type.__dataclass_fields__.get(f, {})
            if hasattr(datacls_field, "validator") and isinstance(
                datacls_field.validator, LiteralValidator
            ):
                # If field is Literal, use its value as type
                literal = item.data_type.__dataclass_fields__[f].validator.literal
                item.result[item.parent_index]["properties"][_f] = {
                    "type": "string",
                    "title": _f,
                    "enum": literal,
                    "default": literal[0],
                }
                continue
            if item.data_type._fields.get("d_" + f):
                extra["description"] = item.data_type.__dataclass_fields__["d_" + f].default
            if not (ftype.__class__ == _UnionGenericAlias and ftype._name == "Optional"):
                required.append(_f)

            tree.add_to_process(
                data=f,
                data_type=ftype,
                parent_index=_f,
                result=item.result[item.parent_index]["properties"],
                extra=extra,
            )
        item.result[item.parent_index]["required"] = required


class BaseItemHandlerContent(ItemHandler):
    """Content serialization handler for base subclasses."""

    def match(self, item):
        """Match Base subclasses."""
        if TypeNode.from_type(Union[int, str, bool, float, type(None)]) != TypeNode.from_type(
            item.data_type
        ):
            return True

    def process(self, tree, item):
        """Process Base subclasses."""
        item.result[item.parent_index] = {}
        for f in item.data._fields:
            _f = item.data._SERIALIZE_REPLACE_FIELDS.get(f, f)
            data = getattr(item.data, f)
            tree.add_to_process(
                data=data,
                data_type=item.data._fields[f],
                parent_index=_f,
                result=item.result[item.parent_index],
            )


class UnionItemHandlerSchema(ItemHandler):
    """Serialization handler for primitive types items."""

    def match(self, item):
        """Match primitive types items."""
        if item.data_type.__class__ == _UnionGenericAlias:
            return True

    def process(self, tree, item):
        """Process primitive types items."""
        if item.data_type._name != "Optional":
            item.result[item.parent_index] = {
                "title": str(item.data),
                "oneOf": [None] * len(item.data_type.__args__),
                "type": "object",
            }
            if item.extra:
                item.result[item.parent_index].update(item.extra)
            for n, atype in enumerate(item.data_type.__args__):
                tree.add_to_process(
                    data=str(atype),
                    data_type=atype,
                    parent_index=n,
                    result=item.result[item.parent_index]["oneOf"],
                    extra=item.extra,
                )
        else:
            item.result[item.parent_index] = {}
            if item.extra:
                item.result[item.parent_index].update(item.extra)
            tree.add_to_process(
                data=str(item.data),
                data_type=item.data_type.__args__[0],
                parent_index=item.parent_index,
                result=item.result,
                extra=item.extra,
            )


class ToJsonTree(Tree):
    """Tree for serialization."""

    handlers = [
        ListItemHandler(),
        DictItemHandler(),
        EnumItemHandler(),
        BasicItemHandler(),
        PortItemHandler(),
        BaseItemHandler(),
    ]


class ToJsonSchemaTree(Tree):
    """Tree for serialization."""

    handlers = [
        UnionItemHandlerSchema(),
        ListItemHandlerSchema(),
        DictItemHandlerSchema(),
        EnumItemHandlerSchema(),
        BasicItemHandlerSchema(),
        PortItemHandlerSchema(),
        BaseItemHandlerSchema(),
    ]


class BaseObserver:
    """Base observer class."""

    def __init__(self):
        """Initialize observer."""
        self._observers = {}

    def _observed(self, attr, value, extra=None):
        """Observed method."""
        if not hasattr(self, "_observers"):
            return
        for _, (_observer, path) in list(self._observers.items()):
            if path and attr:
                _observer._observed(path + "." + str(attr), value, extra=extra)
            elif path:
                _observer._observed(path, value, extra=extra)
            else:
                _observer._observed(attr, value, extra=extra)


class ContentToJsonTree(Tree):
    """Tree for content serialization."""

    handlers = [
        ListItemHandlerContent(),
        DictItemHandlerContent(),
        EnumItemHandler(),
        BasicItemHandler(),
        PortItemHandlerContent(),
        BaseItemHandlerContent(),
    ]


class NoAttr:
    """Special class indicating class doesn't have this attribute set."""

    pass


class Base(ABase, metaclass=BaseMeta):
    """Base class supporting type validation."""

    _CUSTOM_TYPE_TO_JSON: ClassVar[bool] = Field(default=False, init=False)
    _CUSTOM_TO_JSON: ClassVar[bool] = Field(default=False, init=False)
    _SERIALIZE_REPLACE_FIELDS: ClassVar[dict] = {}

    # dataclasses configuration class
    _config: ClassVar[BaseConfig] = BaseConfig()
    # mapping of class fields
    _fields: ClassVar[Dict[str, Any]] = {}
    # mapping used as lookup dict when creating generic subclasses
    _generic_cache: ClassVar[Dict[str, Type[Any]]] = {}
    # use to store actual parameters when creating generic subclass
    _params: ClassVar[List[Any]] = []
    # used to store original class when creating generic subclass
    _orig_cls: Optional[Type[Any]] = Field(default=None, init=False)

    _observer: BaseObserver = Field(
        default_factory=BaseObserver, init=False, repr=False, compare=False
    )

    _KW_ONLY: ClassVar[bool] = True

    def _extra_observed_events(self, attr, value, extra):
        """Extra observed events."""
        return extra

    def _no_observed(self, attr, value):
        pass

    def _observed(self, attr, value):
        if not hasattr(self, "_observer"):
            return
        self._observer._observed(attr, value, extra=self._extra_observed_events(attr, value, {}))

    def _no_validate_setattr_(self, name: str, value: Any) -> None:
        """Set attribute without any type validation."""
        return super().__setattr__(name, value)

    def _no_update_observers(self, attr, value, path_attr=None):
        pass

    def _update_observers(self, attr, value, path_attr=None):
        if not hasattr(self, "_observer"):
            return

        existing = getattr(self, attr, NoAttr)
        if isinstance(existing, Base):
            if hasattr(existing, "_observer"):
                if id(self._observer) in existing._observer._observers:
                    del existing._observer._observers[id(self._observer)]

        if isinstance(value, Base):
            if not hasattr(value, "_observer"):
                return
            if id(self._observer) not in value._observer._observers:
                value._observer._observers[id(self._observer)] = (
                    self._observer,
                    path_attr if path_attr is not None else attr,
                )

    def _validate_setattr_(self, name: str, value: Any) -> None:
        """Set attribute with type validation."""
        if not name.startswith("_"):  # do not check for private attrs

            if (
                name not in self._fields
                and not self._config.allow_extra
                and name not in self._properties
            ):
                raise AttributeError(f"{self.__class__} doesn't have attribute {name}")

            if name not in self._properties:
                vtype = (
                    value.__orig_class__ if hasattr(value, "__orig_class__") else value.__class__
                )
                tt1 = (
                    vtype._type_node if hasattr(vtype, "_type_node") else TypeNode.from_type(vtype)
                )
                # tt1 = TypeNode.from_type(vtype)
                tt2 = self._field_type_nodes[name]
                if tt1 != tt2:
                    raise TypeError(
                        f"Cannot set attribute {self.__class__}.{name} to {value}({vtype}), "
                        f"expected {self._fields[name]}"
                    )
                if (
                    hasattr(self.__dataclass_fields__[name], "validator")
                    and self.__dataclass_fields__[name].validator
                ):
                    self.__dataclass_fields__[name].validator(value)

            else:
                getattr(self.__class__, name).setter(value)
                return

        if not name.startswith("_"):  # do not observe private attributes
            self._update_observers(name, value)
            self._observed(name, value)

        return super().__setattr__(name, value)

    @staticmethod
    def _make_qualname(cls, params):
        """Return qualname of the class."""
        if params:
            stack = [(cls.__qualname__, params)]
        elif cls._params:
            stack = [(cls.__qualname__, cls._params)]
        else:
            stack = [(cls.__qualname__, [])]

        order = []
        while stack:
            current_qname, current_params = stack.pop(0)
            order.append(current_qname)
            if current_params:
                stack.insert(0, ("]", []))
                order.append("[")
            for p in current_params:
                if get_origin(p) is Optional:
                    stack.insert(0, ("Optional", get_args(p)))
                if get_origin(p) is Union:
                    stack.insert(0, ("Union", sorted(get_args(p), key=lambda x: repr(x))))
                elif hasattr(p, "_params"):
                    if hasattr(p, "__orig_qualname__"):
                        stack.insert(0, (p.__orig_qualname__, p._params))
                    elif hasattr(p, "__qualname__"):
                        stack.insert(0, (p.__qualname__, p._params))
                    else:
                        stack.insert(0, (p.__name__, p._params))
                    if p != current_params[-1]:
                        stack.insert(0, (",", []))

                elif hasattr(p, "__orig_qualname__"):
                    order.append(p.__qualname__)
                    if p != current_params[-1]:
                        order.append(",")
                elif hasattr(p, "__qualname__"):
                    order.append(p.__qualname__)
                    if p != current_params[-1]:
                        order.append(",")
                elif get_origin(p) is ForwardRef:
                    order.append(p.__forward_arg__)
                    if p != current_params[-1]:
                        order.append(",")
                elif isinstance(p, TypeVar):
                    # for typevar we need to add id to qualname,
                    # otherwise typevar replacement won't work
                    # as class_getitem could return cached version with typevar with different id
                    order.append(f"{p.__name__}[{id(p)}]")
                    if p != current_params[-1]:
                        order.append(",")
        return "".join(order)

    def __class_getitem__(cls, param, params_map={}):
        """Create subclass with generic params."""
        _params = param if isinstance(param, tuple) else (param,)
        if len(_params) != len(cls.__parameters__):
            raise TypeError(f"Expected {len(cls.__parameters__)} parameters, got {len(_params)}")

        # param ids for caching as TypeVars are class instances
        # therefore has to compare with id() to get good param replacement
        _param_ids = tuple([id(p) for p in param]) if isinstance(param, tuple) else (id(param),)

        # if there's already existing class, return it instead
        if f"{id(cls)}[{_param_ids}]" in cls._generic_cache:
            ret = cls._generic_cache[f"{id(cls)}[{_param_ids}]"]
            return ret

        bases = [x for x in resolve_bases([cls] + list(cls.__bases__)) if x is not Generic]
        attrs = {k: v for k, v in cls._attrs.items() if k not in ("_attrs", "_fields")}

        meta, ns, kwds = prepare_class(f"{cls.__name__}[{param}]", bases, attrs)

        _params_map = params_map.copy()
        _params_map.update(dict(zip(cls.__parameters__, _params)))

        # Fields needs to be copied to new subclass, otherwise
        # it's stays shared with base class
        for attr, type_ in cls._fields.items():
            tn = TypeNode.from_type(type_)

            # field params needs to be replaced as field can also reffer to TypeVar
            tn.replace_params(_params_map)
            new_type = tn.to_type(types_cache=cls._generic_cache, params_map=_params_map)

            if not tn.json_compatible():
                raise JSONIncompatibleError(f"Attribute  {attr}: {new_type} is not json compatible")
            # if attr not in kwds:
            #    kwds[attr] = new_type
            kwds["__annotations__"][attr] = new_type

        for k, kf in kwds.items():
            if not isinstance(kf, Field):
                continue
            new_kf = Field(
                default=kf.default,
                default_factory=kf.default_factory,
                init=kf.init,
                repr=kf.repr,
                hash=kf.hash,
                compare=kf.compare,
                metadata=kf.metadata,
                kw_only=kf.kw_only,
            )

            if hasattr(new_kf.default_factory, "replace_params"):
                new_default_factory = new_kf.default_factory.copy(generic_cache=cls._generic_cache)
                new_default_factory.replace_params(_params_map, cls._generic_cache)
                new_kf.default_factory = new_default_factory
            else:
                if new_kf.default_factory is not dataclasses.MISSING:
                    new_default_factory = TypeNode.from_type(new_kf.default_factory)
                    new_default_factory.replace_params(_params_map)
                    new_default_factory = new_default_factory.to_type(
                        types_cache=cls._generic_cache
                    )
                    new_kf.default_factory = new_default_factory

            kwds[k] = new_kf

        kwds["_params"] = _params
        if cls._orig_cls:
            kwds["_orig_cls"] = cls._orig_cls
        else:
            kwds["_orig_cls"] = cls

        kwds["__orig_qualname__"] = kwds.get("__orig_qualname__", kwds["__qualname__"])
        kwds["__qualname__"] = cls._make_qualname(cls, _params)

        ret = meta(kwds["__qualname__"], tuple(bases), kwds)

        sys.modules[ret.__module__].__dict__[ret.__qualname__] = ret

        cls._generic_cache[f"{id(cls)}[{_param_ids}]"] = ret
        return ret

    def to_json(self) -> Dict[str, Any]:
        """Return json representation of Base object.

        Function is written to dump a base object
        to json from which can be considered as serialized object, therefore it's possible to
        use this representation to load the very same object.
        However compare to python serializer, performance of much slower.
        """
        result = {}
        tree = ToJsonTree(result)
        tree.add_to_process(data=self, data_type=self.__class__, parent_index="root", result=result)
        tree.process()
        return result["root"]

    @classmethod
    def to_json_schema(cls) -> Dict[str, Any]:
        """Return json representation of Base object.

        Function is written to dump a base object
        to json from which can be considered as serialized object, therefore it's possible to
        use this representation to load the very same object.
        However compare to python serializer, performance of much slower.
        """
        result = {}
        tree = ToJsonSchemaTree(result)
        tree.add_to_process(data=cls, data_type=cls, parent_index="root", result=result)
        tree.process()
        return result["root"]

    def content_to_json(self) -> Dict[str, Any]:
        """Similar to `Base.to_json` method, but doesn't include information of type of the object.

        Only it's content. This is exit only method. Output if this method cannot be used as
        input for any 'load' method.
        """
        result = {}
        tree = ContentToJsonTree(result)
        tree.add_to_process(data=self, data_type=self.__class__, parent_index="root", result=result)
        tree.process()
        return result["root"]

    @classmethod
    def _gather_child_classes(cls, parent_cls) -> List[Type[Any]]:
        stack = [parent_cls]
        results = []
        while stack:
            current_cls = stack.pop(0)
            if hasattr(current_cls, "_child_classes") and current_cls._child_classes:
                for child_cls in current_cls._child_classes:
                    if child_cls not in results:
                        results.append(child_cls)
                        stack.append(child_cls)
        return results

    @classmethod
    def content_from_json(cls, json_dict: Dict[str, Any]):
        """Deserialize class from json data."""
        root_init_fields = {}
        stack: List[
            Tuple[
                Type[ANY],  # parent class
                Union[Optional[str], Optional[int]],  # parent field
                Union[TList[JSON_COMPATIBLE], Dict[str, JSON_COMPATIBLE]],  # parent_init_fields
                Dict[str, ANY],  # json_data
                str,  # path to current object
            ]
        ] = []
        order: List[
            Tuple[Type[Any], Dict[str, JSON_COMPATIBLE], Union[str, int], JSON_COMPATIBLE]
        ] = []
        stack.append(((cls,), "root", json_dict, root_init_fields, "#"))

        ANY_LIST_TYPE_NODE = TypeNode.from_type(TList[ANY])
        ANY_DICT_TYPE_NODE = TypeNode.from_type(TDict[ANY, ANY])
        BASE_TYPE_NODE = TypeNode.from_type(Base)

        while stack:
            (parent_cls_candidates, parent_key, _json_dict, parent_init_fields, parent_path) = (
                stack.pop(0)
            )
            optional_uargs = [
                x
                for x in parent_cls_candidates
                if x.__class__ == _UnionGenericAlias and x._name == "Optional"
            ]
            union_uargs = [
                x
                for x in parent_cls_candidates
                if x.__class__ == _UnionGenericAlias and x._name != "Optional"
            ]

            list_uargs = [
                x
                for x in parent_cls_candidates
                if TypeNode.from_type(x) == ANY_LIST_TYPE_NODE
                and x not in union_uargs
                and x not in optional_uargs
            ]
            dict_uargs = [
                x
                for x in parent_cls_candidates
                if TypeNode.from_type(x) == ANY_DICT_TYPE_NODE
                and x not in union_uargs
                and x not in optional_uargs
            ]
            obj_uargs = [
                x
                for x in parent_cls_candidates
                if x not in list_uargs
                and x not in dict_uargs
                and TypeNode.from_type(x) == BASE_TYPE_NODE
                and x not in optional_uargs
                and x not in union_uargs
            ]
            other_uargs = [
                x
                for x in parent_cls_candidates
                if x not in list_uargs
                and x not in dict_uargs
                and x not in obj_uargs
                and x not in optional_uargs
                and x not in union_uargs
            ]

            if optional_uargs:
                if _json_dict is None:
                    parent_init_fields[parent_key] = None
                else:
                    non_optional_classes = [x.__args__[0] for x in optional_uargs]
                    stack.append(
                        (
                            non_optional_classes
                            + list_uargs
                            + dict_uargs
                            + obj_uargs
                            + union_uargs
                            + other_uargs,
                            parent_key,
                            _json_dict,
                            parent_init_fields,
                            parent_path,
                        )
                    )
            elif union_uargs:
                for union_uarg in union_uargs:
                    all_uargs = []
                    for uarg in union_uarg.__args__:
                        all_uargs.append(uarg)
                    stack.append(
                        (
                            optional_uargs
                            + list_uargs
                            + dict_uargs
                            + obj_uargs
                            + all_uargs
                            + other_uargs,
                            parent_key,
                            _json_dict,
                            parent_init_fields,
                            parent_path,
                        )
                    )
            elif list_uargs and isinstance(_json_dict, list):
                init_fields = [None for x in range(len(_json_dict))]
                for n, value in enumerate(_json_dict):
                    if isinstance(value, (str, int, float, type(None), bool)):
                        init_fields[n] = value
                    else:
                        stack.append(
                            (
                                [x._params[0] for x in list_uargs],
                                n,
                                value,
                                init_fields,
                                parent_path + f".{n}",
                            )
                        )
                for l_cls in list_uargs:
                    order.insert(
                        0, (l_cls, parent_init_fields, parent_key, init_fields, parent_path)
                    )
            elif (obj_uargs or dict_uargs) and isinstance(_json_dict, dict):
                for d_cls in dict_uargs:
                    init_fields = {}
                    if d_cls._params[1] in (str, int, float, type(None)):
                        for k, v in _json_dict.items():
                            init_fields[k] = v
                    else:
                        for k, v in _json_dict.items():
                            stack.append(
                                (
                                    [x._params[1] for x in dict_uargs],
                                    k,
                                    v,
                                    init_fields,
                                    parent_path + f".{k}",
                                )
                            )
                    order.insert(
                        0, (d_cls, parent_init_fields, parent_key, init_fields, parent_path)
                    )
                for o_cls in obj_uargs:
                    if hasattr(o_cls, "_child_classes"):
                        candidates = [o_cls] + o_cls._child_classes
                    else:
                        candidates = [o_cls]
                    for candidate in candidates:
                        init_fields = {}
                        for f in _json_dict:
                            _f = (
                                [
                                    k
                                    for k, v in candidate._SERIALIZE_REPLACE_FIELDS.items()
                                    if v == f
                                ]
                                or [f]
                            )[0]
                            if _f in candidate._fields:
                                ftype = candidate._fields[_f]
                                if ftype in (str, int, float, type(None), bool):
                                    init_fields[_f] = _json_dict[f]
                                else:
                                    stack.append(
                                        (
                                            (ftype,),
                                            _f,
                                            _json_dict[f],
                                            init_fields,
                                            parent_path + f".{_f}",
                                        )
                                    )
                            else:
                                # In this situation field which is uknown to the class is added
                                # to init fields which fails in initialization of class candidate
                                # and eliminate it from the candidate list
                                init_fields[_f] = _json_dict[f]

                        order.insert(
                            0, (o_cls, parent_init_fields, parent_key, init_fields, parent_path)
                        )
            elif other_uargs:
                for o_cls in other_uargs:
                    init_fields = {}
                    if o_cls in (
                        str,
                        int,
                        float,
                        type(None),
                        bool,
                    ):  # or issubclass(o_cls, enum.Enum):
                        parent_init_fields[parent_key] = _json_dict
                    elif issubclass(o_cls, enum.Enum):
                        # parent_init_fields[parent_key] = json_dict
                        init_fields = {"value": _json_dict}
                        order.insert(
                            0, (o_cls, parent_init_fields, parent_key, init_fields, parent_path)
                        )
                    else:
                        raise ValueError(f"Cannot handle {o_cls}")
            else:
                raise ValueError(
                    f"Cannot process data '{_json_dict}' with classes '{parent_cls_candidates}' in"
                    f"'{parent_path}'"
                )

        errors = {}
        for _cls_candidate, parent_init_fields, parent_key, init_fields, parent_path in order:

            # gather sub classes of cls candidate
            if hasattr(_cls_candidate, "_child_classes"):
                child_classes = cls._gather_child_classes(_cls_candidate)
                child_classes.append(_cls_candidate)
            else:
                child_classes = [_cls_candidate]

            for cls_candidate in child_classes:
                if parent_key in parent_init_fields and not isinstance(
                    parent_init_fields[parent_key], Exception
                ):
                    continue
                if (
                    TypeNode.from_type(cls_candidate) == ANY_LIST_TYPE_NODE
                    or TypeNode.from_type(cls_candidate) == ANY_DICT_TYPE_NODE
                ):
                    try:
                        if TypeNode.from_type(cls_candidate) == ANY_DICT_TYPE_NODE:
                            error_items = [
                                (k, f) for k, f in init_fields.items() if isinstance(f, Exception)
                            ]
                        else:
                            error_items = [
                                (n, f)
                                for n, f in enumerate(init_fields)
                                if isinstance(f, Exception)
                            ]
                        if error_items:
                            for n, f in error_items:
                                errors.setdefault(parent_path, []).append(
                                    {"fields": init_fields, "exception": f}
                                )
                            parent_init_fields[parent_key] = ItemSerializationError(
                                [n for n, _ in error_items]
                            )
                            continue
                        parent_init_fields[parent_key] = cls_candidate(init_fields)
                    except Exception as e:
                        errors.setdefault(parent_path, []).append(
                            {"fields": init_fields, "exception": e}
                        )
                        if parent_key not in parent_init_fields:
                            parent_init_fields[parent_key] = e
                    else:
                        break

                else:
                    try:
                        parent_init_fields[parent_key] = cls_candidate(**init_fields)
                    except Exception as e:
                        errors.setdefault(parent_path, []).append(
                            {"fields": init_fields, "exception": e}
                        )
                        if parent_key not in parent_init_fields:
                            parent_init_fields[parent_key] = e
                    else:
                        break

        if "root" not in root_init_fields and errors:
            raise SerializationError(errors)
        if isinstance(root_init_fields["root"], Exception):
            raise SerializationError(errors)
        return root_init_fields["root"]

    @classmethod
    def type_to_json(cls) -> Dict[str, Any]:
        """Similar to `Base.to_json` method, but dumps information only of type of the object."""
        pre_order: Dict[str, Any] = {}
        # stack is list of (current_cls_to_process, current_parent, current_key, current_default)
        stack: List[Tuple[Type[Base], Dict[str, Any], str, Optional[JSON_COMPATIBLE]]] = [
            (cls, pre_order, "root", None)
        ]
        while stack:
            current, current_parent, parent_key, current_default = stack.pop(0)
            if hasattr(current, "_TYPE"):
                current_parent["_TYPE"] = current._TYPE
            if (
                hasattr(current, "_CUSTOM_TYPE_TO_JSON")
                and current._CUSTOM_TYPE_TO_JSON
                and current != cls
            ):
                current_parent[parent_key] = current.type_to_json()
            elif hasattr(current, "_fields"):
                current_parent[parent_key] = {"$type": TypeNode.from_type(current).to_json()}
                for f, ftype in current._fields.items():
                    if type(current.__dataclass_fields__[f].default) in (
                        str,
                        int,
                        float,
                        None,
                        bool,
                    ):
                        stack.append(
                            (
                                ftype,
                                current_parent[parent_key],
                                f,
                                current.__dataclass_fields__[f].default,
                            )
                        )
                    else:
                        _f = (
                            [k for k, v in cls._SERIALIZE_REPLACE_FIELDS.items() if v == f] or [f]
                        )[0]
                        stack.append((ftype, current_parent[parent_key], _f, None))
            else:
                current_parent[parent_key] = {
                    "$type": TypeNode.from_type(current).to_json(),
                    "default": current_default,
                }
        return pre_order["root"]

    @classmethod
    def type_defaults_to_json(cls) -> Dict[str, Any]:
        """Similar to `Base.to_json` method, but dumps information only of type of the object."""
        pre_order: Dict[str, Any] = {}
        # stack is list of (current_cls_to_process, current_parent, current_key, current_default)
        stack: List[Tuple[Type[Base], Dict[str, Any], str, Optional[JSON_COMPATIBLE]]] = [
            (cls, pre_order, "root", None)
        ]
        while stack:
            current, current_parent, parent_key, current_default = stack.pop(0)
            if hasattr(current, "_TYPE"):
                current_parent["_TYPE"] = current._TYPE
            if (
                hasattr(current, "_CUSTOM_TYPE_TO_JSON")
                and current._CUSTOM_TYPE_TO_JSON
                and current != cls
            ):
                current_parent[parent_key] = current.type_to_json()
            elif hasattr(current, "_fields"):
                current_parent[parent_key] = {
                    #    "$type": current
                }
                for f, ftype in current._fields.items():
                    if type(current.__dataclass_fields__[f].default) in (
                        str,
                        int,
                        float,
                    ):
                        stack.append(
                            (
                                ftype,
                                current_parent[parent_key],
                                f,
                                current.__dataclass_fields__[f].default,
                            )
                        )
                    else:
                        stack.append((ftype, current_parent[parent_key], f, ftype))
            else:
                if isinstance(current_default, (int, float)):
                    current_parent[parent_key] = current_default
                else:
                    current_parent[parent_key] = str(current_default)
        return pre_order["root"]

    @classmethod
    def from_json(cls, json_data: JSON_COMPATIBLE, _locals={}) -> _Base:
        """Oposite to `Base.to_json` method. Method returns dumped instance of a Base class."""
        stack: List[
            Tuple[
                Dict[str, JSON_COMPATIBLE],
                str,
                Dict[str, JSON_COMPATIBLE],
                Type[Optional[Base]],
                Dict[str, ANY],
            ]
        ] = []
        post_order = []
        root_args: Dict[str, Any] = {"root": None}
        if json_data.get("$type"):
            json_cls = TypeNode.from_json(json_data.get("$type"), _locals=_locals).to_type()
        else:
            json_cls = cls

        stack.append((root_args, "root", json_data, json_cls, {}))

        while stack:
            parent_args, parent_key, data, type_, type_args = stack.pop(0)
            _parent_key = (
                [k for k, v in cls._SERIALIZE_REPLACE_FIELDS.items() if v == parent_key]
                or [parent_key]
            )[0]
            if hasattr(type_, "__qualname__") and type_.__qualname__ in (
                "Optional",
                "Union",
            ):
                if isinstance(data, dict):
                    data_type = data.get("$type", None)
                else:
                    data_type = TypeNode.from_type(data.__class__).to_json()
                for uarg in get_args(type_):
                    if TypeNode.from_type(uarg) == TypeNode.from_json(data_type, _locals=_locals):
                        stack.append((parent_args, _parent_key, data, uarg, type_args))
                        break
                else:
                    stack.append((parent_args, _parent_key, data, type(None), type_args))

            elif hasattr(type_, "__qualname__") and type_.__qualname__ == "Union":
                if isinstance(data, dict):
                    data_type = data.get("$type", None)
                else:
                    data_type = TypeNode.from_type(data.__class__).to_json()
                for uarg in get_args(type_):
                    if (
                        TypeNode.from_type(uarg).to_json()
                        == TypeNode.from_json(data_type, _locals=_locals).to_json()
                    ):
                        stack.append((parent_args, _parent_key, data, uarg, type_args))
                        break
            elif TypeNode.from_type(type_) == TypeNode.from_type(TList[ANY]):
                parent_args[_parent_key] = type_.from_json(data, _locals=_locals)
            elif TypeNode.from_type(type_) == TypeNode.from_type(TDict[ANY, ANY]):
                parent_args[_parent_key] = type_.from_json(data, _locals=_locals)
            elif type_ not in (int, str, bool, float, type(None)) and not issubclass(
                type_, enum.Enum
            ):
                for key in type_._fields:
                    field_args = {}
                    if "$data" in data:
                        stack.append(
                            (
                                type_args,
                                key,
                                data["$data"].get(key),
                                type_._fields[key],
                                field_args,
                            )
                        )
                    else:
                        stack.append(
                            (
                                type_args,
                                key,
                                data.get(key),
                                type_._fields[key],
                                field_args,
                            )
                        )
                if "$data" in data:
                    extra = data.get("$data", {}).keys() - type_._fields.keys()
                else:
                    extra = data.keys() - type_._fields.keys()
                if extra:
                    raise ValueError(f"There are extra attributes uknown to type {type_}: {extra}")

                post_order.insert(0, (parent_args, parent_key, type_, type_args))
            elif issubclass(type_, enum.Enum):
                parent_args[parent_key] = type_(data)
            else:
                parent_args[parent_key] = data

        for parent_args, parent_key, type_, type_args in post_order:
            init_fields = {}
            for k, v in type_args.items():
                if type_.__dataclass_fields__[k].init:
                    init_fields[k] = v
            parent_args[parent_key] = type_(**init_fields)
            for k, v in type_args.items():
                if not type_.__dataclass_fields__[k].init:
                    setattr(parent_args[parent_key], k, v)

        return root_args["root"]

    def __eq__(self, other):
        """Test if class is equal to other."""
        if TypeNode.from_type(type(self)) != TypeNode.from_type(type(other)):
            return False
        for f in self._fields:
            if getattr(self, f) != getattr(other, f):
                return False
        return True


class _REMOVED(Base):
    """Class used to indicate that field was removed from the list or dict."""

    pass


REMOVED = _REMOVED()


T = TypeVar("T")
TK = TypeVar("TK")
TV = TypeVar("TV")


class TList(Base, ATList, Generic[T]):
    """Similar to python list except this checks type of content.

    Class won't allow to add content of
    different type then what is provided as typevar.

    Example usage: TList[str](['foo'])
    """

    _list: List[T] = Field(default_factory=list)

    def __new__(cls, *args, **kwargs):
        """Return new TList instance."""
        if not cls._params:
            raise TypeError("Cannot create TList without subtype, construct with TList[<type>]")
        return Base.__new__(cls)

    def __init__(self, iterable=[]):
        """Initialize Tlist with iterable."""
        self._list = []
        for item in iterable:
            if TypeNode.from_type(type(item)) != TypeNode.from_type(self._params[0]):
                raise TypeError(
                    f"Cannot assign item {type(item)} to list of type {self._params[0]}"
                )
        list.__init__(self._list, iterable)
        self._observer = BaseObserver()

    def __add__(self, value):
        """Return two lists joined together."""
        if TypeNode.from_type(type(value)) != TypeNode.from_type(type(self)):
            raise TypeError(f"Cannot extend list {type(self)} with {type(value)}")
        return self.__class__(self._list.__add__(value._list))

    def __contains__(self, value):
        """Test if list contains a value."""
        if TypeNode.from_type(type(value)) != TypeNode.from_type(self._params[0]):
            raise TypeError(f"Cannot call __contains__ on {self._params[0]} with {type(value)}")
        return self._list.__contains__(value)

    def __delitem__(self, x):
        """Remove item from the list."""
        self._observed(x, None)
        return self._list.__delitem__(x)

    def __getitem__(self, x):
        """Return item from the list."""
        return self._list.__getitem__(x)

    def __iter__(self):
        """Iterate over the list."""
        for x in self._list:
            yield x

    def __len__(self):
        """Return length of the list."""
        return self._list.__len__()

    def __reversed__(self):
        """Return __reversed__ of the list."""
        return self._list.__reversed__()

    def __setitem__(self, key, value):
        """Set item to the list."""
        if TypeNode.from_type(type(value)) != TypeNode.from_type(self._params[0]):
            raise TypeError(f"Cannot assign item {type(value)} to list of type {self._params[0]}")
        self._update_observers(f"[{key}]", value)
        self._observed(f'["{key}"]', value)
        self._list.__setitem__(key, value)

    def append(self, obj: T) -> None:
        """Append item to the end of the list."""
        if TypeNode.from_type(type(obj)) != TypeNode.from_type(self._params[0]):
            raise TypeError(f"Cannot assign item {type(obj)} to list of type {self._params[0]}")
        self._observed(f"[{len(self._list)}]", obj)
        self._list.append(obj)

    def clear(self):
        """Clear the list."""
        return self._list.clear()

    def count(self, value):
        """Return number of occurences of the value in the list."""
        return self._list.count(value)

    def extend(self, iterable):
        """Extend list if the given iterable."""
        if TypeNode.from_type(type(iterable)) != TypeNode.from_type(type(self)):
            raise TypeError(
                f"Cannot extend list {self.__class__.__name__} with {iterable.__class__.__name__}"
            )
        self._list.extend(iterable._list)

    def index(self, value, start=0, stop=-1):
        """Return index of the value in the list."""
        for i, v in enumerate(self._list, start=start):
            if v == value:
                return i
            if stop != -1 and i > stop:
                break
        else:
            raise ValueError(f"{value} is not in the list")

    def insert(self, index, obj):
        """Insert item to the list."""
        if TypeNode.from_type(type(obj)) != TypeNode.from_type(self._params[0]):
            raise TypeError(
                f"Cannot assign item {type(obj)} to list of type {type(self._params[0])}"
            )
        self._list.insert(index, obj)

    def pop(self, *args, **kwargs):
        """Remove and return item from the list."""
        return self._list.pop(*args, **kwargs)

    def remove(self, value):
        """Remove item from the list."""
        self._observed(value, REMOVED)
        return self._list.remove(value)

    def reverse(self):
        """Return reversed list."""
        return self._list.reverse()

    def sort(self, *args, **kwargs):
        """Return sorted list."""
        return self._list.sort(*args, **kwargs)

    def to_json(self) -> Dict[str, Any]:
        """Serialize TList to json representation."""
        result = {}
        tree = ToJsonTree(result)
        tree.add_to_process(data=self, data_type=self.__class__, parent_index="root", result=result)
        tree.process()
        return result["root"]

    def content_to_json(self) -> Dict[str, Any]:
        """Serialize TList content to json representation."""
        result = {}
        tree = ContentToJsonTree(result)
        tree.add_to_process(data=self, data_type=self.__class__, parent_index="root", result=result)
        tree.process()
        return result["root"]

    @classmethod
    def from_json(cls, json_data, _locals={}) -> _Base:
        """Deserialize TList from json data."""
        ANY_LIST_TYPE_NODE = TypeNode.from_type(TList[ANY])
        ANY_DICT_TYPE_NODE = TypeNode.from_type(TDict[ANY, ANY])

        stack = []
        post_order = []
        self_type_json = TypeNode.from_type(cls).to_json()
        root_args: Dict[str, Any] = {"root": None}
        if TypeNode.from_json(json_data["$type"], _locals=_locals) != TypeNode.from_type(cls):
            raise ValueError(f"Cannot load {json_data['$type']} to {self_type_json}")

        root_type_args = {"iterable": []}
        for n, item in enumerate(json_data["$data"]):
            root_type_args["iterable"].append(None)
            if not isinstance(item, (int, str, bool, float, type(None))):
                item_type = TypeNode.from_json(item["$type"], _locals=_locals).to_type(
                    types_cache=cls._generic_cache
                )
            else:
                item_type = cls._params[0]
            stack.append((root_type_args["iterable"], n, item, item_type, {}))

        while stack:
            parent_args, parent_key, data, type_, type_args = stack.pop(0)
            if hasattr(type_, "__qualname__") and type_.__qualname__ in (
                "Optional",
                "Union",
            ):
                if isinstance(data, dict):
                    data_type = data.get("$type", None)
                else:
                    data_type = TypeNode.from_type(data.__class__).to_json()
                for uarg in get_args(type_):
                    if TypeNode.from_json(data_type, _locals=_locals) == TypeNode.from_type(uarg):
                        stack.append((parent_args, parent_key, data, uarg, type_args))
                        break
                    if data_type == TypeNode.from_type(uarg).to_json():
                        stack.append((parent_args, parent_key, data, uarg, type_args))
                        break
                else:
                    stack.append((parent_args, parent_key, data, type(None), type_args))
            elif (
                TypeNode.from_type(type_) == ANY_LIST_TYPE_NODE
                or TypeNode.from_type(type_) == ANY_DICT_TYPE_NODE
            ):
                parent_args[parent_key] = type_.from_json(data, _locals=_locals)
            elif type_ not in (int, str, bool, float, type(None)) and not issubclass(
                type_, enum.Enum
            ):
                for key in type_._fields:
                    field_args = {}
                    if "$data" in data:
                        stack.append(
                            (
                                type_args,
                                key,
                                data["$data"].get(key),
                                type_._fields[key],
                                field_args,
                            )
                        )
                    else:
                        stack.append(
                            (
                                type_args,
                                key,
                                data.get(key),
                                type_._fields[key],
                                field_args,
                            )
                        )

                post_order.insert(0, (parent_args, parent_key, type_, type_args))
            elif issubclass(type_, enum.Enum):
                parent_args[parent_key] = type_(data)
            else:
                if type_ is type(None):
                    parent_args[parent_key] = None
                else:
                    parent_args[parent_key] = data

        for parent_args, parent_key, type_, type_args in post_order:
            init_fields = {}
            for k, v in type_args.items():
                if type_.__dataclass_fields__[k].init:
                    init_fields[k] = v
            parent_args[parent_key] = type_(**init_fields)
            for k, v in type_args.items():
                if not type_.__dataclass_fields__[k].init:
                    setattr(parent_args[parent_key], k, v)

        root_args["root"] = cls(root_type_args["iterable"])
        return root_args["root"]


ANY_LIST_TYPE_NODE = TypeNode.from_type(TList[ANY])


class TDict(Base, ATDict, Generic[TK, TV]):
    """Similar to python dict except this checks type of content.

    Class won't allow to add content of
    different type then what is provided as typevar.

    Example usage: TDict[str, int]({'foo': 1})
    """

    _dict: Dict[TK, TV] = Field(default_factory=dict)

    def __new__(cls, *args, **kwargs):
        """Return new TDict instance."""
        if not cls._params:
            raise TypeError("Cannot create TDict without subtype, construct with TDict[<type>]")
        ret = Base.__new__(cls)
        _tk = ret._params[0]
        _tv = ret._params[1]
        ret._type_node_key = TypeNode.from_type(_tk)
        ret._type_node_value = TypeNode.from_type(_tv)

        return ret

    def __contains__(self, key: TK) -> bool:
        """Test if dict contains the key."""
        _tk = self._params[0]
        _tv = self._params[1]
        if TypeNode.from_type(type(key)) != self._type_node_key:
            raise TypeError(
                f"Cannot check key {key} of type {type(key)} in dict of type {Dict[_tk, _tv]}"
            )
        return self._dict.__contains__(key)

    def __delitem__(self, key: TK):
        """Remove item from the dict by given key."""
        _tk = self._params[0]
        _tv = self._params[1]
        if TypeNode.from_type(type(key)) != self._type_node_key:
            raise TypeError(
                f"Cannot remove key {key} of type {type(key)} in dict of type {Dict[_tk, _tv]}"
            )
        self._dict.__delitem__(key)
        self._observed(f'["{key}"]', REMOVED)

    def __getitem__(self, key: TK) -> TV:
        """Return item from the dict for given key."""
        _tk = self._params[0]
        _tv = self._params[1]
        if TypeNode.from_type(type(key)) != self._type_node_key:
            raise TypeError(
                f"Cannot get item by key {key} of type {type(key)} in dict of type {Dict[_tk, _tv]}"
            )
        return self._dict.__getitem__(key)

    def __init__(self, d={}):
        """Initialize the TDict instance."""
        self._dict = {}
        for k, v in d.items():
            self.__setitem__(k, v)
        self._observer = BaseObserver()

    def __iter__(self):
        """Iterate over the dict."""
        return self._dict.__iter__()

    def __len__(self):
        """Return length of the dict."""
        return self._dict.__len__()

    def __reversed__(self):
        """Return __reversed__ of the dict."""
        return self._dict.__reversed__()

    def __setitem__(self, k: TK, v: TV):
        """Set item to the dict to given key."""
        _tk = self._params[0]
        _tv = self._params[1]
        k_tn = type(k)._type_node if hasattr(type(k), "_type_node") else TypeNode.from_type(type(k))
        v_tn = type(v)._type_node if hasattr(type(v), "_type_node") else TypeNode.from_type(type(v))

        if k_tn != self._type_node_key:
            raise TypeError(
                f"Cannot set item by key {k} of type {type(k)} in dict of type {Dict[_tk, _tv]}"
            )
        if v_tn != TypeNode.from_type(_tv):
            raise TypeError(
                f"Cannot set item {v} of type {type(v)} in dict of type {Dict[_tk, _tv]}"
            )
        self._update_observers(f'["{k}"]', v)
        self._observed(f'["{k}"]', v)
        self._dict.__setitem__(k, v)

    def clear(self):
        """Clear the dict."""
        self._dict.clear()

    def fromkeys(self, iterable, value):
        """Return new TDict instance with items from iterable."""
        _tk = self._params[0]
        _tv = self._params[1]
        for key in iterable:
            if TypeNode.from_type(type(key)) != self._type_node_key:
                raise TypeError(
                    f"Cannot set item by key {key} of type {type(key)} in dict of "
                    f"type {Dict[_tk, _tv]}"
                )
        if TypeNode.from_type(type(value)) != TypeNode.from_type(_tv):
            raise TypeError(
                f"Cannot set item {value} of type {type(value)} in dict of type {Dict[_tk, _tv]}"
            )
        new_d = self._dict.fromkeys(iterable, value)
        return self.__class__(new_d)

    def get(self, key: TK, default=None):
        """Get item from the dict or return default if not found."""
        if TypeNode.from_type(type(key)) != self._type_node_key:
            raise TypeError(
                f"Cannot get item by key {key} of type {type(key)} in dict of type "
                f"TDict[{self._params}]"
            )
        return self._dict.get(key, default)

    def items(self):
        """Return items of the dict."""
        return self._dict.items()

    def keys(self):
        """Return keys of the items of the dict."""
        return self._dict.keys()

    def pop(self, k: TK, d=None):
        """Remove and return item from the dict by given key."""
        _tk = self._params[0]
        _tv = self._params[1]
        if TypeNode.from_type(type(k)) != TypeNode.from_type(_tk):
            raise TypeError(
                f"Cannot pop item by key {k} of type {type(k)} in dict of type {Dict[_tk, _tv]}"
            )
        self._observed(f'["{k}"]', REMOVED)
        return self._dict.pop(k, d)

    def popitem(self) -> Tuple[TK, TV]:
        """Remove item from the dict."""
        return self._dict.popitem()

    def setdefault(self, key, default):
        """Set dict default value for given key."""
        _tk = self._params[0]
        _tv = self._params[1]
        if TypeNode.from_type(type(key)) != self._type_node_key:
            raise TypeError(
                f"Cannot setdefault for key {key} of type {type(key)} in dict of "
                f"type {Dict[_tk, _tv]}"
            )
        if TypeNode.from_type(type(default)) != TypeNode.from_type(_tv):
            raise TypeError(
                f"Cannot setdefault '{default}' of type {type(default)} in dict of "
                f"type {Dict[_tk, _tv]}"
            )
        self._observed(f'["{key}"]', default)
        return self._dict.setdefault(key, default)

    def update(self, other):
        """Update dict with another dict."""
        if TypeNode.from_type(type(other)) != TypeNode.from_type(type(self)):
            raise TypeError(f"Cannot update dict {Dict[TK, TV]} with type {type(other)}")
        self._dict.update(other)

    def values(self):
        """Return item values of the dict."""
        return self._dict.values()

    def to_json(self) -> Dict[str, Any]:
        """Serialize TDict to json representation."""
        result = {}
        tree = ToJsonTree(result)
        tree.add_to_process(data=self, data_type=self.__class__, parent_index="root", result=result)
        tree.process()
        return result["root"]

    def content_to_json(self) -> Dict[str, Any]:
        """Serialize TDict content to json representation."""
        result = {}
        tree = ContentToJsonTree(result)
        tree.add_to_process(data=self, data_type=self.__class__, parent_index="root", result=result)
        tree.process()
        return result["root"]

    @classmethod
    def from_json(cls, json_data, _locals={}) -> "TDict":
        """Deserialize TDict from json."""
        stack = []
        post_order = []
        self_type_json = TypeNode.from_type(cls).to_json()
        root_args: Dict[str, Any] = {"root": None}
        if TypeNode.from_json(json_data["$type"], _locals=_locals) != TypeNode.from_type(cls):
            raise ValueError(f"Cannot load {json_data['$type']} to {self_type_json}")

        root_type_args = {"iterable": cls({})}
        for k, v in json_data["$data"].items():
            if not isinstance(v, (int, str, bool, float, type(None))):
                item_type = TypeNode.from_json(v["$type"], _locals=_locals).to_type(
                    types_cache=cls._generic_cache
                )
            else:
                item_type = cls._params[1]
            # need to insert in reversed order as traversal reverse the dictionary
            if type(k) is not cls._params[0]:
                json_key = cls._params[0].from_json(json.loads(k), _locals=_locals)
            else:
                json_key = k
            stack.insert(0, (root_type_args["iterable"], json_key, v, item_type, {}))

        while stack:
            parent_args, parent_key, data, type_, type_args = stack.pop(0)
            if hasattr(type_, "__qualname__") and type_.__qualname__ in (
                "Optional",
                "Union",
            ):
                if isinstance(data, dict):
                    data_type = data.get("$type", None)
                else:
                    data_type = TypeNode.from_type(data.__class__).to_json()
                for uarg in get_args(type_):
                    if TypeNode.from_json(data_type, _locals=_locals) == TypeNode.from_type(uarg):
                        stack.append((parent_args, parent_key, data, uarg, type_args))
                        break
                    if data_type == TypeNode.from_type(uarg).to_json():
                        stack.append((parent_args, parent_key, data, uarg, type_args))
                        break
                else:
                    stack.append((parent_args, parent_key, data, type(None), type_args))
            elif TypeNode.from_type(type_) == ANY_LIST_TYPE_NODE or TypeNode.from_type(
                type_
            ) == TypeNode.from_type(TDict[ANY, ANY]):
                parent_args[parent_key] = type_.from_json(data, _locals=_locals)
            elif type_ not in (int, str, bool, float, type(None)) and not issubclass(
                type_, enum.Enum
            ):
                for key in type_._fields:
                    field_args = {}
                    if "$data" in data:
                        # need to take type from data, in the case of subclass is used instead of
                        # generic type
                        data_type = (
                            TypeNode.from_json(data["$type"], _locals=_locals)
                            .to_type()
                            ._fields[key]
                        )
                        stack.append(
                            (
                                type_args,
                                key,
                                data["$data"].get(key),
                                data_type,
                                field_args,
                            )
                        )
                    else:
                        stack.append(
                            (
                                type_args,
                                key,
                                data.get(key),
                                type_._fields[key],
                                field_args,
                            )
                        )

                post_order.insert(0, (parent_args, parent_key, type_, type_args))
            elif issubclass(type_, enum.Enum):
                parent_args[parent_key] = type_(data)
            else:
                if type_ is type(None):
                    parent_args[parent_key] = None
                else:
                    parent_args[parent_key] = data

        for parent_args, parent_key, type_, type_args in post_order:
            init_fields = {}

            for k, v in type_args.items():
                if type_.__dataclass_fields__[k].init:
                    init_fields[k] = v
            parent_args[parent_key] = type_(**init_fields)
            for k, v in type_args.items():
                if not type_.__dataclass_fields__[k].init:
                    setattr(parent_args[parent_key], k, v)

        root_args["root"] = cls(root_type_args["iterable"])
        return root_args["root"]


# Currently not used
# class IOStore:
#     def data(self, key: str) -> Any:
#         pass
#
#     def set_data(self, key: str, val: Any):
#         pass


class STMDSingleIn(Base, Generic[T]):
    """Class used for input of a Tractor instance."""

    _TYPE: ClassVar[str] = "PORT"

    _KW_ONLY: ClassVar[bool] = False

    _ref: Optional[T] = Field(repr=False, init=False, default=None, compare=False)
    # data here are actually not used after input is assigned to some output
    # it's just to deceive mypy
    data: Optional[T] = None
    _data_proxy: Optional[list[str]] = None
    _name: str = Field(repr=False, init=False, default=None, compare=False)
    _owner: Optional[_Traction] = Field(repr=False, init=False, default=None, compare=False)
    # _io_store: IOStore = dataclasses.field(repr=False, init=False, compare=False)
    _uid: str = Field(repr=False, init=False, default=None, compare=False)

    def __post_init__(self):
        """Post init of STMDIn class."""
        # Currently not used
        # self._io_store = DefaultIOStore.io_store
        pass

    def _validate_setattr_(self, name: str, value: Any) -> None:
        """Set attribute to class instance with type validation."""
        if name in ("_name", "_owner"):
            # old_uid = self._uid
            self._uid = None
            object.__setattr__(self, name, value)
            return
        if not name.startswith("_"):  # do not check for private attrs
            if (
                name not in self._fields
                and not self._config.allow_extra
                and name not in self._properties
            ):
                raise AttributeError(f"{self.__class__} doesn't have attribute {name}")
            if name == "data":
                # if not hasattr(self, "_io_store"):
                #    self._io_store = DefaultIOStore.io_store
                vtype = (
                    value.__orig_class__ if hasattr(value, "__orig_class__") else value.__class__
                )
                tt1 = TypeNode.from_type(vtype, type_aliases=[(type(True), _defaultBool)])
                tt2 = TypeNode.from_type(
                    self._fields[name], type_aliases=[(type(True), _defaultBool)]
                )
                if tt1 != tt2:
                    raise TypeError(
                        f"Cannot set attribute {self.__class__}.{name} to type {value}({vtype}),"
                        f"expected {self._fields[name]}"
                    )

                self._update_observers(name, value, path_attr="")
                self._observed(None, value)
                object.__setattr__(self, name, value)
                # getattr(self.__class__, name).setter(value)
                return
                # return self._io_store.set_data(self.uid, value)
            elif name not in self._properties:
                vtype = (
                    value.__orig_class__ if hasattr(value, "__orig_class__") else value.__class__
                )
                tt1 = TypeNode.from_type(vtype)
                tt2 = TypeNode.from_type(self._fields[name])
                if tt1 != tt2:
                    raise TypeError(
                        f"Cannot set attribute {self.__class__}.{name} to type {value}({vtype}),"
                        f"expected {self._fields[name]}"
                    )
            else:
                getattr(self.__class__, name).setter(value)
                return
        return object.__setattr__(self, name, value)

    def __getattribute__(self, name) -> Any:
        """Get attribute."""
        if name == "data":
            if object.__getattribute__(self, "_ref"):
                data_obj = object.__getattribute__(self, "_ref")
                if object.__getattribute__(self, "_data_proxy"):
                    for path in object.__getattribute__(self, "_data_proxy"):
                        data_obj = object.__getattribute__(data_obj, path)
                else:
                    data_obj = object.__getattribute__(data_obj, "data")
                return data_obj
            else:
                return object.__getattribute__(self, "data")
        else:
            ret = object.__getattribute__(self, name)
            return ret

    def uid(self):
        """Return uid of the object."""
        if not self._name:
            self._name = str(uuid.uuid4())
        if self._uid is None:
            self._uid = (self._owner.fullname if self._owner else "") + "::" + self._name
        return self._uid


class Port(STMDSingleIn, Generic[T]):
    """Class used for inputs, outputs, args or resources."""

    pass


class NullPort(Port, Generic[T]):
    """Class used for input of a Tractor instance.

    It's same as `In` class but tractions won't set owner to input to self.
    Class needs to be used with specific type as generic Typevar

    Example: NullPort[str]()
    """

    pass


# Currently not used
# class MemoryIOStore(IOStore):
#     def __init__(self):
#         self._data = {}
#
#     def data(self, key: str) -> Any:
#         print("MEMORY STORE GET", key, self._data.get(key))
#
#         return self._data.get(key, NoData())
#
#     def move_data(self, old_key: str, new_key: str) -> Any:
#         data = self._data.pop(old_key, None)
#         self._data[new_key] = data
#
#     def set_data(self, key: str, val: Any):
#         print("MEMORY STORE SET", key, val)
#         self._data[key] = val
#
#
# class _DefaultIOStore:
#     def __init__(self):
#         self.io_store = MemoryIOStore()
#
#
# DefaultIOStore = _DefaultIOStore()


# class Res(Base, Generic[T]):
#     """Class represeting Traction resources.
#
#     Usage: Res[GithubClient](r=gh_client)
#     """
#
#     _TYPE: ClassVar[str] = "RES"
#     r: T


# class TRes(Res, Generic[T]):
#     """Class represeting Traction resources.
#
#     Usage: Res[GithubClient](r=gh_client)
#     """
#
#     _TYPE: ClassVar[str] = "RES"
#     r: Optional[T] = None
#
#     def __post_init__(self):
#         """Set attributes after instance initialization."""
#         self._no_validate_setattr_("r", None)
#
#
# ANY_RES_TYPE_NODE = TypeNode.from_type(Res[ANY])


# class Arg(Base, Generic[T]):
#     """Class represeting Traction argument.
#
#     Usage: Arg[int](a=10)
#     """
#
#     _TYPE: ClassVar[str] = "ARG"
#     a: T
#
#
# ANY_ARG_TYPE_NODE = TypeNode.from_type(Arg[ANY])

ANY_PORT_TYPE_NODE = TypeNode.from_type(Port[ANY])


class MultiArgMeta(BaseMeta):
    """MultiArg metaclass."""

    @classmethod
    def _attribute_check(cls, attr, type_, all_attrs):
        """Check if given attribute is type of arg."""
        if attr.startswith("a_"):
            if TypeNode.from_type(type_, subclass_check=False) != ANY_PORT_TYPE_NODE:
                raise TypeError(f"Attribute {attr} has to be type Port[ANY], but is {type_}")
        else:
            raise TypeError(f"Attribute {attr} has start with i_, o_, a_ or r_")


class MultiArg(Base, metaclass=MultiArgMeta):
    """Multiarg class."""

    pass
