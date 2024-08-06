import enum
import dataclasses
import importlib
from typing import _UnionGenericAlias, Any, Optional, Dict, Type, Tuple, List

from lark import Lark, Transformer
import yaml

from .base import TypeNode, ANY, TList, TDict, JSON_COMPATIBLE, Base


class StrUnion(yaml.YAMLObject):
    """Class for different yaml string formatting."""

    yaml_tag = "OneOf"

    def __init__(self, _list):
        """Initialize StrUnion."""
        self.choices = _list

    def __getitem__(self, n):
        """Get item from choices."""
        return self.choices[n]

    def __setitem__(self, n, x):
        """Set item in choices."""
        self.choices[n] = x


class StrParam:
    """Class for different yaml string formatting."""

    def __init__(self, strparam):
        """Initialize StrParam."""
        self.strparam = strparam


def str_presenter(dumper, data):
    """Yaml string representer with indent."""
    if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)


def str_param(dumper, data):
    """Yaml str param representer with indent."""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data.strparam, style="|")


def str_multichoice(dumper, data):
    """Yaml str param representer with indent."""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data.strparam, style="|")


yaml.add_representer(str, str_presenter)
yaml.add_representer(StrParam, str_param)


traction_str_grammar = r"""
start: type
type: fullname | fullname "[" typelist "]"
fullname: name ":" name
typelist: type | type "," typelist
name: NAME
NAME: /([a-zA-Z_])+([a-z0-9A-Z_\.])*/
"""


class TTransformer(Transformer):
    """Lark grammar transformer.

    This class is used to by lark parser to import and return class from provided string input
    """

    def __init___(self, *args, **kwargs):
        """Tranformer initializer."""
        super().__init__(*args, **kwargs)
        self.stack = []
        self.current_traction = {"mod": None, "name": None, "args": []}

    def type(self, token):
        """Return generic subclass of class in token."""
        if len(token) > 1:
            return token[0].__class_getitem__(tuple(token[1]))
        else:
            return token[0]

    def typelist(self, token):
        """Handle typelist token."""
        if len(token) > 1:
            return tuple(token)
        else:
            return token

    def name(self, token):
        """Extract name from the parsed token."""
        return str(token[0])

    def fullname(self, token):
        """Import class from module fullname and return class."""
        mod, clsname = token
        mod_ = importlib.import_module(mod)
        return getattr(mod_, clsname)

    def start(self, token):
        """Handle grammer start token."""
        return token[0]


def parse_traction_str(traction_str):
    """Parse traction string and return traction class."""
    parser = Lark(traction_str_grammar, parser="lalr", transformer=TTransformer())
    parsed = parser.parse(traction_str)
    return parsed


def generate_type_description(type_, indent=0):
    """Generate description for provided type."""
    if type_ in (str, int, float, bool, type(None)):
        return f"{type_.__name__}"
    elif type_.__class__ == _UnionGenericAlias:
        return f"Optional[{generate_type_description(type_.__args__[0], indent=indent)}]"
    elif TypeNode.from_type(type_) == TypeNode.from_type(TList[ANY]):
        return f"List[{generate_type_description(type_._params[0], indent=indent)}]"
    elif TypeNode.from_type(type_) == TypeNode.from_type(TDict[ANY, ANY]):
        return (
            f"Dict[{generate_type_description(type_._params[0], indent=indent)},"
            f"{generate_type_description(type_._params[1], indent=indent)}]"
        )
    elif issubclass(type_, enum.Enum):
        return f"{type_.__name__}"
    else:
        fields = ",\n".join(
            [
                " " * (indent + 4) + f"{f}: {generate_type_description(t, indent=indent + 4)}"
                for f, t in type_._fields.items()
                if not f.startswith("_")
            ]
        )
        ret = f"{type_.__name__}(\n{fields}\n)"
    return ret


def generate_ari_description(traction, field):
    """Generate description for arg, resource or input field of traction."""
    description = f"""DESCRIPTION: '{getattr(traction, "d_" + field, None) or ""}'

TYPE: {generate_type_description(traction._fields[field])}"""
    return description


def generate_traction_ari(traction):
    """Generate traction args, resources and inputs."""
    args = []
    inputs = []
    resources = []
    for f, fv in traction._fields.items():
        if f.startswith("a_"):
            arg = {"name": f, "description": generate_ari_description(traction, f)}
            args.append(arg)
        elif f.startswith("i_"):
            input_ = {"name": f, "description": generate_ari_description(traction, f)}
            inputs.append(input_)
        elif f.startswith("r_"):
            resource = {"name": f, "description": generate_ari_description(traction, f)}
            resources.append(resource)
    ari_str = yaml.dump_all(
        [
            {"name": p["name"], "data": StrParam(f'{p["description"]}')}
            for p in args + resources + inputs
        ]
    )
    return ari_str


def get_traction_defaults(traction):
    """Get defaults for all traction attributes."""
    defaults = {}
    for f in traction._fields:
        if hasattr(traction, f):
            tf = getattr(traction, f, dataclasses._MISSING_TYPE)
        else:
            tf = traction.__dataclass_fields__[f].default
        if f.startswith("a_"):
            if not isinstance(tf, dataclasses._MISSING_TYPE):
                defaults[f] = StrParam(yaml.dump(tf.content_to_json()))
    if f.startswith("i_"):
        if not isinstance(tf, dataclasses._MISSING_TYPE):
            defaults[f] = StrParam(yaml.dump(tf.content_to_json()))
        if f.startswith("r_"):
            if not isinstance(tf, dataclasses._MISSING_TYPE):
                defaults[f] = StrParam(yaml.dump(tf.content_to_json()))
    return defaults


def gen_default_inputs(traction_cls):
    """Get defaults for all traction attributes."""
    defaults = {str: "string", int: 12345, float: 1.2345, None: None, bool: False, type(None): None}
    pre_order: Dict[str, Any] = {}
    # stack is list of (current_cls_to_process, current_parent, current_key, current_default)
    stack: List[Tuple[Type[Base], Dict[str, Any], str, Optional[JSON_COMPATIBLE]]] = []
    current = traction_cls
    current_parent = pre_order
    parent_key = "root"
    current_default = None

    current_parent[parent_key] = {}
    for f, ftype in current._fields.items():
        if not (f.startswith("a_") or f.startswith("i_") or f.startswith("r_")):
            continue
        if type(current.__dataclass_fields__[f].default) in (str, int, float, None, bool):
            stack.append(
                (
                    ftype,
                    current_parent[parent_key],
                    f,
                    current.__dataclass_fields__[f].default,
                )
            )
        else:
            stack.append((ftype, current_parent[parent_key], f, None))

    while stack:
        current, current_parent, parent_key, current_default = stack.pop(0)
        if current.__class__ == _UnionGenericAlias:  # and current._name != "Optional":
            current_parent[parent_key] = StrUnion([None] * len(current.__args__))
            for n, uarg in enumerate(current.__args__):
                if type(uarg) in (str, int, float, None, bool):
                    stack.append(
                        (
                            uarg,
                            current_parent[parent_key],
                            n,
                            uarg,
                        )
                    )
                else:
                    stack.append((uarg, current_parent[parent_key], n, None))
        elif issubclass(current, enum.Enum):
            current_parent[parent_key] = list(current.__members__.keys())
        elif TypeNode.from_type(current) == TypeNode.from_type(TList[ANY]):
            current_parent[parent_key] = [None]
            stack.append((current._params[0], current_parent[parent_key], 0, None))

        elif hasattr(current, "_fields"):
            current_parent[parent_key] = {}
            for f, ftype in current._fields.items():
                if type(current.__dataclass_fields__[f].default) in (str, int, float, None, bool):
                    stack.append(
                        (
                            ftype,
                            current_parent[parent_key],
                            f,
                            current.__dataclass_fields__[f].default,
                        )
                    )
                else:
                    stack.append((ftype, current_parent[parent_key], f, None))
        else:
            current_parent[parent_key] = (
                defaults[current] if current_default is None else current_default
            )
    return pre_order["root"]
