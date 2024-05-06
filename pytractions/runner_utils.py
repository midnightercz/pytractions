import enum
import importlib
from typing import Optional, _SpecialForm, _UnionGenericAlias

from lark import Lark, Transformer
import yaml

from .base import Base, TypeNode, ANY, TList, TDict


class StrParam:
    def __init__(self, strparam):
        self.strparam = strparam

def str_presenter(dumper, data):
    if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)


def str_param(dumper, data):
    return dumper.represent_scalar('tag:yaml.org,2002:str', data.strparam, style='|')

yaml.add_representer(str, str_presenter)
yaml.add_representer(StrParam, str_param)


traction_str_grammar = r"""
start: type
type: fullname | fullname "[" typelist "]"
fullname: name ":" name
typelist: type | type "," typelist
name: NAME
NAME: /([a-zA-Z_])+([a-zA-Z_\.])*/
"""


class TTransformer(Transformer):
    def __init___(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stack = []
        self.current_traction = {
            'mod': None,
            'name': None,
            'args': []
        }

    def type(self, token):
        if len(token) > 1:
            return token[0].__class_getitem__(tuple(token[1]))
        else:
            return token[0]

    def typelist(self, token):
        if len(token) > 1:
            return tuple(token)
        else:
            return token

    def name(self, token):
        return str(token[0])

    def fullname(self, token):
        mod, clsname = token
        mod_ = importlib.import_module(mod)
        return getattr(mod_, clsname)

    def start(self, token):
        return token[0]


def parse_traction_str(traction_str):
    parser = Lark(traction_str_grammar, parser='lalr', transformer=TTransformer())
    parsed = parser.parse(traction_str)
    return parsed


def generate_type_description(type_, indent=0):
    if type_ in (str, int, float, bool, type(None)):
        return f"{type_.__name__}"
    elif type_.__class__ == _UnionGenericAlias:
        return f"Optional[{generate_type_description(type_.__args__[0], indent=indent)}]"
    elif TypeNode.from_type(type_) == TypeNode.from_type(TList[ANY]):
        return f"List[{generate_type_description(type_._params[0], indent=indent)}]"
    elif TypeNode.from_type(type_) == TypeNode.from_type(TDict[ANY, ANY]):
        return f"Dict[{generate_type_description(type_._params[0], indent=indent)},"\
               f"{generate_type_description(type_._params[1], indent=indent)}]"
    elif issubclass(type_, enum.Enum):
        return f"{type_.__name__}"
    else:
        fields = ',\n'.join([" " * (indent+4) + f"{f}: {generate_type_description(t, indent=indent+4)}"
                            for f, t in type_._fields.items()
                            if not f.startswith("_")])
        ret = f"{type_.__name__}(\n{fields}\n)"
    return ret


def generate_param_description(traction, field):
    description = f"""DESCRIPTION: '{getattr(traction, "d_" + field, None) or ""}'

TYPE: {generate_type_description(traction._fields[field])}"""
    return description


def generate_traction_ari(traction):
    args = []
    inputs = []
    resources = []
    for f, fv in traction._fields.items():
        if f.startswith("a_"):
            arg = {"name": f,
                     "description": generate_param_description(traction, f)}
            args.append(arg)
        elif f.startswith("i_"):
            input_ = {"name": f,
                     "description": generate_param_description(traction, f)}
            inputs.append(input_)
        elif f.startswith("r_"):
            resource = {"name": f,
                     "description": generate_param_description(traction, f)}
            resources.append(resource)
    ari_str = yaml.dump_all([{"name": p["name"],
                               "data": StrParam(f'{p["description"]}')} for p in args+resources+inputs])
    return ari_str
