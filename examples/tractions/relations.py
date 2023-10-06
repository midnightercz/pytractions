from uuid import uuid4

import dataclasses

from typing import Any, _UnionGenericAlias

from typing import Optional, Union, Type, List, TypeVar, Generic, Dict

import pytraction.base
from pytraction.base import Base, Traction, TDict, TList, In, Out, Res, Arg, TypeNode, ANY
from pytraction.tractor import Tractor


import couchdb


T = TypeVar('T')


class AModel(Base):
    pass


class Ref(Base):
    _model_name: str = ""
    uid: str = ""
    _model: Optional[AModel] = None


class RefDefault:
    def __init__(self, model_name):
        self.model_name = model_name

    def __call__(self):
        r = Ref()
        r._model_name = self.model_name
        return r


class BackRef(Base):
    _model_name: str = ""
    uid: str = ""
    _model: Optional[AModel] = None


def _hasattr(inst, name):
    try:
        object.__getattribute__(inst, name)
    except AttributeError:
        return False
    return True


class Model(AModel):
    model_name: str
    uid: str
    

    def _validate_setattr_(self, name, value):
        if name.startswith("_"):
            return super().__setattr__(name, value)
        if name not in self._fields:
            raise AttributeError("Cannot set attribute {} on model {}".format(name, self.model_name))
        if self._fields[name] == Optional[Ref]:
            # inital setattr
            if not _hasattr(self, name):
                # default value
                if isinstance(value, Ref):
                    object.__setattr__(self, name, value)
                    return
                # override in __init__
                if isinstance(value, Model):
                    # first init the Ref instance
                    object.__setattr__(self, name, self.__dataclass_fields__[name].default_factory())
                    object.__getattribute__(self, name)._model = value
                    if self not in [bref._model for bref in getattr(value, f'backref_{self.model_name}')]:
                        getattr(value, f'backref_{self.model_name}').append(BackRef(_model=self, _model_name=self.model_name))
                    return
                else:
                    object.__setattr__(self, name, self.__dataclass_fields__[name].default_factory())

            if value is None:
                # if model is already set
                if object.__getattribute__(self, name)._model is not None:
                    # remove back reference from value model first
                    omodel = object.__getattribute__(self, name)._model
                    getattr(omodel, f'backref_{self.model_name}').remove(BackRef(_model=self, _model_name=self.model_name))
                object.__getattribute__(self, name)._model = value

                return
            if not isinstance(value, Model):
                raise TypeError("Cannot set attribute {} on model {} to value {} of type {}. Expected {}".format(name, self.model_name, value, type(value), Model))
            if value.model_name != object.__getattribute__(self, name)._model_name:
                raise TypeError("Cannot set attribute {} on model {} to value {} of model {}. Expected {}".format(name, self.model_name, value, value.model_name, self._fields[name].type_._model_name))

            # if model is already set
            if object.__getattribute__(self, name)._model is not None:
                # remove back reference from value model first
                omodel = object.__getattribute__(self, name)._model
                getattr(omodel, f'backref_{self.model_name}').remove(BackRef(_model=self, _model_name=self.model_name))
            object.__getattribute__(self, name)._model = value
            if self not in [bref._model for bref in getattr(value, f'backref_{self.model_name}')]:
                getattr(value, f'backref_{self.model_name}').append(BackRef(_model=self, _model_name=self.model_name))
            return
        else:
            return super().__setattr__(name, value)

    def __getattribute__(self, name):
        attr = object.__getattribute__(self, name)
        #raise AttributeError("Model {} has no attribute {}".format(object.__getattribute__(self, 'model_name'), name))
        if isinstance(object.__getattribute__(self, name), Ref):
            return object.__getattribute__(self, name)._model
        return attr



class DefObj(Base):
    pass


class DefStr(DefObj):
    _TNAME = "str"


defstr = DefStr()


class DefInt(DefObj):
    _TNAME = "int"


defint = DefInt()


class DefFloat(DefObj):
    _TNAME = "float"


deffloat = DefFloat()


class DefBool(DefObj):
    _TNAME = "bool"


defbool = DefBool()


class DefRef(DefObj):
    model_name: str


class DefList(DefObj):
    dtype: Optional[DefObj]


class DefDict(Base):
    dtype: Optional[DefObj]


class DefBackRef(DefObj):
    model_name: str
    field_name: str


class ModelDefinition(Base):
    model_name: str
    fields: TDict[str, DefObj]


class Modelizer(Base):
    definitions: TDict[str, ModelDefinition] = TDict[str, ModelDefinition]({})
    _models: TDict[str, Type[Model]] = TDict[str, Type[Model]]({})

    def new_definition(self, mod_name, fields: TDict[str, ANY]):
        self.definitions[mod_name] = ModelDefinition(model_name=mod_name, fields=fields)
        model_definition = self.definitions[mod_name]
        root = {"type": "ModelDefinition",
                "name": model_definition.model_name,
                "fields": {}}
        stack = []
        for fname, f in model_definition.fields.items():
            if isinstance(f, DefObj):
                stack.append({"fieldname": fname, "field": f})
        while stack:
            stack_entry = stack.pop()
            fieldname, field = stack_entry['fieldname'], stack_entry['field']
            if isinstance(field, DefRef):
                edef = self.definitions[field.model_name]
                edef.fields['backref_' + mod_name] = DefBackRef(model_name=mod_name, field_name=fieldname)
            elif isinstance(field, DefList):
                stack.append({"field": field.dtype, "fieldname": fieldname})
            elif isinstance(field, DefDict):
                stack.append({"field": field.dtype, "fieldname": fieldname})
        self.consistency_check()

    def definition_to_json(self, mod_name):
        model_definition = self.definitions[mod_name]
        root = {"type": "ModelDefinition",
                "name": model_definition.model_name,
                "fields": {}}
        stack = []
        for fname, f in model_definition.fields.items():
            stack.append({"current": f, "parent": root['fields'], "parent_key": fname})
        while stack:
            stack_entry = stack.pop()
            current, parent, parent_key = stack_entry['current'], stack_entry['parent'], stack_entry['parent_key']
            if isinstance(current, DefRef):
                parent[parent_key] = {"type": "Model",
                                      "name": current.model_name}
            elif isinstance(current, DefBackRef):
                parent[parent_key] = {"type": "BackRef",
                                      "name": current.model_name,
                                      "field_name": current.field_name}
            elif isinstance(current, DefList):
                parent[parent_key] = {"type": "List",
                                      "dtype": None}
                stack.append({"current": current.dtype, "parent": parent[parent_key], "parent_key": "dtype"})
            elif isinstance(current, DefDict):
                parent[parent_key] = {"type": "Dict",
                                      "dtype": None}
                stack.append({"current": current.dtype, "parent": parent[parent_key], "parent_key": "dtype"})
            else:
                parent[parent_key] = current._TNAME
        return root

    def definition_from_json(self, root):
        model_name = root['name']
        root_fields = root['fields']
        mod_fields = {}
        stack = []
        for fname, f in root_fields.items():
            stack.append({"current": f, "parent": mod_fields, "parent_key": fname, "stype": "field"})
        while stack:
            stack_entry = stack.pop()
            current, parent, parent_key, stype = stack_entry['current'], stack_entry['parent'], stack_entry['parent_key'], stack_entry['stype']
            if isinstance(parent, dict):
                _setter = lambda x, key, val: x.__setitem__(key, val)
            else:
                _setter = lambda x, key, val: x.__setattr__(key, val)
            if isinstance(current, dict):
                if current['type'] == 'Model':
                    _setter(parent, parent_key, DefRef(model_name=current['name']))
                elif current['type'] == 'BackRef':
                    _setter(parent, parent_key, DefBackRef(model_name=current['name'], field_name=current['field_name']))
                elif current['type'] in ('List', 'Dict'):
                    if current['type']:
                        dtypei = DefList(dtype=None)
                    else:
                        dtypei = DefDict(dtype=None)
                    #if stype == 'field':
                    _setter(parent, parent_key, dtypei)
                    #else:
                    #    setattr(parent, parent_key, dtypei)
                    stack.append({'current': current['dtype'], 'parent': dtypei, 'parent_key': 'dtype', 'stype': 'dtype'})
            else:
                if current == 'str':
                    #parent[parent_key] = defstr
                    _setter(parent, parent_key, defstr)
                elif current == 'int':
                    #parent[parent_key] = defint
                    _setter(parent, parent_key, defint)
                elif current == 'float':
                    #parent[parent_key] = deffloat
                    _setter(parent, parent_key, deffloat)
                elif current == 'bool':
                    #parent[parent_key] = defbool
                    _setter(parent, parent_key, defbool)

        return ModelDefinition(model_name=model_name, fields=TDict[str, DefObj](mod_fields))

    def consistency_check(self):
        found_definitions = []
        found_references = {}

        for mod_name, model_definition in self.definitions.items():
            found_definitions.append(mod_name)
            root = {"type": "ModelDefinition",
                    "name": model_definition.model_name,
                    "fields": {}}
            stack = []
            for fname, f in model_definition.fields.items():
                stack.append({"current": f, "parent": root['fields'], "parent_key": fname})
            while stack:
                stack_entry = stack.pop()
                current, parent, parent_key = stack_entry['current'], stack_entry['parent'], stack_entry['parent_key']
                if isinstance(current, DefRef):
                    parent[parent_key] = {"type": "Model",
                                          "name": current.model_name}
                    found_references[current.model_name] = model_definition.model_name
                elif isinstance(current, DefList):
                    parent[parent_key] = {"type": "List",
                                          "dtype": None}
                    stack.append({"current": current.dtype, "parent": parent[parent_key], "parent_key": "dtype"})
                elif isinstance(current, DefDict):
                    parent[parent_key] = {"type": "Dict",
                                          "dtype": None}
                    stack.append({"current": current.dtype, "parent": parent[parent_key], "parent_key": "dtype"})

            missing = []
            for found_ref, ref_model in found_references.items():
                if found_ref not in found_definitions:
                    missing.append((found_ref, ref_model))
            return missing

    def model_from_schema(self, model_name):
        model_definition = self.definitions[model_name]
        annot = {}
        stack = []
        order = []
        for fname, ftype in model_definition.fields.items():
            if isinstance(ftype, DefStr):
                annot[fname] = str
            if isinstance(ftype, DefInt):
                annot[fname] = int
            if isinstance(ftype, DefBool):
                annot[fname] = bool
            if isinstance(ftype, DefFloat):
                annot[fname] = float
            if isinstance(ftype, DefRef):
                annot[fname] = Optional[Ref]
            elif isinstance(ftype, DefBackRef):
                annot[fname] = TList[BackRef]
            elif isinstance(ftype, DefList):
                order.insert(0, {"type": TList, "children": [], "parent": annot, 'key': fname})
                stack.append({"parent": annot, "key": fname, "current": ftype.dtype, "order_parent": order[0]})
            elif isinstance(ftype, DefDict):
                order.insert(0, {"type": TDict, "children": [str], "parent": annot, 'key': fname})
                stack.append({"parent": annot, "key": fname, "current": ftype.dtype, "order_parent": order[0]})

        while stack:
            s_entry = stack.pop()
            parent, parent_key, current, order_parent = s_entry['parent'], s_entry['key'], s_entry['current'], s_entry['order_parent']
            if isinstance(s_entry['current'], DefStr):
                order_parent['children'].append({"final_type": str})
            if isinstance(current, DefInt):
                order_parent['children'].append({"final_type": int})
            if isinstance(current, DefBool):
                order_parent['children'].append({"final_type": bool})
            if isinstance(current, DefFloat):
                order_parent['children'].append({"final_type": float})
            if isinstance(current, DefRef):
                order_parent['children'].append({"final_type": Ref})
            elif isinstance(current, DefBackRef):
                order_parent['children'].append({"final_type": BackRef})
            elif isinstance(current, DefList):
                order_parent['children'].append({"type": TList, "children": []})
                order.insert(0, order_parent['children'][-1])
                stack.append({"parent": None,
                              "key": 0, "current": ftype.dtype, "order_parent": order[0]})
            elif isinstance(current, DefDict):
                order_parent['children'].append({"type": TDict, "children": [str]})
                order.insert(0, order_parent['children'][-1])
                stack.append({"parent": None,
                              "key": 0, "current": ftype.dtype, "order_parent": order[0]})

        while order:
            order_entry = order.pop(0)
            if order_entry['children']:
                order_entry['final_type'] = order_entry['type'].__class_getitem__(*[ch['final_type']  for ch in order_entry['children']])
            if 'key' in order_entry:
                order_entry['parent'][order_entry['key']] = Optional[order_entry['final_type']]

        annot['model_name'] = str
        annot['uid'] = str
        defaults = {}
        for aname, atype in annot.items():
            if atype.__class__ == _UnionGenericAlias and atype.__args__ == (ANY, type(None)) and Ref not in atype.__args__:
                defaults[aname] = None
            elif TypeNode.from_type(atype) == TypeNode.from_type(TList[ANY]):
                defaults[aname] = dataclasses.field(default_factory=atype)
            elif TypeNode.from_type(atype) == TypeNode.from_type(TDict[ANY]):
                defaults[aname] = dataclasses.field(default_factory=atype)
            elif TypeNode.from_type(atype) == TypeNode.from_type(Optional[Ref]):
                defaults[aname] = dataclasses.field(default_factory=RefDefault(model_definition.fields[aname].model_name))
        attrs = {"__annotations__": annot,
                 'model_name': model_name,
                 "uid": dataclasses.field(default_factory=lambda: str(uuid4().hex))
        }
        attrs.update(defaults)
        self._models[model_name] = type(
            model_name, (Model,), attrs
        )
        return self._models[model_name]


class ModelStore(Base):
    _server: Optional[couchdb.Server] = None
    _db: Optional[couchdb.Database] = None
    uri: str
    modelizer: Modelizer

    @property
    def server(self):
        if self._server is None:
            self._server = couchdb.Server(self.uri)
        return self._server

    @property
    def db(self):
        if not self._db:
            try:
                self._db = self.server.create("models")
            except Exception as e:
                print(e)
                pass
            self._db = self.server["models"]
        return self._db

    def store_model_definition(self, model_name):
        json_def = self.modelizer.definition_to_json(model_name)
        self.db.create(
            {"type": "ModelDefinition",
             "name": model_name,
             "definition": json_def}
        )


    def store_model(self, model: Model):
        m_out = {}
        out = {"type": "Model", "model_name": model.model_name, "data": m_out}
        stack = []
        for f, ftype in model._fields.items():
            if TypeNode.from_type(ftype) == TypeNode(TList[ANY]):
                m_out[f] = [None]*len(getattr(model, f))
                for n, x in enumerate(getattr(model, f)):
                    stack.append((x, n, m_out[f]))
            elif TypeNode.from_type(ftype) == TypeNode(TDict[str, ANY]):
                m_out[f] = {}
                for k, v in getattr(model, f).items():
                    stack.append((v, k, m_out[f]))
            elif TypeNode.from_type(ftype) == TypeNode(Ref):
                if getattr(model, f) is not None:
                    m_out[f] = getattr(model, f).uid
                else:
                    m_out[f] = None
            else:
                m_out[f] = getattr(model, f)

        while stack:
            current, parent_key, c_out = stack.pop()
            if TypeNode.from_type(type(current)) == TypeNode(TList[ANY]):
                c_out[parent_key] = [None,]* len(current)
                for n, x in enumerate(current):
                    stack.append((x, n, c_out[parent_key]))
            elif TypeNode.from_type(type(current)) == TypeNode(TDict[str, ANY]):
                c_out[parent_key] = {}
                for k, v in current.items():
                    stack.append((v, k, c_out[parent_key]))
            elif isinstance(current, Model):
                if current:
                    c_out[parent_key] = current.uid
                else:
                    c_out[parent_key] = None
            elif isinstance(current, BackRef):
                c_out[parent_key] = current._model.uid
            else:
                c_out[parent_key] = current
        out['_id'] = out['data']['uid']
        _id, rev = self.db.save(out)
        model.uid = _id

    def load_model_definition(self, mod_name):
        model_definition_dict = next(self.db.find({'selector': {'type': 'ModelDefinition', 'name': mod_name}}))
        definition = self.modelizer.definition_from_json(model_definition_dict['definition'])
        self.modelizer.new_definition(mod_name, definition.fields)
        return definition

    def load_model(self, uid: str, max_depth: int) -> Model:
        mdict = self.db.get(uid)
        model_definition = self.load_model_definition(mdict['model_name'])
        i = 0
        mfields = mdict['data']
        stack = [{"fields": mfields, "parent": None, "parent_key": None, "type": model_definition}]
        pre_order = [stack[0]]
        loaded_uids = {}
        backrefs = {}

        while stack:
            stack_entry = stack.pop()
            if isinstance(stack_entry['type'], ModelDefinition):
                for fname, ftype in stack_entry['type'].fields.items():
                    if ftype not in (str, int, bool, float):
                        stack.append(
                            {"fields": stack_entry['fields'][fname],
                             "parent": stack_entry['fields'],
                             "parent_key": fname,
                             "type": ftype
                             }
                        )
            elif TypeNode.from_type(stack_entry['type']) == TypeNode.from_type(TDict[str, ANY]):
                for k, v in stack_entry['fields'].items():
                    stack.append(
                        {"fields": stack_entry['fields'][k],
                         "parent": stack_entry['fields'],
                         "parent_key": k,
                         "type": stack_entry['type']._args[1]
                         }
                    )
                    pre_order.insert(0, stack[-1])
            elif TypeNode.from_type(stack_entry['type']) == TypeNode.from_type(TList[ANY]):
                for n, x in enumerate(stack_entry['fields']):
                    stack.append(
                        {"fields": stack_entry['fields'][n],
                         "parent": stack_entry['fields'],
                         "parent_key": n,
                         "type": stack_entry['type']._args[0]
                         }
                    )
                    pre_order.insert(0, stack[-1])
            elif isinstance(stack_entry['type'], DefRef):
                if not stack_entry['fields']:
                    continue
                if stack_entry['fields'] not in loaded_uids:
                    rdict = self.db.get(stack_entry['fields'])
                    loaded_uids[stack_entry['fields']] = rdict
                    rmodel_definition = self.load_model_definition(rdict['model_name'])
                    rfields = rdict['data']
                    stack.append(
                        {"fields": rfields,
                         "parent": stack_entry['parent'],
                         "parent_key": stack_entry['parent_key'],
                         "type": rmodel_definition}
                    )
                    pre_order.insert(0, stack[-1])
                else:
                    rdict = loaded_uids[stack_entry['fields']]
                    rmodel_definition = self.load_model_definition(rdict['model_name'])
                    rfields = rdict['data']
                    pre_order.insert(0,
                        {"fields": rfields,
                         "parent": stack_entry['parent'],
                         "parent_key": stack_entry['parent_key'],
                         "type": rmodel_definition}
                    )
            elif isinstance(stack_entry['type'], DefBackRef):
                if stack_entry['fields'] is None:
                    continue
                rmodel_definition = self.load_model_definition(stack_entry['type'].model_name)
                #rfields = rdict['data']
                for n, uid in enumerate(stack_entry['fields']):
                    if uid not in loaded_uids:
                        rdict = self.db.get(uid)
                        stack.append(
                            {"fields": rdict['data'],
                             "parent": stack_entry['fields'],
                             "parent_key": n,
                             "type": rmodel_definition
                             }
                        )
                        pre_order.insert(0, stack[-1])
                        loaded_uids[uid] = rdict
                        backrefs.setdefault(stack_entry['parent']['uid'], {})
                        backrefs[stack_entry['parent']['uid']].setdefault(stack_entry['parent_key'], {})
                        backrefs[stack_entry['parent']['uid']][stack_entry['parent_key']] = {n: uid}
                    else:
                        rdict = loaded_uids[uid]
                        backrefs.setdefault(stack_entry['parent']['uid'], {})
                        backrefs[stack_entry['parent']['uid']].setdefault(stack_entry['parent']['parent_key'], {})
                        backrefs[stack_entry['parent']['uid']][stack_entry['parent']['parent_key']] = {n: uid}
                stack_entry['parent'][stack_entry['parent_key']] = TList[BackRef]()
                    

        root_model = None
        loaded_models = {}
        model_by_uid = {}
        while pre_order:
            entry = pre_order.pop(0)
            if isinstance(entry['type'], ModelDefinition):
                modelcls = self.modelizer.model_from_schema(entry['type'].model_name)
                if entry['parent']:
                    if entry['fields']['uid'] in model_by_uid:
                        model = model_by_uid[entry['fields']['uid']]
                    else:
                        model=modelcls(**entry['fields'])
                        model_by_uid[entry['fields']['uid']] = model
                    loaded_models[model.uid] = model
                    entry['parent'][entry['parent_key']] =  Ref(_model=model, _model_name=entry['type'].model_name)
                else:
                    root_model=modelcls(**entry['fields']) 
            elif TypeNode.from_type(entry['type']) == TypeNode.from_type(TList[ANY]):
                tlist = entry['type'](entry['fields'])
                entry['parent'][entry['parent_key']] = tlist
            elif TypeNode.from_type(entry['type']) == TypeNode.from_type(TDict[str, ANY]):
                tdict = entry['type'](entry['fields'])
                entry['parent'][entry['parent_key']] = tdict
            else:
                if entry['parent']:
                    entry['parent'][entry['parent_key']] = entry['fields']

        for puid, bfields in backrefs.items():
            pmodel = loaded_models[puid]
            for bfield, brefs in bfields.items():
                pfield = getattr(pmodel, bfield)
                for n, refuid in sorted(brefs.items(), key=lambda x: x[0]):
                    if refuid in loaded_models:
                        refmodel = loaded_models[refuid]
                        pfield.append(BackRef(_model=refmodel, _model_name=refmodel.model_name))
                    else:
                        pfield.append(BackRef(uid=refuid))

        return (root_model, loaded_models)


modelizer = Modelizer()

modelizer.new_definition("Date", fields=TDict[str, ANY]({
        "date": defstr
    }))
modelizer.new_definition("Organization", fields=TDict[str, ANY]({
        "name": defstr,
        "created": DefRef(model_name="Date")
    }))
modelizer.new_definition('Person', fields=TDict[str, ANY]({
        "name": defstr,
        "organization": DefRef(model_name="Organization"),
        "created": DefRef(model_name="Date")
}))
modelizer.new_definition('Project', fields=TDict[str, ANY]({
        "name": defstr,
        "organization": DefRef(model_name="Organization"),
        "created": DefRef(model_name="Date"),
        "engineers": DefList(dtype=DefRef(model_name="Person"))
}))

Project = modelizer.model_from_schema('Project')
Person = modelizer.model_from_schema('Person')
Organization = modelizer.model_from_schema('Organization')
Date = modelizer.model_from_schema('Organization')

print(Date._fields.items())
print("--")

org1 = Organization(name='Umbrela')
org2 = Organization(name='RainCoat')
p1 = Person(name='John Doe', organization=org1)
p1.organization = org2
print(p1)

model_store = ModelStore(uri="http://admin:password@localhost:5984", modelizer=modelizer)
model_store.store_model_definition("Date")
model_store.store_model_definition("Project")
model_store.store_model_definition("Person")
model_store.store_model_definition("Organization")
model_store.store_model(p1)
model_store.store_model(org1)
model_store.store_model(org2)

p1loaded, other_models = model_store.load_model(p1.uid, max_depth=10)
print(p1loaded)
print(p1==p1loaded)

