from uuid import uuid4

import dataclasses

from typing import Any

from typing import Optional, Union, Type, List, TypeVar, Generic, Dict

import pytraction.base
from pytraction.base import Base, Traction, TDict, TList, In, Out, Res, Arg, TypeNode, ANY
from pytraction.tractor import Tractor


import couchdb


T = TypeVar('T')


class Ref(Base, Generic[T]):
    model: Optional[T | str] = None


DB_COMPATIBLE = Union[None, int, float, str, bool, TList, TDict, Ref]


class Model(Base):
    model_name: str
    uid: str


class ModelDefinition(Base):
    model_name: str
    fields: TDict[str, ANY]


class Modelizer(Base):
    _models: Dict[str, Model] = dataclasses.field(default_factory=dict)

    def model_from_definition(self, definition: ModelDefinition) -> Type[Model]:
        if definition.model_name not in self._models:
            print("not cached", definition.model_name)
            annot = dict(definition.fields.items())
            annot['model_name'] = str
            annot['uid'] = str
            self._models[definition.model_name] = type(
                definition.model_name, (Model,), {
                    "__annotations__": annot,
                    'model_name': definition.model_name,
                    "uid": dataclasses.field(default_factory=lambda: str(uuid4().hex))}
            )
        mod = self._models[definition.model_name]
        setattr(pytraction.base, definition.model_name, mod)
        return mod


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

    def store_model(self, model: Model):
        m_out = {}
        out = {"type": "Model", "model_name": model.model_name, "data": m_out}
        stack = []
        for f, ftype in model._fields.items():
            if TypeNode.from_type(ftype) == TypeNode(TList[ANY]):
                m_out[f] = []
                for n, x in enumerate(getattr(model, f)):
                    stack.append((x, n, m_out[f]))
            elif TypeNode.from_type(ftype) == TypeNode(TDict[str, ANY]):
                m_out[f] = {}
                for k, v in getattr(model, f).items():
                    stack.append((v, k, m_out[f]))
            elif TypeNode.from_type(ftype) == TypeNode(Ref[ANY]):
                if isinstance(getattr(model, f).model, str):
                    m_out[f] = getattr(model, f).model
                elif isinstance(getattr(model, f).model, Model):
                    m_out[f] = getattr(model, f).model.uid
                else:
                    m_out[f] = None
            else:
                m_out[f] = getattr(model, f)

        while stack:
            current, parent_key, c_out = stack.pop()
            if TypeNode.from_type(type(current)) == TypeNode(TList[DB_COMPATIBLE]):
                c_out[parent_key] = []
                for n, x in enumerate(current):
                    stack.append((x, n, c_out[parent_key]))
            elif TypeNode.from_type(ftype) == TypeNode(TDict[str, DB_COMPATIBLE]):
                c_out[parent_key] = {}
                for k, v in current.items():
                    stack.append((v, k, c_out[parent_key]))
            elif TypeNode.from_type(ftype) == TypeNode(Ref[ANY]):
                if isinstance(current.model, str):
                    m_out[f] = getattr(current, f).model
                elif isinstance(current.model, Model):
                    m_out[f] = getattr(current, f).model.uid
                else:
                    m_out[f] = None

                c_out[parent_key] = current.model.uid
            else:
                c_out[parent_key] = current
        print("out", out)
        out['_id'] = out['data']['uid']
        _id, rev = self.db.save(out)
        model.uid = _id

    def load_model_definition(self, mod_name: str):
        model_definition_dict = next(self.db.find({'selector': {'type': 'ModelDefinition', 'name': mod_name}}))
        return ModelDefinition(
            model_name=model_definition_dict['name'],
            fields=TDict[str, ANY]({fname: TypeNode.from_json(ftype).to_type() for fname, ftype in model_definition_dict['fields'].items()})
        )

    def store_model_definition(self, model_definition):
        self.db.create(
            {"type": "ModelDefinition",
             "name": model_definition.model_name,
             "fields": {fname: TypeNode.from_type(ftype).to_json() for fname, ftype in model_definition.fields.items()}}
        )

    def load_model(self, uid: str, max_depth: int) -> Model:
        mdict = self.db.get(uid)
        model_definition = self.load_model_definition(mdict['model_name'])
        i = 0
        mfields = mdict['data']
        stack = [{"fields": mfields, "parent": None, "parent_key": None, "type": model_definition}]
        pre_order = [stack[0]]
        while stack:
            stack_entry = stack.pop()
            if isinstance(stack_entry['type'], ModelDefinition):
                #print("SE modddef", stack_entry)
                for fname, ftype in stack_entry['type'].fields.items():
                    if ftype not in (str, int, bool, float):
                        stack.append(
                            {"fields": stack_entry['fields'][fname],
                             "parent": stack_entry['fields'],
                             "parent_key": fname,
                             "type": ftype
                             }
                        )
                        #pre_order.insert(0, stack[-1])
            elif TypeNode.from_type(stack_entry['type']) == TypeNode.from_type(TDict[str, ANY]):
                for k, v in stack_entry['fields'][fname].items():
                    stack.append(
                        {"fields": stack_entry['fields'][fname][k],
                         "parent": stack_entry['fields'],
                         "parent_key": fname,
                         "type": stack_entry['type']._args[1]
                         }
                    )
                    pre_order.insert(0, stack[-1])
            elif TypeNode.from_type(stack_entry['type']) == TypeNode.from_type(TList[ANY]):
                for n, x in enumerate(stack_entry['fields'][fname]):
                    stack.append(
                        {"fields": stack_entry['fields'][fname][n],
                         "parent": stack_entry['fields'],
                         "parent_key": fname,
                         "type": stack_entry['type']._args[0]
                         }
                    )
                    pre_order.insert(0, stack[-1])
            elif TypeNode.from_type(stack_entry['type']) == TypeNode.from_type(Ref[ANY]):
                #print("SE ref", stack_entry)
                rdict = self.db.get(stack_entry['fields'])
                rmodel_definition = self.load_model_definition(rdict['model_name'])
                rfields = rdict['data']
                stack.append(
                    {"fields": rfields,
                     "parent": stack_entry['parent'],
                     "parent_key": stack_entry['parent_key'],
                     "type": rmodel_definition}
                )
                pre_order.insert(0, stack[-1])

        #print(pre_order)
        #print("--")
        root_model = None
        loaded_models = []
        model_by_uid = {}
        while pre_order:
            entry = pre_order.pop(0)
            #print(entry)
            if isinstance(entry['type'], ModelDefinition):
                #print("mod def")
                modelcls = self.modelizer.model_from_definition(entry['type'])
                if entry['parent']:
                    if entry['fields']['uid'] in model_by_uid:
                        model = model_by_uid[entry['fields']['uid']]
                    else:
                        model=modelcls(**entry['fields'])
                        model_by_uid[entry['fields']['uid']] = model
                    loaded_models.append(model)
                    entry['parent'][entry['parent_key']] =  Ref[modelcls](model=model)
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

        return (root_model, loaded_models)


class ModelToDbForm(Traction):
    i_model: In[Model]
    i_model_definition: In[ModelDefinition]
    o_model: Out[DB_COMPATIBLE] = Out[DB_COMPATIBLE](data=None)

    def _run(self, on_update=None):
        m_out = {}
        out = {"type": "Model", "model_name": self.i_model_definition.data.name, "data": m_out}
        stack = []
        for f, ftype in self.in_model.data._fields.items():
            if TypeNode.from_type(ftype) == TypeNode(TList[DB_COMPATIBLE]):
                m_out[f] = []
                for n, x in enumerate(getattr(self.in_model.data, f)):
                    stack.append((x, n, m_out[f]))
            elif TypeNode.from_type(ftype) == TypeNode(TDict[str, DB_COMPATIBLE]):
                m_out[f] = {}
                for k, v in getattr(self.in_model.data, f).items():
                    stack.append((v, k, m_out[f]))
            elif isinstance(f, Ref):
                m_out[f] = getattr(self.in_model.data, f).model.uid
            else:
                m_out[f] = getattr(self.in_model.data, f)

        while stack:
            current, parent_key, c_out = stack.pop()
            if TypeNode.from_type(type(current)) == TypeNode(TList[DB_COMPATIBLE]):
                c_out[parent_key] = []
                for n, x in enumerate(current):
                    stack.append((x, n, c_out[parent_key]))
            elif TypeNode.from_type(ftype) == TypeNode(TDict[str, DB_COMPATIBLE]):
                for k, v in current.items():
                    stack.append((v, k, c_out[parent_key]))
            elif isinstance(current, Ref):
                c_out[parent_key] = current.model.uid
            else:
                c_out[parent_key] = current

        self.out_model.data = out


modelizer = Modelizer()

DateDef = ModelDefinition(
    model_name="Date",
    fields=TDict[str, ANY]({
        "date": str
    })
)
Date = modelizer.model_from_definition(DateDef)

OrganizationDef = ModelDefinition(
    model_name="Organization",
    fields=TDict[str, ANY]({
        "name": str,
        "created": Ref[Date]
    })
)
Organization = modelizer.model_from_definition(OrganizationDef)

PersonDef = ModelDefinition(
    model_name="Person",
    fields=TDict[str, ANY]({
        "name": str,
        "organization": Ref[Organization],
        "created": Ref[Date],
    })
)
Person = modelizer.model_from_definition(PersonDef)

model_store = ModelStore(uri="http://admin:password@localhost:5984", modelizer=modelizer)
model_store.store_model_definition(DateDef)
model_store.store_model_definition(OrganizationDef)
model_store.store_model_definition(PersonDef)

print("--1--")

d1 = Date(date="2023-01-01T11:20:00")
o1 = Organization(name='Organization 1', created=Ref[Date](model=d1))
p1 = Person(name="John Doe", organization=Ref[Organization](model=o1), created=Ref[Date](model=d1))

model_store.store_model(d1)
model_store.store_model(o1)
model_store.store_model(p1)


p11 = model_store.load_model(p1.uid, 10)
print(p11)
