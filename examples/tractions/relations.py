import couchdb
import dataclasses

from typing import Any

from typing import Optional, Union, Type, List, TypeVar, Generic, Dict

from pytraction.base import Base, Traction, TDict, TList, In, Out, Res, Arg, TypeNode, ANY
from pytraction.tractor import Tractor


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


class ModelStore(Base):
    _server: Optional[couchdb.Server] = None
    _db: Optional[couchdb.Database] = None
    uri: str

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
            except Exception:
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
            elif isinstance(f, Ref):
                if isinstance(f.model, str):
                    m_out[f] = getattr(model, f).model
                elif isinstance(f.model, Model):
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
            elif isinstance(current, Ref):
                if isinstance(current.model, str):
                    m_out[f] = getattr(current, f).model
                elif isinstance(current.model, Model):
                    m_out[f] = getattr(current, f).model.uid
                else:
                    m_out[f] = None

                c_out[parent_key] = current.model.uid
            else:
                c_out[parent_key] = current
        self.db.save(out)

    def load_model_definition(self, mod_name: str):
        model_definition_dict = self.db.find({'selector': {'type': 'ModelDefinition', 'name': mod_name}})[0]
        return ModelDefinition(
            model_name=model_definition_dict['name'],
            fields=TDict[str, ANY]({fname: TypeNode.from_json(ftype).to_type() for fname, ftype in model_definition_dict['fields'].items()})
        )

    def store_model_definition(self, model_definition):
        self.db.create(
            {"type": "ModelDefinition",
             "name": model_definition.model_name,
             "fields": {fname: TypeNode.from_type(ftype).to_json() for fname, ftype in model_definition.items()}}
        )

        return ModelDefinition(
            model_name=model_definition_dict['name'],
            fields=TDict[str, ANY]({fname: TypeNode.from_json(ftype).to_type() for fname, ftype in model_definition_dict['fields'].items()})
        )

    def load_model(self, uid: str, max_depth: int) -> Model:
        mdict = self.db.get(uid)
        model_definition = self.load_model_definition(mdict['model_name'])
        i = 0
        mfields = mdict['data']
        stack = [{"fields": mfields, "parent": None, "parent_key": None, "type": model_definition}]
        pre_order = []
        while stack:
            stack_entry = stack.pop()
            if TypeNode.from_type(stack_entry['type']) == TypeNode.from_type(ModelDefinition):
                for fname, ftype in model_definition._fields.items():
                    stack.append(
                        {"fields": efields[fname][n],
                         "parent": stack_entry['fields'],
                         "parent_key": fname,
                         "type": ftype
                         }
                    )
                    pre_order.insert(0, stack[-1])
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
                rdict = self.db.get(stack_entry['fields'])
                rmodel_definition = self.load_model_definition(mdict['model_name'])
                rfields = rdict['data']
                stack.append(
                    {"fields": rfields,
                     "parent": stack_entry['fields'],
                     "parent_key": stack_entry['parent_key'],
                     "type": rmodel_definition}
                )
                pre_order.insert(0, stack[-1])

        while pre_order:
            entry = pre_order.pop(0)
            if TypeNode.from_type(entry['type']) == TypeNode.from_type(ModelDefinition):
                modelcls = self.modelizer.model_from_definition(entry['type'])
                entry['parent'][entry['parent_key']] = modelcls(**entry['fields'])
            elif TypeNode.from_type(entry['type']) == TypeNode.from_type(TList[ANY]):
                tlist = entry['type'](entry['fields'])
                entry['parent'][entry['parent_key']] = tlist
            elif TypeNode.from_type(entry['type']) == TypeNode.from_type(TDict[str, ANY]):
                tdict = entry['type'](entry['fields'])
                entry['parent'][entry['parent_key']] = tdict
            else:
                entry['parent'][entry['parent_key']] = entry['fields']

        print(entry)


class Modelizer(Base):
    _models: Dict[str, Model] = dataclasses.field(default_factory=dict)

    def model_from_definition(self, definition: ModelDefinition) -> Type[Model]:
        if definition.model_name not in self._models.items():
            self._models[definition.model_name] = type(
                definition.model_name, (Model,), {"__annotations__": definition.fields}
                )
        return self._models[definition.model_name]


class ModelDefinitionToDbForm(Traction):
    i_model: In[ModelDefinition]
    o_out: Out[DB_COMPATIBLE] = Out[DB_COMPATIBLE](data=None)

    def _run(self, on_update=None):
        fields = {}
        out = {"type": "ModelDefinition", "name": self.i_model.data.name, "fields": fields}
        for fname, ftype in self.in_model.fields.items():
            fields[fname] = TypeNode.from_type(ftype).to_json()
        self.o_out.data = out


class ModelDefinitionFromDbForm(Traction):
    i_in: In[DB_COMPATIBLE]
    o_out: Out[ModelDefinition] = ModelDefinition(model_name="", fields=TDict[str, ANY]({}))

    def _run(self, on_update=None):
        self.o_out.data = ModelDefinition(
            name=self.i_in.data["name"],
            fields={fname: TypeNode.from_json(ftype).to_type() for fname, ftype in self.i_in.data["fields"].items()},
        )


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

model_store = ModelStore(uri="http://localhost:5984")
model_store.store_model_definition(DateDef)
model_store.store_model_definition(OrganizationDef)
model_store.store_model_definition(PersonDef)

