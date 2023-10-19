from flask import Flask, jsonify, send_from_directory, request, abort
import couchdb
from marshmallow import Schema, fields, ValidationError
from marshmallow_union import Union

from relations import ModelStore, Modelizer, defint, defstr, defbool, deffloat, DefObj, TDict, DefRef, DefList, DefDict

app = Flask(__name__, static_url_path='/static')

modelizer = Modelizer()
model_store = ModelStore(uri="http://admin:password@localhost:5984", modelizer=modelizer)
existing_definitions = model_store.get_model_def_names()


def validate_field_name(key):
    import re
    pattern = re.compile(r'^([a-z_]+)([a-zA-Z0-9_*]+)$')
    if not pattern.match(key):
        raise ValidationError(f"Key '{key}' does not match the pattern.")


class ContainerSchema(Schema):
    item_type = Union(fields=[fields.String(required=True), fields.Nested(lambda: ContainerSchema())])
    type = fields.String(required=True)


class FieldSchema(Schema):
    type = Union(fields=[fields.String(required=False), fields.Nested(ContainerSchema)])
    name = fields.String(required=True, validate=validate_field_name)


class ModelDefinitionSchema(Schema):
    name = fields.String(required=True)
    fields = fields.List(fields.Nested(FieldSchema), required=True)


model_schema = ModelDefinitionSchema()


@app.route('/model_definitions', methods=['GET'])
def get_model_definitions():
    return jsonify(model_store.get_model_def_names())

def type_to_ftype(stype):
    
    parent = {}
    stack = [(stype, parent, 'root', 'dict')]
    while stack:
        _type, _parent, _key, _parent_type = stack.pop()
        print(_type, _parent, _key, _parent_type)
        if _parent_type == 'dict':
            setter = dict.__setitem__
            getter = dict.__getitem__
        else:
            setter = object.__setattr__
            getter = object.__getattribute__
            
        if _type == 'string':
            setter(_parent, _key, defstr)
        elif _type == 'bool':
            setter(_parent, _key, defbool)
        elif _type == 'float':
            setter(_parent, _key, deffloat)
        elif _type == 'int':
            setter(_parent, _key, defint)
        elif isinstance(_type, str) and _type.startswith('mod:'):
            setter(_parent, _key, DefRef(model_name=_type[4:]))
        elif isinstance(_type, dict):
            if _type['type'] == 'list':
                setter(_parent, _key, DefList(dtype=None))
            else:
                setter(_parent, _key, DefDict(dtype=None))
            stack.append((_type['item_type'], getter(_parent, _key), 'dtype', 'object'))

    return parent['root']


@app.route('/model_definition', methods=['POST'])
def post_model_definition():
    # Expecting JSON data in the POST request
    data = request.get_json()
    try:
        # Validate and deserialize input
        data = model_schema.load(data)
    except ValidationError as err:
        print("validation error", err.messages)
        return jsonify(err.messages), 400

    # Save the model definition to CouchDB
    print(data)
    fields = TDict[str, DefObj]({x['name']: type_to_ftype(x['type']) for x in data['fields']})
    modelizer.new_definition(data['name'], fields)
    model_store.store_model_definition(data['name'])
    return jsonify({"status": "success", "message": "Model definition saved successfully"}), 201

@app.route('/model_definition/<string:model_name>', methods=['GET'])
def get_model_definition(model_name):
    model_store.load_model_definition(model_name)
    data = modelizer.definition_to_json(model_name)
    return jsonify(data), 200


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'relations.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)


if __name__ == '__main__':
    model_schema.load({"name": "test", "fields": [{"name": "test", "type": "string"}]})
    app.run(debug=True)
