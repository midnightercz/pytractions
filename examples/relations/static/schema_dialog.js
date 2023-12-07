function createModelSearch(models, onclick_callback) {
  model_div = document.createElement('div');
  model_div.classList.add('search-inner-dialog');
  model_search_div = document.createElement('div');
  model_search_div.classList.add('search-div');

  model_label = document.createElement('label');
  model_label.textContent = 'Model';
  model_search_input = document.createElement('input');
  model_search_input.type = 'text';
  model_search_input.placeholder = 'Search';

  model_search_div.appendChild(model_label);
  model_search_div.appendChild(model_search_input);

  model_results = document.createElement('div');
  model_results.classList.add('scroll-div');

  model_div.classList.add('search-inner-dialog');
  
  function handleSearch() {
      const query = model_search_input.value.toLowerCase();
      const filteredItems = models.filter(item => item.toLowerCase().includes(query));
      displayItems(filteredItems);
  }
  function displayItems(filteredItems) {
      model_results.innerHTML = ''; // Clear the list first
      for (const item of filteredItems) {
          const buttondiv = document.createElement('div');
          button = create_button(item, onclick_callback);
          buttondiv.appendChild(button);
          model_results.appendChild(buttondiv);
      }
  }
  model_search_input.addEventListener('input', handleSearch);
  model_div.appendChild(model_search_div);
  model_div.appendChild(model_results);
  handleSearch();
  return model_div;
}


function createSchemaMenu(x, y, options, models, schemaForm, schema, callback) {
    const menu = document.createElement('div');
    menu.style.left = x + "px";
    menu.style.top = y + "px";
    menu.id = 'contextMenu';
    menu.classList.add('floating-menu');

    const menuList = document.createElement('div');
    closediv = document.createElement('div');
    closediv.style.textAlign = 'right';
    closediv.style.marginBottom = '5px';

    closeb = create_close_button((e) => {
        document.body.removeChild(menu);
    });
    closediv.appendChild(closeb);
    menuList.appendChild(closediv);
    menu.appendChild(menuList);

    options.forEach(option => {
        const listItem = document.createElement('button');
        listdiv = document.createElement('div');
        listItem.classList.add('label-button');
        listItem.textContent = option;
        listItem.addEventListener('click', (e) => {
            document.body.removeChild(menu);
            callback(schemaForm, schema, e.target.textContent, models);
        });
        listdiv.appendChild(listItem);
        menuList.appendChild(listdiv);
    });

  model_form = createModelSearch(models, (e) => {
      document.body.removeChild(menu);
      callback(schemaForm, schema, {'type': 'Model', 'name': e.target.textContent});
    });
  menu.appendChild(model_form);
  return menu;
}


function createErrorDialog(error_message) {
  const dialog = document.createElement('div');
  dialog.classList.add('dialog');
  dialog.classList.add('error');
  closediv = document.createElement('div');
  closediv.style.textAlign = 'right';
  closeb = create_close_button((e) => { document.body.removeChild(dialog)});
  closediv.appendChild(closeb);
  dialog.appendChild(closediv);
  const title = document.createElement('h3');
  title.textContent = 'Error';
  dialog.appendChild(title);
  label = document.createElement('label');
  label.textContent = error_message;
  dialog.appendChild(label);
  return dialog;
}

function createNotification(message, type='info') { 
  const dialog = document.createElement('div');
  closediv = document.createElement('div');
  closediv.style.textAlign = 'right';
  closeb = create_close_button((e) => { document.body.removeChild(dialog)});
  closediv.appendChild(closeb);
  dialog.appendChild(closediv);
  dialog.classList.add('notification');
  dialog.classList.add(type);
  text = document.createElement('label');
  text.textContent = message;
  dialog.appendChild(text);
  setTimeout(function() {
    dialog.style.opacity = '0';
      setTimeout(function() {
        document.body.removeChild(dialog);
      }, 1000);
    }, 5000);
  return dialog;
}

function viewType2JsonType(vtype){
  switch (vtype) {
    case 'int number':
      return 'int';
    case 'float number':
      return 'float';
    case 'text':
      return 'string';
    case 'switch':
      return 'bool';
    case 'list':
      return {'type': 'List', 'dtype': null};
    case 'dict':
      return {'type': 'Dict', 'dtype': null};
    default:
      return vtype;
  }
}

function jsonType2ViewType(vtype){
  console.log('json2viewtype')
  console.log(vtype);
  switch (vtype) {
    case 'int':
      return 'int number';
    case 'float':
      return 'float number';
    case 'str':
      return 'text';
    case 'bool':
      return 'switch';
  }

  if (typeof vtype == 'object') {
    console.log('vtype ' + vtype);
    if (vtype.type == 'List') {
      return 'List[' + jsonType2ViewType(vtype.dtype) + ']';
    }
    if (vtype.type == 'Dict') {
      return 'Dict[' + jsonType2ViewType(vtype.dtype) + ']';
    }
    if (vtype.type == 'Model') {
      return 'mod:' + vtype.name;
    }
  }
}

function createEntryType(parent, schema, type, models) {
  schema.item_type = viewType2JsonType(type);

  if (type != 'list' && type != 'map') {
    valuelabel = document.createElement('label');
    valuelabel.textContent = type;
  } else {
    valuelabel = document.createElement('span');
    valuetext = document.createElement('label');
    valuetext.textContent = type + ' of [';
    
    buttonspan = document.createElement('span');
    valuebutton = create_button(
      '+',
      (e) => {
        schm = createSchemaMenu(
          e.clientX, e.clientY, 
          ['int number', 'float number', 'text', 'switch', 'list', 'dict'], 
          models, 
          buttonspan, 
          schema, (form, schema, type) => {buttonspan.removeChild(valuebutton); createEntryType(buttonspan, schema.item_type, type, models)});
        document.body.appendChild(schm);}
    );

    valuetext2 = document.createElement('label');
    valuetext2.textContent = ']';
    valuelabel.appendChild(valuetext);
    valuelabel.appendChild(buttonspan);
    valuelabel.appendChild(valuetext2);
  }
  parent.appendChild(valuelabel);
}


function createSchemaEntry(form, schema, type, models) {

  console.log("SCHEMA", schema);
  console.log("TYPE", type);
  entrydiv = document.createElement('div');
  var entrylabel = document.createElement('input');
  entrylabel.type = 'text';
  entrylabel.placeholder = 'Field name';
  entrylabel.style = 'margin-right: 10px; margin-left: 10px;';

  entry = {name: () => {return entrylabel.value}, type: viewType2JsonType(type)};
  schema.fields.push(entry);


  if (type.type != 'list' && type.type != 'map' && type.type != 'Model') {
    valuelabel = document.createElement('label');
    valuelabel.textContent = type;
  } else if (type.type == 'Model') {
    valuelabel.textContent = 'mod:'+type.name;
  } else {
    valuelabel = document.createElement('div');
    valuetext = document.createElement('label');
    valuetext.textContent = type + ' of [';

    buttonspan = document.createElement('span');
    valuebutton = create_button(
      valuebutton, 
      (e) => {
        schm = createSchemaMenu(
          e.clientX, e.clientY, 
          ['int number', 'float number', 'text', 'switch', 'list', 'dict'], 
          models, 
          form, 
          schema, (form, schema, type) => {buttonspan.removeChild(valuebutton); createEntryType(buttonspan, schema.fields[schema.fields.length-1].type, type, models)});
        document.body.appendChild(schm);
      }
    );

    valuetext2 = document.createElement('label');
    valuetext2.textContent = ']';
    valuelabel.appendChild(valuetext);
    valuelabel.appendChild(buttonspan);
    valuelabel.appendChild(valuetext2);
  }

  remove_button = create_button(
    '-', 
    (e) => {
      form.removeChild(entrydiv);
      schema.fields = schema.fields.filter((field) => {
        field.name != entrylabel.value;
      });
  });
  entrydiv.appendChild(removebutton);
  entrydiv.appendChild(entrylabel);
  entrydiv.appendChild(valuelabel);
  form.appendChild(entrydiv);
}

function validateSchema(schema, models, update=false) {
  const schema_name = schema.name();
  if (schema_name == '') {
    d = createErrorDialog('Model name cannot be empty');
    document.body.appendChild(d);
    throw new Error('Model name cannot be empty');
  }
  if (!update && models.includes(schema_name)) {
    d = createErrorDialog('Model name "'+ schema_name +'" already exists');
    document.body.appendChild(d);
    throw new Error('Model name "'+ schema_name +'" already exists');
  }
  if (/^([a-zA-Z]+)([a-zA-Z0-9_]*)$/.exec(schema_name) === null) {
    d = createErrorDialog("Model name '" + schema_name + "' doesn't match allowed pattern ([a-z]+)([a-zA-Z0-9_])*");
    document.body.appendChild(d);
    throw new Error('Field name cannot be empty');
  }
  if (schema.fields.length == 0) {
      d = createErrorDialog('Model must have at least one field');
      document.body.appendChild(d);
      throw new Error('Model must have at least one field');
  }

  schema.fields.forEach((field) => {
    if (field.name() == '') {
      d = createErrorDialog('Field name cannot be empty');
      document.body.appendChild(d);
      throw new Error('Field name cannot be empty');
    }
    if (/^([a-z]+)([a-zA-Z0-9_]*)$/.exec(field.name()) === null) {
      d = createErrorDialog("Field name '"+field.name()+"' doesn't match allowed pattern ([a-z]+)([a-zA-Z0-9_])*");
      document.body.appendChild(d);
      throw new Error('Field name cannot be empty');
    }
  });
}

function evalSchema(schema) {
  const new_schema = {};
  new_schema.name = schema.name();
  new_schema._id = schema._id;
  new_schema.fields = [];
  schema.fields.forEach((field) => {
    console.log('eval field');
    console.log(field);
    new_schema.fields.push({type: field.type, name: field.name()});
  });
  console.log(new_schema);
  return new_schema;
}

function loadModels(models, success_callback, error_callback) {
  models.length = 0;
  fetch('/model_definitions')
    .then(response => {
      if (!response.ok) {
        document.body.appendChild(createErrorDialog('Network Error'));
        throw new Error('Network response was not ok');
      }
      return response.json()
    }).then(data => {
      data.forEach((model) => {
        models.push(model);
      })
      if (success_callback != undefined){
        success_callback(data);
      }
    }).catch(error => {
      if (error_callback != undefined){
        error_callback(error);
      }
      document.body.appendChild(createErrorDialog('Network Error'));
     console.log('There was a problem with the fetch operation:', error.message);
    })
}

function createNewSchemaDialog(schema, submit_action) {
  dialog = create_dialog_plane('Create new model type', has_close_button=true);
  const plusdiv = document.createElement('div');
  plusdiv.style = 'text-align: right;';
  const plusbutton = create_button(
    '+',
    (e) => {
       loadModels(models, 
        (data) => {document.body.appendChild(createSchemaMenu(e.clientX, e.clientY, ['int number', 'float number', 'text', 'switch', 'list', 'map'], models, dialog.body, schema, createSchemaEntry))}, 
        (error) => {document.body.appendChild(createErrorDialog('Failed to load models'))});
  });
  plusdiv.appendChild(plusbutton);
  dialog.head.appendChild(plusdiv);

  const namediv = document.createElement('div');
  namediv.style = 'margin-bottom: 10px;';
  const namelabel = document.createElement('label');
  namelabel.textContent = 'Name';
  namelabel.style = 'margin-left: 10px; margin-right: 10px;';
  namediv.appendChild(namelabel);
  const nameinput = document.createElement('input');
  nameinput.type = 'text';
  nameinput.placeholder = 'Model Name';
  schema.name = () => { return nameinput.value; };

  namediv.appendChild(nameinput);
  dialog.body.appendChild(namediv);

  const button = create_button(
    'Add',
    (e) => {
      validateSchema(schema, models);
      const eschema = evalSchema(schema);
      console.log('eschema');
      console.log(eschema);
      fetch('/model_definition', 
        {
          method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(eschema)
        })
        .then(response => response.json())  // assuming server responds with json
        .then(data => {
          console.log('Success:', data);
          notification = createNotification('Model "'+schema.name()+'" created');
          loadModels(models);
          document.body.appendChild(notification);
          document.body.removeChild(dialog.root);
        })
        .catch((error) => {
          createErrorDialog('Failed to create model "'+schema.name()+'"', 'error');
          console.error('Error:', error);
          document.body.removeChild(dialog.root);
        });
  });
  dialog.body.appendChild(button);
  document.body.appendChild(dialog.root);
}


function createEditSchemaDialog(schema, submit_action) {
  console.log(schema);
  const models = [];
  dialog = create_dialog_plane('Edit model type', has_close_button=true);
  const plusdiv = document.createElement('div');
  plusdiv.style = 'text-align: right;';
  plusbutton = create_button(
    '+',
    (e) => {
       loadModels(models, 
        (data) => {document.body.appendChild(createSchemaMenu(e.clientX, e.clientY, ['int number', 'float number', 'text', 'switch', 'list', 'map'], models, dialog.body, schema, createSchemaEntry))}, 
        (error) => {document.body.appendChild(createErrorDialog('Failed to load models'))});
    }
  );
  plusdiv.appendChild(plusbutton);
  const namediv = document.createElement('div');
  namediv.style = 'margin-bottom: 10px;';
  const namelabel = document.createElement('label');
  namelabel.textContent = 'Name';
  namelabel.style = 'margin-left: 10px; margin-right: 10px;';
  namediv.appendChild(namelabel);
  const namevalue = document.createElement('label');
  namevalue.type = 'text';
  namevalue.textContent = schema.name;
  namediv.appendChild(namevalue);
  dialog.body.appendChild(namediv);
  dialog.head.appendChild(plusdiv);

  schema.fields.forEach((n) => {

    field = n.type;
    const field_name = n.name;
    n.name = ()=>field_name;

    entrydiv = document.createElement('div');
    entrylabel = document.createElement('label');
    entrylabel.type = 'text';
    entrylabel.textContent = field_name
    entrylabel.style = 'margin-right: 10px; margin-left: 10px;';
    valuelabel = document.createElement('label');
    valuelabel.textContent = jsonType2ViewType(field);
    removebutton = create_button(
      '-',
      (e) => {
        dialog.body.removeChild(entrydiv);
        schema.fields = schema.fields.filter((field) => {
          field.name != entrylabel.value;
        });
      }
    );
    entrydiv.appendChild(removebutton);
    entrydiv.appendChild(entrylabel);
    entrydiv.appendChild(valuelabel);
    dialog.body.appendChild(entrydiv);
  });


  button = create_button(
    'Update',
    (e) => {
      console.log(schema);
      const schema_name = schema.name;
      console.log("SCHEMA NAME" + schema_name);
      schema.name = () => schema_name;
      validateSchema(schema, models, update=true);
      const eschema = evalSchema(schema);
      fetch('/model_definition', 
        {
          method: 'PUT',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(eschema)
        })
        .then(response => response.json())  // assuming server responds with json
        .then(data => {
          console.log('Success:', data);
          notification = createNotification('Model "'+schema.name()+'" created');
          loadModels(models);
          document.body.appendChild(notification);
          document.body.removeChild(dialog.root);
        })
        .catch((error) => {
          createErrorDialog('Failed to create model "'+schema.name()+'"', 'error');
          console.error('Error:', error);
          document.body.removeChild(dialog.root);
        });
    }
  );
  dialog.tail.appendChild(button);
  document.body.appendChild(dialog.root);
}

function modelsDialog(schema, submit_action) {
  dialog = create_dialog_plane('Models', true);
  function createModelEntry(model) {
    model_div = document.createElement('div');
    model_label = document.createElement('label');
    model_label.textContent = model;
    model_edit_button = create_button(
      'Edit',
      (e) => {
        fetch('/model_definition/' + model, 
          {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json'
            },
          })
          .then(response => response.json())  // assuming server responds with json
          .then(data => {
            // JsonSchema2ViewSchema 
            createEditSchemaDialog(data, submit_action)
          })
          .catch((error) => {
            createErrorDialog('Failed to load model definition"' + schema + '"', 'error');
            console.error('Error:', error);
            document.body.removeChild(dialog);
          });
        }
    );
    model_remove_button = document.createElement('button');
    model_remove_button.textContent = 'Remove';
    model_remove_button.classList.add('label-button');
    model_div.appendChild(model_label);
    model_div.appendChild(model_edit_button);
    model_div.appendChild(model_remove_button);
    return model_div;
  }
  var models = [];

  function handleSearch() {
      const query = model_search_input.value.toLowerCase();
      const filteredItems = models.filter(item => item.toLowerCase().includes(query));
      displayItems(filteredItems);
  }
  function displayItems(filteredItems) {
      model_results_div.innerHTML = ''; // Clear the list first
      for (const item of filteredItems) {
        model_div = createModelEntry(item);
        model_results_div.appendChild(model_div);
      }
  }
  model_search_div = document.createElement('div');
  model_search_input = document.createElement('input');
  model_search_input.type = 'text';
  model_search_input.placeholder = 'Search';
  model_search_input.addEventListener('input', handleSearch);
  model_search_div.appendChild(model_search_input);
  model_results_div =document.createElement('div'); 

  dialog.body.appendChild(model_search_div);
  dialog.body.appendChild(model_results_div);

  loadModels(models, 
             (data)=>{ handleSearch()}, 
             (error)=>{});
  document.body.appendChild(dialog.root);
}
