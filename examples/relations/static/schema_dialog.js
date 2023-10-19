
function createModelSearch(models, onclick_callback) {
  model_div = document.createElement('div');
  model_div.classList.add('search-inner-dialog');
  //model_div.style = 'margin-left: 10px; margin-right: 10px; background: red';
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
          const button = document.createElement('button');
          button.textContent = item;
          button.addEventListener('click', onclick_callback);
          button.classList.add('label-button');
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

function makeCloseButton(onclick_callback) {
  closeb = document.createElement('button');
  closeb.textContent = 'x';
  closeb.addEventListener('click', onclick_callback);
  closeb.classList.add('close-button');
  return closeb
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

    closeb = makeCloseButton((e) => {
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
      callback(schemaForm, schema, 'mod:'+e.target.textContent);
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
  closeb = makeCloseButton((e) => { document.body.removeChild(dialog)});
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
  closeb = makeCloseButton((e) => { document.body.removeChild(dialog)});
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
      return {'type': 'list', 'item_type': null};
    case 'dict':
      return {'type': 'dict', 'item_type': null};
    default:
      return vtype;
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
    valuebutton = document.createElement('button');
    valuebutton.classList.add('label-button');
    buttonspan.appendChild(valuebutton);

    valuebutton.textContent = '+';
    valuebutton.addEventListener('click', (e) => {
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
  entry = {name: () => {return entrylabel.value}, type: viewType2JsonType(type)};
  schema.fields.push(entry);

  entrydiv = document.createElement('div');
  entrylabel = document.createElement('input');
  entrylabel.type = 'text';
  entrylabel.placeholder = 'Field name';
  entrylabel.style = 'margin-right: 10px; margin-left: 10px;';


  if (type != 'list' && type != 'map') {
    valuelabel = document.createElement('label');
    valuelabel.textContent = type;
  } else {
    valuelabel = document.createElement('div');
    valuetext = document.createElement('label');
    valuetext.textContent = type + ' of [';

    buttonspan = document.createElement('span');
    valuebutton = document.createElement('button');
    valuebutton.classList.add('label-button');
    buttonspan.appendChild(valuebutton);

    valuebutton.textContent = '+';
    valuebutton.addEventListener('click', (e) => {
      schm = createSchemaMenu(
        e.clientX, e.clientY, 
        ['int number', 'float number', 'text', 'switch', 'list', 'dict'], 
        models, 
        form, 
        schema, (form, schema, type) => {buttonspan.removeChild(valuebutton); createEntryType(buttonspan, schema.fields[schema.fields.length-1].type, type, models)});
      document.body.appendChild(schm);}
    );

    valuetext2 = document.createElement('label');
    valuetext2.textContent = ']';
    valuelabel.appendChild(valuetext);
    valuelabel.appendChild(buttonspan);
    valuelabel.appendChild(valuetext2);
  }

  removebutton = document.createElement('button');
  removebutton.textContent = '-';
  removebutton.classList.add('label-button');
  removebutton.addEventListener('click', (e) => {
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

function validateSchema(schema, models) {
  const schema_name = schema.name();
  if (schema_name == '') {
    d = createErrorDialog('Model name cannot be empty');
    document.body.appendChild(d);
    throw new Error('Model name cannot be empty');
  }
  if (models.includes(schema_name)) {
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
  new_schema.fields = [];
  schema.fields.forEach((field) => {
    new_schema.fields.push({type: field.type, name: field.name()});
  });
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
        error_callback(data);
      }
      document.body.appendChild(createErrorDialog('Network Error'));
     console.log('There was a problem with the fetch operation:', error.message);
    })
}

function createNewSchemaDialog(schema, submit_action) {
  const dialog = document.createElement('div');
  dialog.classList.add('dialog');
  closediv = document.createElement('div');
  closediv.style.textAlign = 'right';
  closeb = makeCloseButton((e) => { document.body.removeChild(dialog)});
  closediv.appendChild(closeb);
  dialog.appendChild(closediv);

  const title = document.createElement('h3');
  title.textContent = 'Add model type';
  dialog.appendChild(title);

  const form = document.createElement('form');
  form.action = submit_action;
  form.method = 'POST';

  const plusdiv = document.createElement('div');
  plusdiv.style = 'text-align: right;';
  const plusbutton = document.createElement('button');
  plusbutton.textContent = '+';
  var models = [];
  plusbutton.addEventListener('click', (e) => {
     loadModels(models, 
		  (data) => {document.body.appendChild(createSchemaMenu(e.clientX, e.clientY, ['int number', 'float number', 'text', 'switch', 'list', 'map'], models, formdiv, schema, createSchemaEntry))}, 
		  (error) => {document.body.appendChild(createErrorDialog('Failed to load models'))});
  })

  plusbutton.classList.add('label-button');
  plusdiv.appendChild(plusbutton);
  const formdiv = document.createElement('div');
  formdiv.style = 'border: 1px solid black; padding: 10px; margin: 10px;';

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
  formdiv.appendChild(namediv);

  dialog.appendChild(plusdiv);

  const button = document.createElement('button');
  button.type = 'button';
  button.addEventListener('click', (e) => {
    console.log(schema);
    validateSchema(schema, models);
    const eschema = evalSchema(schema);
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
        document.body.removeChild(dialog);
      })
      .catch((error) => {
        createErrorDialog('Failed to create model "'+schema.name()+'"', 'error');
        console.error('Error:', error);
        document.body.removeChild(dialog);
      });
  });
  button.textContent = 'Add';
  button.classList.add('label-button');
  form.appendChild(button);

  dialog.appendChild(formdiv);
  dialog.appendChild(form);
  document.body.appendChild(dialog);
}


function createEditSchemaDialog(schema, submit_action) {
  const dialog = document.createElement('div');
  dialog.classList.add('dialog');
  closediv = document.createElement('div');
  closediv.style.textAlign = 'right';
  closeb = makeCloseButton((e) => { document.body.removeChild(dialog)});
  closediv.appendChild(closeb);
  dialog.appendChild(closediv);

  const title = document.createElement('h3');
  title.textContent = 'Edit model type';
  dialog.appendChild(title);

  const form = document.createElement('form');
  form.action = submit_action;
  form.method = 'POST';

  const plusdiv = document.createElement('div');
  plusdiv.style = 'text-align: right;';
  const plusbutton = document.createElement('button');
  plusbutton.textContent = '+';
  var models = [];
  plusbutton.addEventListener('click', (e) => {
     loadModels(models, 
		  (data) => {document.body.appendChild(createSchemaMenu(e.clientX, e.clientY, ['int number', 'float number', 'text', 'switch', 'list', 'map'], models, formdiv, schema, createSchemaEntry))}, 
		  (error) => {document.body.appendChild(createErrorDialog('Failed to load models'))});
  })

  plusbutton.classList.add('label-button');
  plusdiv.appendChild(plusbutton);
  const formdiv = document.createElement('div');
  formdiv.style = 'border: 1px solid black; padding: 10px; margin: 10px;';

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
  formdiv.appendChild(namediv);
  dialog.appendChild(plusdiv);

  const button = document.createElement('button');
  button.addEventListener('click', (e) => {
    console.log(schema);
    validateSchema(schema, models);
    const eschema = evalSchema(schema);
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
        document.body.removeChild(dialog);
      })
      .catch((error) => {
        createErrorDialog('Failed to create model "'+schema.name()+'"', 'error');
        console.error('Error:', error);
        document.body.removeChild(dialog);
      });
  });
  button.textContent = 'Save';
  button.classList.add('label-button');
  form.appendChild(button);

  dialog.appendChild(formdiv);
  dialog.appendChild(form);
  document.body.appendChild(dialog);
}

function modelsDialog(schema, submit_action) {
  const dialog = document.createElement('div');
  dialog.classList.add('dialog');
  closediv = document.createElement('div');
  closediv.style.textAlign = 'right';
  closeb = makeCloseButton((e) => { document.body.removeChild(dialog)});
  closediv.appendChild(closeb);
  dialog.appendChild(closediv);

  const title = document.createElement('h3');
  title.textContent = 'Models';
  dialog.appendChild(title);

  const formdiv = document.createElement('div');
  formdiv.style = 'border: 1px solid black; padding: 10px; margin: 10px;';


  function createModelEntry(model) {
    model_div = document.createElement('div');
    model_label = document.createElement('label');
    model_label.textContent = model;
    model_edit_button = document.createElement('button');
    model_edit_button.addEventListener('click', (e) => {
    fetch('/model_definition/' + model, 
      {
        method: 'GET',
	headers: {
	  'Content-Type': 'application/json'
	},
    })
      .then(response => response.json())  // assuming server responds with json
      .then(data => {
        createEditSchemaDialog(data, submit_action)
      })
      .catch((error) => {
        createErrorDialog('Failed to load model definition"' + schema.name() + '"', 'error');
        console.error('Error:', error);
        document.body.removeChild(dialog);
      });
    });

    model_edit_button.textContent = 'Edit';
    model_edit_button.classList.add('label-button');
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

  formdiv.appendChild(model_search_div);
  formdiv.appendChild(model_results_div);

  loadModels(models, 
             (data)=>{ handleSearch()}, 
             (error)=>{});
  dialog.appendChild(formdiv);
  document.body.appendChild(dialog);
}
