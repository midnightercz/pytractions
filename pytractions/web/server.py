import os

from flask import Flask, render_template, request

from ..catalog import catalog

app = Flask(__name__)
app.template_folder = os.path.dirname(__file__)


# Sample data

@app.route('/')
def index():
    cards = []
    dists, all_inputs, all_outputs, all_resources, all_args, all_tags = catalog()
    for dist in dists:
        for traction in dist['tractions']:
            card = {
                'name': traction['name'],
                'type': traction['type'],
                'module': traction['module'],
                'docs': traction['docs'],
                'tags': traction.get('tags'),
                'inputs': traction['inputs'],
                'outputs': traction['outputs'],
                'resources': traction['resources'],
                'args': traction['args']
            }
            cards.append(card)

    # Get filter parameters from query string
    name_filter = request.args.get('name')
    tag_filter = request.args.getlist('tags')
    inputs_filter = request.args.getlist('inputs')
    outputs_filter = request.args.getlist('outputs')
    args_filter = request.args.getlist('args')
    resources_filter = request.args.getlist('resources')
    
    # Filter cards based on parameters
    filtered_cards = filter(lambda card: 
                            (not name_filter or name_filter.lower() in card['name'].lower()) and
                            (not tag_filter or set(tag_filter).issubset(set(card['tags']))) and
                            (not inputs_filter or any(inp['name'] in inputs_filter for inp in card['inputs'])) and
                            (not outputs_filter or any(out['name'] in outputs_filter for out in card['outputs'])) and
                            (not args_filter or any(arg['name'] in args_filter for arg in card['args'])) and
                            (not resources_filter or any(res['name'] in resources_filter for res in card['resources'])),
                            cards)
    
    return render_template('index.html',
                           cards=filtered_cards,
                           all_inputs=all_inputs,
                           all_outputs=all_outputs, 
                           all_resources=all_resources,
                           all_args=all_args,
                           all_tags=all_tags)

if __name__ == '__main__':
    app.run(debug=True)
