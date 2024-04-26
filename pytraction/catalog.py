import argparse
import re

import importlib.metadata

def filter_name(name_filter, name):
    if isinstance(name_filter, re.Pattern):
        return re.match(name_filter, name)
    return name_filter in name

def filter_tag(tag_filter, tags):
    if isinstance(tag_filter, re.Pattern):
        return any(re.match(tag_filter, t) for t in tags)
    return tag_filter in tags


ignore_types = (
    "TList", "TDict", "Union", "Optional", "In", "Res", "Out", "TRes", "Arg", "STMDSingleIn", "ABase", "TIn"
)

def gather_simple_types(type_json):

    simple_types = []
    print(type_json)
    if type_json['type']['$type']['type'] not in ignore_types:
        simple_types.append(type_json['type']['$type']['type'])
    stack = []
    for arg in type_json['type']['$type'].get('args', []):
        stack.append(arg)
    while stack:
        arg = stack.pop()
        if arg['type'] not in ignore_types:
            simple_types.append(arg['type'])
        for a in arg.get('args', []):
            stack.append(a)
    print(simple_types)
    return list(set(simple_types))


def tractions_discovery(tag_filter=None, type_filter=None, name_filter=None):
    ret = []
    seen = set()
    for d in importlib.metadata.distributions():
        if d.entry_points.select(group='tractions'):
            if d._path in seen:
                continue
            seen.add(d._path)
            dist = inspect_distribution(d, tag_filter=None, type_filter=None, name_filter=None)
            if dist:
                ret.append(dist)
    return ret


def inspect_traction_ep(traction_ep, tag_filter=None, type_filter=None):
    t = {}
    traction = traction_ep.load()
    if tag_filter and not filter_tag(tag_filter, traction.tags):
        return None
    if type_filter and traction._TYPE != type_filter:
        return None
    t['name'] = str(traction)
    t['type'] = traction._TYPE
    t['module'] = traction.__module__
    t['docs'] = getattr(traction, 'd_', None)
    #t['tags'] = traction.tags
    t['inputs'] = [
        {"name": k,
         "type": v.type_to_json()
        } for k, v in traction._fields.items() if k.startswith('i_')
    ]
    t['outputs'] = [
        {"name": k,
         "type": v.type_to_json()
        } for k, v in traction._fields.items() if k.startswith('o_')
    ]
    t['resources'] = [
        {"name": k,
         "type": v.type_to_json()
        } for k, v in traction._fields.items() if k.startswith('r_')
    ]
    t['args'] = [
        {"name": k,
         "type": v.type_to_json()
        } for k, v in traction._fields.items() if k.startswith('a_')
    ]
    return t


def inspect_distribution(distribution, tag_filter=None, type_filter=None, name_filter=None):
    d = {}
    if name_filter and not filter_name(name_filter, distribution.metadata['Name']):
        return None
    d['name'] = distribution.metadata['Name']
    d['tags'] = []
    d['tractions'] = []
    d['args'] = []
    d['inputs'] = []
    d['outputs'] = []
    d['resources'] = []
    for t in distribution.entry_points.select(group='tractions'):
        t_meta = inspect_traction_ep(t, tag_filter=tag_filter, type_filter=type_filter)
        if t_meta:
            d['tractions'].append(t_meta)
            d['args'].extend(sum([gather_simple_types(a) for a in t_meta['args']],[]))
            d['resources'].extend(sum([gather_simple_types(a) for a in t_meta['resources']], []))
            d['inputs'].extend(sum([gather_simple_types(a) for a in t_meta['inputs']], []))
            d['outputs'].extend(sum([gather_simple_types(a) for a in t_meta['outputs']], []))
            d['args'] = list(set(d['args']))
            d['resources'] = list(set(d['resources']))
            d['inputs'] = list(set(d['inputs']))
            d['outputs'] = list(set(d['outputs']))

    return d


def catalog(tag_filter=None, type_filter=None, name_filter=None):
    tractions = tractions_discovery(tag_filter=None, type_filter=None, name_filter=None)
    all_inputs = list(set([inp for t in tractions for inp in t['inputs']]))
    all_outputs = list(set([out for t in tractions for out in t['outputs']]))
    all_resources = list(set([res for t in tractions for res in t['resources']]))
    all_args = list(set([arg for t in tractions for arg in t['args']]))
    tags = []
    return tractions, all_inputs, all_outputs, all_resources, all_args, tags


def catalog_main(args):
    tractions = catalog(tag_filter=args.tag_filter, type_filter=args.type_filter, name_filter=args.name_filter)
    if args.output == 'json':
        import json
        print(json.dumps(tractions[0], indent=2))
    elif args.output == 'yaml':
        import yaml
        print(yaml.dump(tractions[0]))

def make_parsers(subparsers):
    p_catalog = subparsers.add_parser('catalog', help='Explore all locally available tractions')
    p_catalog.add_argument('--tag', '-t', dest='tag_filter', help='Filter tractions by tag', default=None)
    p_catalog.add_argument('--type', '-y', dest='type_filter', help='Filter tractions by type',
                        choices=['TRACTOR', 'TRACTION', 'STMD'], default=None)
    p_catalog.add_argument('--name', '-n', dest='name_filter', help='Filter by distribution name', default=None)
    p_catalog.add_argument('--output-format', '-o', dest='output', help='Output format', choices=['json', 'yaml'], default='json')
    p_catalog.set_defaults(command=catalog_main)
    return p_catalog


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pytraction container ep")
    subparsers = parser.add_subparsers(required=True, dest='command')
    make_parsers(subparsers)
    args = parser.parse_args()
    tractions, _, _, _, _, _ = catalog(tag_filter=args.tag_filter, type_filter=args.type_filter, name_filter=args.name_filter)
    if args.output == 'json':
        import json
        print(json.dumps(tractions, indent=2))
    elif args.output == 'yaml':
        import yaml
        print(yaml.dump(tractions))
