import argparse
import re

import importlib.metadata


def filter_name(name_filter, name):
    """Test if name matches filter_name."""
    if isinstance(name_filter, re.Pattern):
        return re.match(name_filter, name)
    return name_filter in name


def filter_tag(tag_filter, tags):
    """Test if tag is in list of tags."""
    if isinstance(tag_filter, re.Pattern):
        return any(re.match(tag_filter, t) for t in tags)
    return tag_filter in tags


ignore_types = (
    "TList",
    "TDict",
    "Union",
    "Optional",
    "Port",
    "NullPort",
    "STMDSingleIn",
    "ABase",
)


def gather_simple_types(type_json):
    """Gather simple types from complex type in json format."""
    simple_types = []
    if type_json["type"]["$type"]["type"] not in ignore_types:
        simple_types.append(type_json["type"]["$type"]["type"])
    stack = []
    for arg in type_json["type"]["$type"].get("args", []):
        stack.append(arg)
    while stack:
        arg = stack.pop()
        if arg["type"] not in ignore_types:
            simple_types.append(arg["type"])
        for a in arg.get("args", []):
            stack.append(a)
    return list(set(simple_types))


def tractions_discovery(
    tag_filter=None,
    type_filter=None,
    name_filter=None,
    inputs_type_filter=None,
    outputs_type_filter=None,
    resources_type_filter=None,
    args_type_filter=None,
    type_to_str=False,
):
    """Inspect all available distributions for tractions entry points."""
    ret = []
    seen = set()
    for d in importlib.metadata.distributions():
        if d.entry_points.select(group="tractions"):
            if d._path in seen:
                continue
            seen.add(d._path)
            dist = inspect_distribution(
                d,
                tag_filter=tag_filter,
                type_filter=type_filter,
                name_filter=name_filter,
                inputs_type_filter=inputs_type_filter,
                outputs_type_filter=outputs_type_filter,
                resources_type_filter=resources_type_filter,
                args_type_filter=args_type_filter,
                type_to_str=type_to_str,
            )
            if dist and dist not in ret:
                ret.append(dist)
    return ret


def _type_to_str(type_json):
    """Return string representation for given type."""
    out = []
    stack = []
    out.append(type_json["$type"]["type"])
    if type_json["$type"].get("args", []):
        stack.insert(0, "]")
        for n, arg in enumerate(type_json["$type"].get("args", [])[::-1]):
            stack.insert(0, arg)
            if n != len(type_json["$type"].get("args", [])) - 1:
                stack.insert(0, ",")
        stack.insert(0, "[")

    while stack:
        item = stack.pop(0)
        if isinstance(item, str):
            out.append(item)
            continue
        out.append(item["type"])
        if item.get("args", []):
            stack.insert(0, "]")
            for n, a in enumerate(item.get("args", [])[::-1]):
                stack.insert(0, a)
                if n != len(item.get("args", [])) - 1:
                    stack.insert(0, ",")
            stack.insert(0, "[")
    return "".join(out)


def convert_types_to_str(t_meta):
    """Return string representation of attribute type."""
    t_meta["inputs"] = [
        {"name": i["name"], "type": _type_to_str(i["type"]), "docs": str(i["docs"])}
        for i in t_meta["inputs"]
    ]
    t_meta["outputs"] = [
        {"name": i["name"], "type": _type_to_str(i["type"]), "docs": str(i["docs"])}
        for i in t_meta["outputs"]
    ]
    t_meta["resources"] = [
        {"name": i["name"], "type": _type_to_str(i["type"]), "docs": str(i["docs"])}
        for i in t_meta["resources"]
    ]
    t_meta["args"] = [
        {"name": i["name"], "type": _type_to_str(i["type"]), "docs": str(i["docs"])}
        for i in t_meta["args"]
    ]
    return t_meta


def inspect_traction_ep(traction_ep, tag_filter=None, name_filter=None, type_filter=None):
    """Inspect traction entry point."""
    t = {}
    traction = traction_ep.load()
    t["name"] = str(traction)
    t["type"] = traction._TYPE
    t["module"] = traction.__module__
    t["docs"] = getattr(traction, "d_", None)
    t["tags"] = []

    if name_filter and not filter_name(name_filter, t["name"]):
        return {}
    if tag_filter and not filter_tag(tag_filter, traction.tags):
        return {}
    if type_filter and traction._TYPE not in type_filter:
        return {}

    t["inputs"] = [
        {"name": k, "type": v.type_to_json(), "docs": getattr(traction, "d_" + k, None)}
        for k, v in traction._fields.items()
        if k.startswith("i_")
    ]
    t["outputs"] = [
        {"name": k, "type": v.type_to_json(), "docs": getattr(traction, "d_" + k, None)}
        for k, v in traction._fields.items()
        if k.startswith("o_")
    ]
    t["resources"] = [
        {"name": k, "type": v.type_to_json(), "docs": getattr(traction, "d_" + k, None)}
        for k, v in traction._fields.items()
        if k.startswith("r_")
    ]
    t["args"] = [
        {"name": k, "type": v.type_to_json(), "docs": getattr(traction, "d_" + k, None)}
        for k, v in traction._fields.items()
        if k.startswith("a_")
    ]
    return t


def inspect_distribution(
    distribution,
    tag_filter=None,
    type_filter=None,
    name_filter=None,
    inputs_type_filter=None,
    outputs_type_filter=None,
    resources_type_filter=None,
    args_type_filter=None,
    type_to_str=False,
):
    """Inspect python distribution object for tractions entry points."""
    d = {}
    d["name"] = distribution.metadata["Name"]
    d["tags"] = []
    d["tractions"] = []
    d["args"] = []
    d["inputs"] = []
    d["outputs"] = []
    d["resources"] = []
    for t in distribution.entry_points.select(group="tractions"):
        t_meta = inspect_traction_ep(
            t, tag_filter=tag_filter, type_filter=type_filter, name_filter=name_filter
        )
        if t_meta:
            d["args"].extend(sum([gather_simple_types(a) for a in t_meta["args"]], []))
            d["resources"].extend(sum([gather_simple_types(a) for a in t_meta["resources"]], []))
            d["inputs"].extend(sum([gather_simple_types(a) for a in t_meta["inputs"]], []))
            d["outputs"].extend(sum([gather_simple_types(a) for a in t_meta["outputs"]], []))
            d["args"] = list(set(d["args"]))
            d["resources"] = list(set(d["resources"]))
            d["inputs"] = list(set(d["inputs"]))
            d["outputs"] = list(set(d["outputs"]))
        else:
            continue

        t_meta_str_types = convert_types_to_str(t_meta.copy())

        if inputs_type_filter:
            if not any(
                [
                    any([rt in t["type"] for t in t_meta_str_types["inputs"]])
                    for rt in inputs_type_filter
                ]
            ):
                continue
        if outputs_type_filter:
            if not any(
                [
                    any([rt in t["type"] for t in t_meta_str_types["outputs"]])
                    for rt in outputs_type_filter
                ]
            ):
                continue
        if resources_type_filter:
            if not any(
                [
                    any([rt in t["type"] for t in t_meta_str_types["resources"]])
                    for rt in resources_type_filter
                ]
            ):
                continue
        if args_type_filter:
            if not any(
                [
                    any([rt in t["type"] for t in t_meta_str_types["args"]])
                    for rt in args_type_filter
                ]
            ):
                continue
        d["tractions"].append(t_meta_str_types if type_to_str else t_meta)

    return d


def catalog(
    tag_filter=None,
    type_filter=None,
    name_filter=None,
    args_type_filter=None,
    inputs_type_filter=None,
    outputs_type_filter=None,
    resources_type_filter=None,
    type_to_str=False,
):
    """Return list of available tractions based on given filters."""
    tractions = tractions_discovery(
        tag_filter=tag_filter,
        type_filter=type_filter,
        name_filter=name_filter,
        args_type_filter=args_type_filter,
        inputs_type_filter=inputs_type_filter,
        outputs_type_filter=outputs_type_filter,
        resources_type_filter=resources_type_filter,
        type_to_str=type_to_str,
    )
    all_inputs = list(set([inp for t in tractions for inp in t["inputs"]]))
    all_outputs = list(set([out for t in tractions for out in t["outputs"]]))
    all_resources = list(set([res for t in tractions for res in t["resources"]]))
    all_args = list(set([arg for t in tractions for arg in t["args"]]))
    tags = []
    return tractions, all_inputs, all_outputs, all_resources, all_args, tags


def catalog_main(args):
    """Run catalog for CLI."""
    tractions = catalog(
        tag_filter=args.tag_filter,
        type_filter=args.type_filter,
        name_filter=args.name_filter,
        type_to_str=True,
    )
    if args.output == "json":
        import json

        print(json.dumps(tractions[0], indent=2))
    elif args.output == "yaml":
        import yaml

        print(yaml.dump(tractions[0]))


def make_parsers(subparsers):
    """Make parser for catalog subcommand."""
    p_catalog = subparsers.add_parser("catalog", help="Explore all locally available tractions")
    p_catalog.add_argument(
        "--tag", "-t", dest="tag_filter", help="Filter tractions by tag", default=None
    )
    p_catalog.add_argument(
        "--type",
        "-y",
        dest="type_filter",
        help="Filter tractions by type",
        choices=["TRACTOR", "TRACTION", "STMD"],
        default=None,
    )
    p_catalog.add_argument(
        "--name", "-n", dest="name_filter", help="Filter by distribution name", default=None
    )
    p_catalog.add_argument(
        "--output-format",
        "-o",
        dest="output",
        help="Output format",
        choices=["json", "yaml"],
        default="json",
    )
    p_catalog.set_defaults(command=catalog_main)
    return p_catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytraction container ep")
    subparsers = parser.add_subparsers(required=True, dest="command")
    make_parsers(subparsers)
    args = parser.parse_args()
    tractions, _, _, _, _, _ = catalog(
        tag_filter=args.tag_filter,
        type_filter=args.type_filter,
        name_filter=args.name_filter,
        type_to_str=True,
    )
    if args.output == "json":
        import json

        print(json.dumps(tractions, indent=2))
    elif args.output == "yaml":
        import yaml

        print(yaml.dump(tractions))
