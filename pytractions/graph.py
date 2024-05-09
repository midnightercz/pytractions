# flake8: noqa

INPUT_PORTS_GROUP = {
    "position": {"name": "top"},
    "attrs": {"portBody": {"magnet": "passive", "r": 10, "fill": "#023047", "stroke": "#023047"}},
    "label": {
        "position": {"name": "top", "args": {"y": -20, "x": -20, "angle": 45}},
        "markup": [{"tagName": "text", "selector": "label", "className": "label-text"}],
    },
    "markup": [{"tagName": "circle", "selector": "portBody"}],
}
STANDALONE_INPUT_PORTS_GROUP = {
    "position": {"name": "bottom"},
    "attrs": {"portBody": {"magnet": True, "r": 10, "fill": "#023047", "stroke": "#023047"}},
    "label": {
        "position": {
            "name": "right",
            "args": {
                "y": 0,
            },
        },
        "markup": [{"tagName": "text", "selector": "label", "className": "label-text"}],
    },
    "markup": [{"tagName": "circle", "selector": "portBody"}],
}

OUTPUT_PORTS_GROUP = {
    "position": {"name": "bottom"},
    "attrs": {"portBody": {"magnet": True, "r": 10, "fill": "#E6A502", "stroke": "#023047"}},
    "label": {
        "position": {"name": "bottom", "args": {"y": 20, "x": 20, "angle": 45}},
        "markup": [{"tagName": "text", "selector": "label", "className": "label-text"}],
    },
    "markup": [{"tagName": "circle", "selector": "portBody"}],
}

ARGS_PORTS_GROUP = {
    "position": {"name": "left"},
    "attrs": {"portBody": {"magnet": True, "r": 10, "fill": "#11e602", "stroke": "#023047"}},
    "label": {
        "position": {"name": "left", "args": {"y": 6}},
        "markup": [{"tagName": "text", "selector": "label", "className": "label-text"}],
    },
    "markup": [{"tagName": "circle", "selector": "portBody"}],
}

STANDALONE_ARGS_PORTS_GROUP = {
    "position": {"name": "right"},
    "attrs": {"portBody": {"magnet": True, "r": 10, "fill": "#11e602", "stroke": "#023047"}},
    "label": {
        "position": {"name": "right", "args": {"y": 6}},
        "markup": [{"tagName": "text", "selector": "label", "className": "label-text"}],
    },
    "markup": [{"tagName": "circle", "selector": "portBody"}],
}

RESOURCES_PORTS_GROUP = {
    "position": {"name": "left"},
    "attrs": {"portBody": {"magnet": True, "r": 10, "fill": "#bf3b3b", "stroke": "#023047"}},
    "label": {
        "position": {"name": "left", "args": {"y": 0}},
        "markup": [{"tagName": "text", "selector": "label", "className": "label-text"}],
    },
    "markup": [{"tagName": "circle", "selector": "portBody"}],
}

STANDALONE_RESOURCES_PORTS_GROUP = {
    "position": {"name": "right"},
    "attrs": {"portBody": {"magnet": True, "r": 10, "fill": "#bf3b3b", "stroke": "#023047"}},
    "label": {
        "position": {"name": "right", "args": {"y": 0}},
        "markup": [{"tagName": "text", "selector": "label", "className": "label-text"}],
    },
    "markup": [{"tagName": "circle", "selector": "portBody"}],
}


def make_in_ports(traction):
    for f, tf in traction._fields.items():
        if f.startswith("i_"):
            yield {
                "group": "ins",
                "attrs": {
                    "label": {
                        "text": f,
                    }
                },
                "label": {
                    "position": {
                        "name": "top",
                        "args": {"y": -20 - len(f) * 2, "x": -20 - len(f) * 2, "angle": 45},
                    }
                },
                "id": f"{traction.uid}:{f}",
            }


def make_out_ports(traction):
    for f, tf in traction._fields.items():
        if f.startswith("o_"):
            yield {
                "group": "outs",
                "attrs": {"label": {"text": f}},
                "label": {
                    "position": {
                        "name": "top",
                        "args": {"y": 20 + len(f) * 2, "x": 20 + len(f) * 2, "angle": 45},
                    }
                },
                "id": f"{traction.uid}:{f}",
            }


def make_resource_ports(traction):
    for f, tf in traction._fields.items():
        if f.startswith("r_"):
            yield {
                "group": "resources",
                "attrs": {"label": {"text": f}},
                "id": f"{traction.uid}:{f}",
            }


def make_args_ports(traction):
    for f, tf in traction._fields.items():
        if f.startswith("a_"):
            yield {
                "group": "resources",
                "attrs": {
                    "portBody": {"magnet": True, "r": 10, "fill": "#11e602", "stroke": "#023047"},
                    "label": {"text": f},
                },
                "id": f"{traction.uid}:{f}",
            }


def get_traction_args(traction):
    ret = []
    for f, tf in traction._fields.items():
        if f.startswith("a_"):
            ret.append(f)
    return ret


def get_traction_resources(traction):
    ret = []
    for f, tf in traction._fields.items():
        if f.startswith("r_"):
            ret.append(f)
    return ret


def get_traction_inputs(traction):
    ret = []
    for f, tf in traction._fields.items():
        if f.startswith("i_"):
            ret.append(f)
    return ret


def get_traction_outputs(traction):
    ret = []
    for f, tf in traction._fields.items():
        if f.startswith("o_"):
            ret.append(f)
    return ret


def traction_cell(traction, x, y, width, height):
    return {
        "type": "standard.Rectangle",
        "position": {"x": x, "y": y},
        "size": {
            "width": width,
            "height": height,
        },
        "angle": 0,
        "ports": {
            "groups": {
                "ins": INPUT_PORTS_GROUP,
                "outs": OUTPUT_PORTS_GROUP,
                "args": ARGS_PORTS_GROUP,
                "resources": RESOURCES_PORTS_GROUP,
            },
            "items": [
                *make_in_ports(traction),
                *make_out_ports(traction),
                *make_resource_ports(traction),
                *make_args_ports(traction),
            ],
        },
        "id": traction.uid,
        "z": 1,
        "attrs": {
            "body": {"fill": "#8ECAE6"},
            "label": {
                "fontSize": 16,
                "text": traction.__class__.__name__ + "\n" + traction.uid,
                "y": -10,
            },
            "id": traction.__class__.__name__ + ":" + traction.uid,
            "root": {"magnet": False},
        },
    }


def make_standalone_ioar(name, x, y, group):
    return {
        "type": "standard.Rectangle",
        "position": {"x": x, "y": y},
        "size": {"width": 150, "height": 30},
        "angle": 0,
        "ports": {
            "groups": {
                "ins": STANDALONE_INPUT_PORTS_GROUP,
                "outs": OUTPUT_PORTS_GROUP,
                "args": STANDALONE_ARGS_PORTS_GROUP,
                "resources": STANDALONE_RESOURCES_PORTS_GROUP,
            },
            "items": [{"group": group, "attrs": {"label": {}}, "id": name}],
        },
        "id": name,
        "z": 1,
        "attrs": {
            "body": {"fill": "#8ECAE6"},
            "label": {"fontSize": 16, "text": name, "y": 0},
            "id": "#",
            "root": {"magnet": False},
        },
    }


def make_connection(source, source_port, target, target_port):
    return {
        "type": "standard.Link",
        "source": {"id": source, "port": source_port},
        "target": {"id": target, "port": target_port},
        "z": 1,
        "attrs": {"line": {"stroke": "#000000", "strokeWidth": 1}},
    }


def tractor_graph(tractor):
    cells = []
    y = 0
    x = 0
    for f, tf in tractor._fields.items():
        if f.startswith("i_"):
            cells.append(make_standalone_ioar(f, x, y, "ins"))
            x += 200

    y = 200
    x = 0
    for f, tf in tractor._fields.items():
        if f.startswith("r_"):
            cells.append(make_standalone_ioar(f, x, y, "resources"))
            y += 50
        if f.startswith("a_"):
            cells.append(make_standalone_ioar(f, x, y, "args"))
            y += 50

    y = 200
    x = 400
    for f, tf in tractor._fields.items():
        if f.startswith("t_"):
            outputs = get_traction_outputs(getattr(tractor, f))
            inputs = get_traction_inputs(getattr(tractor, f))
            max_out = max([len(x) for x in outputs] or [0]) * 8
            max_in = max([len(x) for x in inputs] or [0]) * 8

            y += max_in
            width = (
                max(len(getattr(tractor, f).uid), len(getattr(tractor, f).__class__.__name__)) + 20
            ) * 10
            height = max(
                (
                    len(get_traction_args(getattr(tractor, f)))
                    + len(get_traction_resources(getattr(tractor, f)))
                )
                * 20,
                70,
            )

            cells.append(traction_cell(getattr(tractor, f), x, y, width, height))
            y += height + max_out
    for (source, source_id), (target, target_id) in tractor._io_map.items():
        print(source, source_id)
        if source != "#":
            source_t = getattr(tractor, source).uid
        else:
            source_t = source_id
        if target != "#":
            target_t = getattr(tractor, target).uid
        else:
            target_t = target_id
        cells.append(
            make_connection(
                source_t, source_t + ":" + source_id, target_t, target_t + ":" + target_id
            )
        )

    return {"cells": cells}
