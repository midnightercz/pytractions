import argparse
import enum
import json
import jsonschema
import logging
import sys
import re
import yaml
from typing import _UnionGenericAlias

from .base import TypeNode, ANY, TList, TDict
from .runner_utils import (
    parse_traction_str,
    StrParam,
    str_presenter,
    str_param,
    get_traction_defaults,
    gen_default_inputs,
)


LOGGER = logging.getLogger()
sh = logging.StreamHandler(stream=sys.stdout)
sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
LOGGER.addHandler(sh)


def enum_param(dumper, data):
    """Yaml enum representer."""
    return dumper.represent_scalar("tag:yaml.org,2002:str", data.value)


yaml.add_representer(str, str_presenter)
yaml.add_representer(StrParam, str_param)
yaml.add_multi_representer(enum.Enum, enum_param)


# to use with safe_dump:
yaml.representer.SafeRepresenter.add_representer(str, str_presenter)
yaml.representer.SafeRepresenter.add_representer(StrParam, str_param)
yaml.representer.SafeRepresenter.add_multi_representer(enum.Enum, enum_param)


def tekton_task_name(name):
    """Generate tekton task name from the name."""
    return re.sub(
        r"(^|[A-Z])",
        lambda match: match.group().lower() if match.start() == 0 else "-" + match.group().lower(),
        name,
    ).replace("_", "-")


def generate_type_description(type_, indent=0):
    """Generate type description."""
    if type_ in (str, int, float, bool, type(None)):
        return f"{type_.__name__}"
    elif type_.__class__ == _UnionGenericAlias:
        if type_.__name__ == "Optional":
            return f"Optional[{generate_type_description(type_.__args__[0], indent=indent)}]"
        else:
            ret = "Union["
            for arg in type_.__args__:
                ret += f"{generate_type_description(arg, indent=indent)}, "
            ret += "]"
            return ret
    elif TypeNode.from_type(type_) == TypeNode.from_type(TList[ANY]):
        return f"List[{generate_type_description(type_._params[0], indent=indent)}]"
    elif TypeNode.from_type(type_) == TypeNode.from_type(TDict[ANY, ANY]):
        return (
            f"Dict[{generate_type_description(type_._params[0], indent=indent)},"
            f"{generate_type_description(type_._params[1], indent=indent)}]"
        )
    elif issubclass(type_, enum.Enum):
        return f"{type_.__name__}"
    else:
        fields = ",\n".join(
            [
                " " * (indent + 4) + f"{f}: {generate_type_description(t, indent=indent+4)}"
                for f, t in type_._fields.items()
                if not f.startswith("_")
            ]
        )
        ret = f"{type_.__name__}(\n{fields}\n{indent * ' '})"
        return ret


def generate_param_description(traction, field):
    """Generate param description."""
    description = f"""DESCRIPTION: '{getattr(traction, "d_" + field, None) or ""}'

TYPE: {generate_type_description(traction._fields[field])}"""
    return description


def generate_traction_name_str(traction, include_module=False):
    """Generate traction name string."""
    if include_module:
        module = traction.__module__ + ":"
    else:
        module = ""
    if traction._params:
        return (
            f"{module}{traction.__orig_qualname__}"
            + f"""[{','.join([generate_traction_name_str(p, include_module=True)
                    for p in traction._params])}]"""
        )
    else:
        return f"{module}{traction.__name__}"


def generate_task_spec(traction, docker_image, inputs_map={}, id_in_tractor=None, volumes={}):
    """Generate tekton task spec."""
    params = []
    results = []
    steps = []
    spec = {
        "description": getattr(traction, "d_", None) or "",
        "params": params,
        "steps": steps,
        "workspaces": [{"name": "outputs"}],
    }
    for f, fv in traction._fields.items():
        if f.startswith("a_"):
            param = {
                "name": f,
                "type": "string",
                "description": generate_param_description(traction, f),
            }
            params.append(param)
        elif f.startswith("i_") and f not in inputs_map:
            param = {
                "name": f,
                "type": "string",
                "description": generate_param_description(traction, f),
            }
            params.append(param)
        elif f.startswith("r_"):
            param = {
                "name": f,
                "type": "string",
                "description": generate_param_description(traction, f),
            }
            params.append(param)
        elif f.startswith("o_"):
            results.append({"name": f, "description": getattr(traction, "d_" + f, None) or ""})
        elif f in ("stats", "state"):
            results.append({"name": f})
    params_str = yaml.dump_all(
        [{"name": p["name"], "data": StrParam(f'$(params["{p["name"]}"])')} for p in params]
    )
    inputs_str = yaml.dump_all(
        [
            {"name": k, "data_file": f"$(workspaces.outputs.path)/{v[0]}::{v[1]}"}
            for k, v in inputs_map.items()
        ]
    )
    delimiter = "---" if inputs_str else ""
    tid = id_in_tractor or traction.uid
    store_results_str = " ".join(
        [
            f'--store-output {r["name"]}=$(workspaces.outputs.path)/{tid}::{r["name"]}'
            for r in results
        ]
    )

    steps.append(
        {
            "name": "run",
            "image": docker_image,
            "workingDir": "/",
            "script": f"""#!/usr/bin/bash --posix
cat <<EOF |
{params_str}
{delimiter}
{inputs_str}
EOF
python -m pytractions.container_runner run {store_results_str}\\
    "{traction.__module__}:{generate_traction_name_str(traction)}"
echo "# Run stats:"
cat $(workspaces.outputs.path)/{tid}::stats
echo "# Run state:"
cat $(workspaces.outputs.path)/{tid}::state
export result=$(cat $(workspaces.outputs.path)/{tid}::state |\
python -c "import yaml; import sys; print(yaml.safe_load(sys.stdin)['data'])")
[ "$result" = "finished" ] || exit 1
""",
        }
    )
    if volumes:
        steps[-1]["volumeMounts"] = []
    for name, path in volumes.items():
        steps[-1]["volumeMounts"].append({"name": name, "mountPath": path})
    return spec


def generate_tekton_task(traction, docker_image, volumes={}):
    """Generate tekton task."""
    result = {
        "apiVersion": "tekton.dev/v1beta1",
        "kind": "Task",
        "metadata": {
            "name": tekton_task_name(traction.__name__),
        },
        "spec": generate_task_spec(traction, docker_image, id_in_tractor="root", volumes=volumes),
    }
    return result


def generate_tekton_pipeline(tractor, docker_image, volumes={}):
    """Generate tekton pipeline."""
    waves = {}
    params = []
    results = []

    for traction, wave in tractor._traction_waves.items():
        waves.setdefault(wave, []).append(tekton_task_name(traction))

    for f, fv in tractor._fields.items():
        if f.startswith("a_") or f.startswith("i_") or f.startswith("r_"):
            param = {
                "name": f,
                "type": "string",
                "description": generate_param_description(tractor, f),
            }
            params.append(param)

    tasks = []
    for f, tf in tractor._fields.items():
        if f.startswith("t_"):
            tfo = getattr(tractor, f)
            ios_map = {}
            args_map = {}
            resources_map = {}
            tparams = {}
            inputs_map = {}
            for (traction, tfield), output in tractor._io_map.items():
                if f == traction:
                    if output[0] == "#":
                        ios_map[tfield] = "$(params.%s)" % (output[1])
                    else:
                        inputs_map[tfield] = output
            for (traction, tfield), arg in tractor._args_map.items():
                if f == traction:
                    args_map[tfield] = "$(params.%s)" % (arg)
            for (traction, tfield), resource in tractor._resources_map.items():
                if f == traction:
                    resources_map[tfield] = "$(params.%s)" % resource

            for field, value in get_traction_defaults(tfo).items():
                tparams[field] = value
            for field, output in ios_map.items():
                tparams[field] = output
                # tparams.append({"name": field, "value": output})
            for field, arg in args_map.items():
                tparams[field] = arg
                # tparams.append({"name": field, "value": arg})
            for field, resource in resources_map.items():
                tparams[field] = resource
                # tparams.append({"name": field, "value": resource})

            task = {
                "name": tekton_task_name(f),
                "taskSpec": generate_task_spec(
                    tfo, docker_image, inputs_map=inputs_map, id_in_tractor=f
                ),
                "params": [{"name": field, "value": value} for field, value in tparams.items()],
                "workspaces": [{"name": "outputs", "workspace": "outputs"}],
            }
            if tractor._traction_waves[f] > 1:
                task["runAfter"] = waves[tractor._traction_waves[f] - 1]
            tasks.append(task)

    result = {
        "apiVersion": "tekton.dev/v1beta1",
        "kind": "Pipeline",
        "metadata": {
            "name": tekton_task_name(tractor.__name__),
        },
        "spec": {
            "params": params,
            "tasks": tasks,
            "results": results,
            "workspaces": [
                {
                    "name": "outputs",
                }
            ],
        },
    }
    return result


def shift(yaml_data):
    """Shift yaml data 2 spaces to right."""
    ret = []
    for x in yaml_data.split("\n"):
        if x != "---":
            ret.append("  " + x)
        else:
            ret.append(x)
    return "\n".join(ret)


def generate_tekton_task_run(traction, secrets_to_volumes={}):
    """Generate tekton task run."""
    params = []
    default_inputs = gen_default_inputs(traction)
    for f, fv in traction._fields.items():
        if f.startswith("a_") or f.startswith("i_") or f.startswith("r_"):
            param = {
                "name": f,
                "value": shift(yaml.dump(default_inputs[f], explicit_start=True).rstrip()),
            }
            params.append(param)
    result = {
        "apiVersion": "tekton.dev/v1beta1",
        "kind": "TaskRun",
        "metadata": {
            "name": tekton_task_name(traction.__name__ + "-run"),
        },
        "spec": {
            "taskRef": {
                "name": tekton_task_name(traction.__name__),
            },
            "params": params,
            "workspaces": [
                {
                    "name": "outputs",
                    "volumeClaimTemplate": {
                        "spec": {
                            "accessModes": ["ReadWriteOnce"],
                            "resources": {"requests": {"storage": "1Gi"}},
                        }
                    },
                }
            ],
        },
    }
    if secrets_to_volumes:
        result["spec"]["podTemplate"] = {"volumes": []}
        for volname, voldata in secrets_to_volumes.items():
            result["spec"]["podTemplate"]["volumes"].append({"name": volname, "secret": voldata})

    return result


def generate_tekton_pipeline_run(traction):
    """Generate tekton pipeline run."""
    params = []
    default_inputs = gen_default_inputs(traction)
    for f, fv in traction._fields.items():
        if f.startswith("a_") or f.startswith("i_") or f.startswith("r_"):
            param = {
                "name": f,
                "value": shift(yaml.dump(default_inputs[f], explicit_start=True).rstrip()),
            }
            params.append(param)
    result = {
        "apiVersion": "tekton.dev/v1beta1",
        "kind": "PipelineRun",
        "metadata": {
            "name": tekton_task_name(traction.__name__ + "-run"),
        },
        "spec": {
            "pipelineRef": {
                "name": tekton_task_name(traction.__name__),
            },
            "workspaces": [
                {
                    "name": "outputs",
                    "volumeClaimTemplate": {
                        "spec": {
                            "accessModes": ["ReadWriteOnce"],
                            "resources": {"requests": {"storage": "1Gi"}},
                        }
                    },
                    "workspace": "outputs",
                }
            ],
            "params": params,
        },
    }
    return result


def field_from_json_str(json_str):
    """Get field from json string."""
    json_dict = json.loads(json_str)
    return json_dict["name"], json_dict["data"]


def generate_tekton_task_main(args):
    """Run tenton Task yaml generation."""
    traction_cls = parse_traction_str(args.traction)
    volumes = dict([vol.split(":") for vol in args.volume])
    print("# This file was generated by command:")
    print("#" + " ".join(sys.argv))
    if args.type == "tractor":
        print(
            yaml.dump(
                generate_tekton_pipeline(traction_cls, args.docker_image, volumes=volumes),
            )
        )
    else:
        print(
            yaml.dump(
                generate_tekton_task(traction_cls, args.docker_image, volumes=volumes),
            )
        )


SECRET_TO_VOLUME_SCHEMA = {
    "type": "object",
    "properties": {
        "volume_name": {"type": "string"},
        "secret_name": {"type": "string"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"key": {"type": "string"}, "path": {"type": "string"}},
            },
        },
    },
}


def generate_tekton_task_run_main(args):
    """Run tenton TaskRun yaml generation."""
    traction_cls = parse_traction_str(args.traction)
    secrets_to_volumes = {}
    for secret_to_volume in args.secret_to_volume:
        parsed = json.loads(secret_to_volume)
        jsonschema.validate(instance=parsed, schema=SECRET_TO_VOLUME_SCHEMA)
        secrets_to_volumes[parsed["volume_name"]] = {
            "secretName": parsed["secret_name"],
            "items": parsed["items"],
        }
    if args.type == "tractor":
        print(
            yaml.dump(
                generate_tekton_pipeline_run(traction_cls),
            )
        )
    else:
        print(
            yaml.dump(
                generate_tekton_task_run(traction_cls, secrets_to_volumes=secrets_to_volumes),
            )
        )


def run_main(args):
    """Run traction in the container."""
    LOGGER.setLevel(getattr(logging, args.level))
    traction_cls = parse_traction_str(args.traction)
    traction_init_fields = {}
    docs = yaml.safe_load_all(sys.stdin.read())
    if not docs:
        LOGGER.warning(f"Not input provided for traction {args.traction}")
    for doc in docs:
        name, data, data_file = doc["name"], doc.get("data"), doc.get("data_file")
        LOGGER.info("Reading input %s", name)
        if data_file:
            data = yaml.safe_load(open(data_file).read())
            data = data["data"]
        if name not in traction_cls._fields:
            raise AttributeError(f"{traction_cls.__name__} doesn't have field {name}")
        traction_init_fields[name] = traction_cls._fields[name].content_from_json(
            yaml.safe_load(data)
        )
    traction = traction_cls(uid="0", **traction_init_fields)
    LOGGER.info(f"Running traction {args.traction}")
    traction.run()
    outputs_map = {}
    for store_output in args.store_output:
        outputs_map[store_output.split("=")[0]] = store_output.split("=")[1]

    for f in outputs_map:
        if f not in traction._fields:
            raise AttributeError(f"{traction_cls.__name__} doesn't have field {f}")

    for f, ftype in traction._fields.items():
        if f in outputs_map:
            LOGGER.info(f"Storing output {f} to {outputs_map[f]}")
            if f == "stats":
                o_yaml = yaml.safe_dump(
                    {"name": f, "data": yaml.safe_dump(getattr(traction, f).content_to_json())}
                )
            elif f == "state":
                o_yaml = yaml.safe_dump({"name": f, "data": getattr(traction, f).value})
            else:
                o_yaml = yaml.safe_dump(
                    {"name": f, "data": StrParam(yaml.dump(getattr(traction, f).content_to_json()))}
                )
            with open(outputs_map[f], "w") as _f:
                _f.write(o_yaml)


def make_parsers(subparsers):
    """Make argparser for all commands in this module."""
    p_generate_tekton_task = subparsers.add_parser(
        "generate_tekton_task", help="Generate tekton task yaml"
    )
    p_generate_tekton_task.add_argument(
        "--type", choices=("traction", "tractor"), default="traction"
    )
    p_generate_tekton_task.add_argument("traction", help="Traction to describe")
    p_generate_tekton_task.add_argument("docker_image", help="docker image for tekton task")
    p_generate_tekton_task.add_argument(
        "--volume", help="volume points for the task", action="append", default=[]
    )
    p_generate_tekton_task.set_defaults(command=generate_tekton_task_main)

    p_generate_tekton_task_run = subparsers.add_parser(
        "generate_tekton_task_run", help="Generate tekton taskrun yaml"
    )
    p_generate_tekton_task_run.add_argument("traction", help="Traction to describe")
    p_generate_tekton_task_run.add_argument(
        "--type", choices=("traction", "tractor"), default="traction"
    )
    p_generate_tekton_task_run.add_argument(
        "--secret-to-volume",
        help=f"Map secret to specific volume "
        f"(required format: {json.dumps(SECRET_TO_VOLUME_SCHEMA)})",
        action="append",
        default=[],
    )
    p_generate_tekton_task_run.set_defaults(command=generate_tekton_task_run_main)

    run_parser = subparsers.add_parser("run", help="Run a traction")
    run_parser.add_argument("traction", help="Traction to run")
    run_parser.add_argument(
        "--store-output",
        action="append",
        help="mapping of output=/file/path where specific output should be stored",
        default=[],
    )
    run_parser.add_argument(
        "--level",
        "-l",
        help="Set log level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    run_parser.set_defaults(command=run_main)


def run_in_container():
    """Run traction in the container (Standalone)."""
    parser = argparse.ArgumentParser(description="Run a traction in a docker container")
    subparsers = parser.add_subparsers(required=True, dest="command")
    make_parsers(subparsers)

    args = parser.parse_args()
    args.command(args)


if __name__ == "__main__":
    run_in_container()
