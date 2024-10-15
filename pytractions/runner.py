import argparse
import json
import logging
import sys
import os
import yaml

from .monitor import StructuredMonitor
from .base import Traction
from .runner_utils import parse_traction_str, gen_default_inputs


LOGGER = logging.getLogger()
sh = logging.StreamHandler(stream=sys.stdout)
sh.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
LOGGER.addHandler(sh)


class SimpleRunner:
    """Simple runner class."""

    def __init__(self, tractor, monitor_file):
        """Initialize the runner."""
        self.tractor = tractor
        self.monitor_file = monitor_file

    def run(self):
        """Run tractor."""
        monitor = StructuredMonitor(self.tractor, self.monitor_file)
        try:
            self.tractor.run(on_update=monitor.on_update)
        finally:
            monitor.close(self.tractor)

    def resubmit(self, traction):
        """Run tractor from specific traction."""
        loading_started = False
        _ttraction = getattr(self.tractor, traction)
        outputs = []

        for tf, ftype in self.tractor._fields.items():
            if tf == traction:
                loading_started = True
            if tf.startswith("t_") and loading_started:
                _ttraction = getattr(self.tractor, traction)
                for f, _ in _ttraction._fields.items():
                    if f.startswith("i_") and self.tractor._io_map[(traction, f)] not in outputs:
                        outputs.append(self.tractor._io_map[(traction, f)])
                traction_path = os.path.join(
                    self.monitor_file,
                    self.tractor.uid + "::" + getattr(self.tractor, tf).uid + ".json",
                )
                if os.path.exists(traction_path):
                    self.tractor.tractions[tf] = ftype.from_json(json.load(open(traction_path)))
        for output in outputs:
            if output[0] != "#":
                traction_path = os.path.join(
                    self.monitor_file,
                    self.tractor.uid + "::" + getattr(self.tractor, output[0]).uid + ".json",
                )
                self.tractor.tractions[output[0]] = getattr(self.tractor, output[0]).from_json(
                    json.load(open(traction_path))
                )
            else:
                traction_path = os.path.join(
                    self.monitor_file, self.tractor.uid + "::" + output[1] + ".json"
                )
                ftype = self.tractor._fields[output[1]]
                setattr(self.tractor, output[1], ftype.from_json(json.load(open(traction_path))))

        for f, ftype in self.tractor._fields.items():
            if f == traction:
                loading_started = True
            if f.startswith("t_") and loading_started:
                inputs = self.tractor._init_traction_input(f, ftype)
                for _in, t_in in inputs.items():
                    object.__setattr__(self.tractor.tractions[f], _in, t_in)

        monitor = StructuredMonitor(self.tractor, self.monitor_file)
        self.tractor.resubmit_from(traction)
        try:
            self.tractor.run(on_update=monitor.on_update)
        finally:
            monitor.close(self.tractor)


def load_yaml_input(traction_cls):
    traction_init_fields = {}
    docs = yaml.safe_load_all(sys.stdin.read())
    for doc in docs:
        name, data, data_file = doc["name"], doc.get("data"), doc.get("data_file")
        if data_file:
            data = yaml.safe_load(open(data_file).read())
            data = data["data"]
        if name not in traction_cls._fields:
            raise AttributeError(f"{traction_cls.__name__} doesn't have field {name}")
        LOGGER.info(f"Loading input: {name} {yaml.safe_load(data)}")
        traction_init_fields[name] = traction_cls._fields[name].content_from_json(
            yaml.safe_load(data)
        )
    return traction_init_fields


def load_json_input(traction_cls):
    json_values = {}
    for param in args.params:
        name, value = param.split("=")
        if value.startswith("@"):
            try:
                _value = json.load(open(value[1:]))
            except json.JSONDecodeError:
                _value = open(value[1:]).read()
        elif value:
            try:
                _value = json.loads(value)
            except json.JSONDecodeError:
                _value = value
        else:
            continue
        nested_name = name.split(".")
        current_nest = json_values
        for v in nested_name:
            if v == nested_name[-1]:
                current_nest[v] = _value
            else:
                current_nest.setdefault(v, {})
            current_nest = current_nest[v]
    for name, json_val in json_values.items():
        traction_init_fields[name] = traction_cls._fields[name].content_from_json(json_val)
    return traction_init_fields


def run_main(args):
    """Run action."""
    traction_cls = parse_traction_str(args.traction)
    traction_init_fields = {}
    LOGGER.setLevel(getattr(logging, args.level))
    if args.io_type == "YAML":
        traction_init_fields = load_yaml_input(traction_cls)
    else:
        traction_init_fields = load_json_input(traction_cls)

    traction = traction_cls(uid="0", **traction_init_fields)
    LOGGER.info("Running simple runner on directory {args.monitor}")
    runner = SimpleRunner(traction, args.monitor)
    runner.run()


def resubmit_main(args):
    """Resubmit action."""
    traction = Traction.from_json(json.load(open(os.path.join(args.monitor, "-root-.json"))))
    runner = SimpleRunner(traction, args.monitor)
    runner.resubmit(args.from_traction)


def gen_inputs_main(args):
    """Generate inputs for the traction."""
    traction_cls = parse_traction_str(args.traction)
    defaults = gen_default_inputs(traction_cls)
    yaml_out = []
    for k, v in defaults.items():
        yaml_out.append({"name": k, "data": v})
    print(yaml.dump_all(yaml_out))


def make_parsers(subparsers):
    """Make runner parser."""
    p_runner = subparsers.add_parser("local_run", help="Run pytraction module")
    p_runner.add_argument("traction", help="Path of traction to run (module:traction)", type=str)
    p_runner.add_argument(
        "--monitor", "-m", help="Path to monitor directory", type=str, default="monitor"
    )
    p_runner.add_argument(
        "--level",
        "-l",
        help="Set log level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    p_runner.add_argument(
        "--io-type",
        "-t",
        help="Choce way how to pass inputs to the traction. For JSON use PARAM to pass inputs"
        "to the traction. For YAML pass YAML formated documents to STDIN",
        type=str,
        choices=["YAML", "JSON"],
        default="JSON",
    )
    p_runner.set_defaults(command=run_main)

    p_resubmit = subparsers.add_parser("local_resubmit", help="Run pytraction module")
    p_resubmit.add_argument(
        "--monitor", "-m", help="Path to monitor directory", type=str, default="monitor"
    )
    p_resubmit.add_argument(
        "--from-traction", "-t", help="Resubmit from specific traction", type=str, default="monitor"
    )
    p_resubmit.set_defaults(command=resubmit_main)

    p_gen_inputs = subparsers.add_parser("generate_inputs", help="Run pytraction module")
    p_gen_inputs.add_argument(
        "traction", help="Path of traction to run (module:traction)", type=str
    )
    p_gen_inputs.set_defaults(command=gen_inputs_main)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytraction simple runner")
    subparsers = parser.add_subparsers(required=True, dest="command")
    make_parsers(subparsers)
    args = parser.parse_args()
    args.command(args)
