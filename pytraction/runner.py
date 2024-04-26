import json
import argparse
import datetime
import sys
import os

from .monitor import StructuredMonitor
from .base import Base


class SimpleRunner:
    def __init__(self, tractor, monitor_file):
        self.tractor = tractor
        self.monitor_file = monitor_file

    def run(self):
        monitor = StructuredMonitor(self.tractor, self.monitor_file)
        try:
            self.tractor.run(on_update=monitor.on_update)
        finally:
            monitor.close(self.tractor)

    def resubmit(self, traction):
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
                    self.tractor.uid + "::" + getattr(self.tractor, tf).uid + ".json")
                if os.path.exists(traction_path):
                    self.tractor.tractions[tf] = ftype.from_json(
                        json.load(open(traction_path))
                    )
        for output in outputs:
            if output[0] != "#":
                traction_path = os.path.join(
                    self.monitor_file,
                    self.tractor.uid + "::" + getattr(self.tractor, output[0]).uid + ".json")
                self.tractor.tractions[output[0]] = getattr(self.tractor, output[0]).from_json(
                    json.load(open(traction_path)))
            else:
                traction_path = os.path.join(
                    self.monitor_file,
                    self.tractor.uid + "::" + output[1] + ".json")
                ftype = self.tractor._fields[output[1]]
                setattr(self.tractor, output[1], ftype.from_json(
                    json.load(open(traction_path))
                ))

        for f, ftype in self.tractor._fields.items():
            if f == traction:
                loading_started = True
            if f.startswith("t_") and loading_started:
                inputs = self.tractor._init_traction_input(f, ftype)
                for _in, t_in in inputs.items():
                    object.__setattr__(self.tractor.tractions[f], _in,  t_in)



        #self.tractor = self.tractor.from_json(json.load(open(os.path.join(self.monitor_file, self.tractor.uid + ".json"))))
        monitor = StructuredMonitor(self.tractor, self.monitor_file)
        self.tractor.resubmit_from(traction)
        try:
            self.tractor.run(on_update=monitor.on_update)
        finally:
            monitor.close(self.tractor)
