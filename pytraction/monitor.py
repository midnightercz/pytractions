import datetime
import json
import os

from .base import Traction
from .tractor import Tractor

class Monitor:
    def update(self):
        #with open(self.filename, "w") as f:
        #    f.write(json.dumps(self.tractor.content_to_json()))
        if not self.updating:
            self.updating = True
            with open("full-"+self.filename, "w") as f:
                f.write(json.dumps(self.tractor.to_json()))
            self.updating = False

        if not self.dead:
            threading.Timer(10, self.update).start()

    def __init__(self, tractor, filename):
        self.updating = False
        self.dead = False
        self.tractor = tractor
        self.filename = filename
        self.last_update_request = datetime.datetime.now()
        self.update()

    def on_update(self, traction):
        self.last_update_request = datetime.datetime.now()

    def close(self):
        self.update()
        self.dead = True


class StructuredMonitor:
    def __init__(self, tractor, path):
        self.path = path
        self.traction_states = {}
        self.tractor = tractor
        with open(os.path.join(self.path, "-root-.json"), "w") as f:
            f.write(json.dumps(tractor.to_json()))

    def on_update(self, traction):
        if traction.uid not in self.traction_states:
            self.traction_states[traction.uid] = traction.state
            with open(os.path.join(self.path, f"{traction.uid}.json"), "w") as f:
                f.write(json.dumps(traction.to_json()))

        if traction == self.tractor:
            if traction.state == self.traction_states[traction.uid]:
                return
            for f in traction._fields:
                if f.startswith("i_"):
                    fpath = os.path.join(self.path, f"{traction.uid}::{f}.json")
                    with open(fpath, "w") as fp:
                        fp.write(json.dumps(getattr(traction, f).to_json()))

        else:
            if traction.state != self.traction_states[traction.uid]:
                with open(os.path.join(self.path, f"{traction.uid}.json"), "w") as f:
                    f.write(json.dumps(traction.to_json()))

    def close(self, traction):
        with open(os.path.join(self.path, f"{traction.uid}.json"), "w") as f:
            f.write(json.dumps(traction.to_json()))
