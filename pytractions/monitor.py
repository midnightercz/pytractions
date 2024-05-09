import json
import os


class StructuredMonitor:
    """Monitor traction runs."""

    def __init__(self, tractor, path):
        """Initialize the monitor."""
        self.path = path
        self.traction_states = {}
        self.tractor = tractor
        with open(os.path.join(self.path, "-root-.json"), "w") as f:
            f.write(json.dumps(tractor.to_json()))

    def on_update(self, traction):
        """Dump updated traction to output directory."""
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
        """Close the monitoring and dump the root tractor."""
        with open(os.path.join(self.path, f"{traction.uid}.json"), "w") as f:
            f.write(json.dumps(traction.to_json()))
