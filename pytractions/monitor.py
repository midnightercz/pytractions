import json
import os
import urllib.parse

import redis

from .base import Base


# class StructuredMonitor:
#     """Monitor traction runs."""
#
#     def __init__(self, tractor, path):
#         """Initialize the monitor."""
#         self.path = path
#         self.traction_states = {}
#         self.tractor = tractor
#         with open(os.path.join(self.path, "-root-.json"), "w") as f:
#             f.write(json.dumps(tractor.to_json()))
#
#     def on_update(self, traction):
#         """Dump updated traction to output directory."""
#         if traction.uid not in self.traction_states:
#             self.traction_states[traction.uid] = traction.state
#             with open(os.path.join(self.path, f"{traction.uid}.json"), "w") as f:
#                 f.write(json.dumps(traction.to_json()))
#
#         if traction == self.tractor:
#             if traction.state == self.traction_states[traction.uid]:
#                 return
#             for f in traction._fields:
#                 if f.startswith("i_"):
#                     fpath = os.path.join(self.path, f"{traction.uid}::{f}.json")
#                     with open(fpath, "w") as fp:
#                         fp.write(json.dumps(getattr(traction, "_raw_" + f).to_json()))
#
#         else:
#             if traction.state != self.traction_states[traction.uid]:
#                 with open(os.path.join(self.path, f"{traction.uid}.json"), "w") as f:
#                     f.write(json.dumps(traction.to_json()))
#
#     def close(self, traction):
#         """Close the monitoring and dump the root tractor."""
#         with open(os.path.join(self.path, f"{traction.uid}.json"), "w") as f:
#             f.write(json.dumps(traction.to_json()))


class NoObserver(Base):
    """Class which does nothing, used as a placeholder for no observers."""

    root: str

    def load_config(self, config):
        """Load configuration from JSON string."""
        pass

    def _observed(self, path, value, extra=None):
        """Do nothing on observed event."""
        pass

    def setup(self, traction):
        """Prepare the observer."""
        pass


class FileObserver(Base):
    """Class which stores observed events into files."""

    root: str
    output_dir: str = "output"

    def load_config(self, config):
        """Load configuration from JSON string."""
        parsed = json.loads(config)
        self.output_dir = parsed["path"]

    def _observed(self, path, value, extra=None):
        encoded_path = urllib.parse.quote(path, safe="")
        with open(os.path.join(self.output_dir, encoded_path), "w") as f:
            if hasattr(value, "content_to_json"):
                f.write(json.dumps(value.content_to_json()))
            else:
                f.write(json.dumps(value))

    def setup(self, traction):
        """Prepare the observer."""
        pass


class RedisObserver(Base):
    """Class which stores observed events into redis database."""

    redis_url: str = "localhost"
    port: int = 6379
    root: str = "root"
    extra_only: bool = False

    def __reduce__(self):
        """Reduce the class for parallelization purposes."""
        return (
            self.__class__,
            (),
            {
                "url": self.redis_url,
                "port": self.port,
                "root": self.root,
                "extra_only": self.extra_only,
            },
        )

    def setup(self, traction):
        """Prepare the observer."""
        self.redis.set(self.root, json.dumps(traction.content_to_json()))
        self.redis.xadd(
            "timeline-%s" % self.root,
            {"path": self.root, "value": json.dumps(traction.content_to_json())},
        )

    def load_config(self, config):
        """Load configuration from JSON string."""
        parsed = json.loads(config)
        self.redis_url = parsed.get("url", self.redis_url)
        self.port = parsed.get("port", self.port)
        self.extra_only = parsed.get("extra_only", self.extra_only)

    @property
    def redis(self):
        """Access the redis client."""
        if not hasattr(self, "_redis"):
            self._redis = redis.Redis(host=self.redis_url, port=self.port, decode_responses=True)
        return self._redis

    def _observed(self, path, value, extra=None):
        if self.extra_only:
            if extra:
                # print("EXTRA", extra, value, path)
                if hasattr(value, "content_to_json"):
                    self.redis.set(path, json.dumps(extra))
                    self.redis.xadd(
                        "timeline-%s" % self.root,
                        {
                            "path": path,
                            "extra": json.dumps(extra),
                            "value": json.dumps(value.content_to_json()),
                        },
                    )
                else:
                    self.redis.set(path, json.dumps(extra))
                    self.redis.xadd(
                        "timeline-%s" % self.root,
                        {"path": path, "extra": json.dumps(extra), "value": json.dumps(value)},
                    )
            else:
                return

        if hasattr(value, "content_to_json"):
            self.redis.set(path, json.dumps(value.content_to_json()))
            self.redis.xadd(
                "timeline-%s" % self.root,
                {"path": path, "value": json.dumps(value.content_to_json())},
            )
        else:
            self.redis.set(path, json.dumps(value))
            self.redis.xadd("timeline-%s" % self.root, {"path": path, "value": json.dumps(value)})


OBSERVERS = {"file": FileObserver, "redis": RedisObserver, "none": NoObserver}
