import logging
import redis


class RedisStreamHandler(logging.StreamHandler):
    """Custom logging handler to send logs to Redis."""

    def __init__(self, redis_url="localhost", port=6379, stream_id="logs"):
        """Initialize RedisStreamHandler."""
        super().__init__()
        self.redis_url = redis_url
        self.port = port
        self.stream_id = stream_id
        self.redis = redis.Redis(host=self.redis_url, port=self.port, decode_responses=True)

    def emit(self, record):
        """Emit a log record to Redis."""
        log_entry = self.format(record)
        self.redis.xadd(self.stream_id, {"log": log_entry, "level": record.levelno})
