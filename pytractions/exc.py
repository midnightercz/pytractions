class TractionFailedError(Exception):
    """Exception indidating failure of a step."""


class UninitiatedResource(Exception):
    """Exception indidating resource is not initiated."""

    def __init__(self, msg):
        """Initialize the exception."""
        self.msg = msg
