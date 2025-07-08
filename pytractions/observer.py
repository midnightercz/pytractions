class Observer:
    """Observer abstract class."""

    def updated(self, uid, attr, value):
        """Process data when observed attribute is updated."""
        pass
