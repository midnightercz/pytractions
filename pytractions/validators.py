class LiteralValidator:
    """Literal validator."""

    def __init__(self, literal, field_name):
        """Initialize."""
        self.literal = literal
        self.field_name = field_name

    def __call__(self, value):
        """Validate the value against the literal."""
        if value != self.literal and value not in self.literal:
            raise ValueError(f"Value {self.field_name} must be in {self.literal}")
