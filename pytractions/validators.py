class LiteralValidator:
    def __init__(self, literal):
        self.literal = literal

    def __call__(self, value):
        if value != self.literal and value not in self.literal:
            raise ValueError(f"Value must be in {self.literal}")
