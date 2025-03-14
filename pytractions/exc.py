class TractionFailedError(Exception):
    """Exception indidating failure of a step."""


class UninitiatedResource(Exception):
    """Exception indidating resource is not initiated."""

    def __init__(self, msg):
        """Initialize the exception."""
        self.msg = msg


class TractionValidationError(Exception):
    """Exception raised when tractor class is constructed."""


class WrongInputMappingError(Exception):
    """Exception raised when traction input is set to wrong port type."""


class WrongArgMappingError(Exception):
    """Exception raised when traction arg is set to wrong port type."""


class WrongResMappingError(Exception):
    """Exception raised when traction arg is set to wrong port type."""


class NoDefaultError(TypeError):
    """Exception raised when no default value is provided."""


class NoAnnotationError(TypeError):
    """Raised when class attribute is missing an annotation."""

    pass


class JSONIncompatibleError(TypeError):
    """Raised when class contains attributes not compatible with json serialization."""

    pass


class SerializationError(Exception):
    """Raise when it's not possible to deserialize class from given string."""

    pass


class ItemSerializationError(Exception):
    """Raise when it's not possible to deserialize class from given string."""

    pass
