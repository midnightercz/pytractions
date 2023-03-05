class LoadWrongStepError(Exception):
    pass

class LoadWrongExtResourceError(Exception):
    pass

class MissingSecretError(KeyError):
    pass

class DuplicateStepError(Exception):
    pass

class DuplicateTractorError(Exception):
    pass

class TractionFailedError(Exception):
    """Exception indidating failure of a step."""
