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
