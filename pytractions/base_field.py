from dataclasses import Field as dataclass_Field
from dataclasses import MISSING


class Field(dataclass_Field):
    def __init__(
        self,
        *args,
        default=MISSING,
        default_factory=MISSING,
        init=True,
        repr=True,
        hash=None,
        compare=True,
        metadata=None,
        kw_only=MISSING,
        validator=None,
    ) -> None:
        super().__init__(
            *args,
            default=default,
            default_factory=default_factory,
            init=init,
            repr=repr,
            hash=hash,
            compare=compare,
            metadata=metadata,
            kw_only=kw_only,
        )
        self.validator = validator
