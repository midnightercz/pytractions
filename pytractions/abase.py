from abc import abstractmethod
from typing import Any, Dict
from typing_extensions import Self


class ABase:

    @abstractmethod
    def __post_init__(self):  # pragma: no cover
        ...

    @abstractmethod
    def _no_validate_setattr_(self, name: str, value: Any) -> None:  # pragma: no cover
        ...

    @abstractmethod
    def _validate_setattr_(self, name: str, value: Any) -> None:  # pragma: no cover
        ...

    @abstractmethod
    def _replace_generic_cache(cls, type_, new_type):  # pragma: no cover
        ...

    @abstractmethod
    def _make_qualname(cls, params):  # pragma: no cover
        ...

    @abstractmethod
    def __class_getitem__(cls, param, params_map={}):  # pragma: no cover
        ...

    @abstractmethod
    def to_json(self) -> Dict[str, Any]:  # pragma: no cover
        ...

    @abstractmethod
    def content_to_json(self) -> Dict[str, Any]:  # pragma: no cover
        ...

    @abstractmethod
    def type_to_json(cls) -> Dict[str, Any]:  # pragma: no cover
        ...

    @abstractmethod
    def from_json(cls, json_data) -> Self:  # pragma: no cover
        ...
