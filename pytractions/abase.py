from abc import abstractmethod
from typing import Any, Dict, Generic, TypeVar, Tuple


class ABase:
    """Abstract base class."""

    @abstractmethod
    def __post_init__(self):  # pragma: no cover
        """Initialize object after construction."""
        ...

    @abstractmethod
    def _no_validate_setattr_(self, name: str, value: Any) -> None:  # pragma: no cover
        """Set attribute with no validation."""
        ...

    @abstractmethod
    def _validate_setattr_(self, name: str, value: Any) -> None:  # pragma: no cover
        """Set attribute with type validation."""
        ...

    @abstractmethod
    def _replace_generic_cache(cls, type_, new_type):  # pragma: no cover
        """Replace generic cache."""
        ...

    @abstractmethod
    def _make_qualname(cls, params):  # pragma: no cover
        """Make qualname for the class."""
        ...

    @abstractmethod
    def __class_getitem__(cls, param, params_map={}):  # pragma: no cover
        """Construct generic subclass."""
        ...

    @abstractmethod
    def to_json(self) -> Dict[str, Any]:  # pragma: no cover
        """Serialize the class to json."""
        ...

    @abstractmethod
    def content_to_json(self) -> Dict[str, Any]:  # pragma: no cover
        """Serialize content of the class to json."""
        ...

    @abstractmethod
    def type_to_json(cls) -> Dict[str, Any]:  # pragma: no cover
        """Serialize type of the class to json."""
        ...

    @abstractmethod
    def from_json(cls, json_data) -> "ABase":  # pragma: no cover
        """Deserialize class from a json data."""
        ...


T = TypeVar("T")
TT = TypeVar("TT")


class ATList(ABase, Generic[T]):
    """Abstract TList class."""

    @abstractmethod
    def __add__(self, value):
        """Return two lists joined together."""
        ...

    @abstractmethod
    def __contains__(self, value):
        """Test if list contains a value."""
        ...

    @abstractmethod
    def __delitem__(self, x):
        """Remove item from the list."""
        ...

    @abstractmethod
    def __getitem__(self, x):
        """Return item from the list."""
        ...

    @abstractmethod
    def __iter__(self):
        """Iterate over the list."""
        ...

    @abstractmethod
    def __len__(self):
        """Return length of the list."""
        ...

    @abstractmethod
    def __reversed__(self):
        """Return __reversed__ of the list."""
        ...

    @abstractmethod
    def __setitem__(self, key, value):
        """Set item to the list."""
        ...

    @abstractmethod
    def append(self, obj: T) -> None:
        """Append item to the end of the list."""
        ...

    @abstractmethod
    def clear(self):
        """Clear the list."""
        ...

    @abstractmethod
    def count(self, value):
        """Return number of occurences of the value in the list."""
        ...

    @abstractmethod
    def extend(self, iterable):
        """Extend list if the given iterable."""
        ...

    @abstractmethod
    def index(self, value, start=0, stop=-1):
        """Return index of the value in the list."""
        ...

    @abstractmethod
    def insert(self, index, obj):
        """Insert item to the list."""
        ...

    @abstractmethod
    def pop(self, *args, **kwargs):
        """Remove and return item from the list."""
        ...

    @abstractmethod
    def remove(self, value):
        """Remove item from the list."""
        ...

    @abstractmethod
    def reverse(self):
        """Return reversed list."""
        ...

    @abstractmethod
    def sort(self, *args, **kwargs):
        """Return sorted list."""
        ...

    @abstractmethod
    def to_json(self) -> Dict[str, Any]:
        """Serialize TList to json representation."""
        ...

    @abstractmethod
    def content_to_json(self) -> Dict[str, Any]:
        """Serialize TList content to json representation."""
        ...

    @abstractmethod
    def from_json(cls, json_data, _locals={}) -> "ABase":
        """Deserialize TList from json data."""
        ...


class ATDict(ABase, Generic[T, TT]):
    """Abstract TDict class."""

    @abstractmethod
    def __contains__(self, key: T) -> bool:
        """Test if dict contains the key."""
        ...

    @abstractmethod
    def __delitem__(self, key: T):
        """Remove item from the dict by given key."""
        ...

    @abstractmethod
    def __getitem__(self, key: T) -> TT:
        """Return item from the dict for given key."""
        ...

    @abstractmethod
    def __iter__(self):
        """Iterate over the dict."""
        ...

    @abstractmethod
    def __len__(self):
        """Return length of the dict."""
        ...

    @abstractmethod
    def __reversed__(self):
        """Return __reversed__ of the dict."""
        ...

    @abstractmethod
    def __setitem__(self, k: T, v: TT):
        """Set item to the dict to given key."""
        ...

    @abstractmethod
    def clear(self):
        """Clear the dict."""
        ...

    @abstractmethod
    def fromkeys(self, iterable, value):
        """Return new TDict instance with items from iterable."""
        ...

    @abstractmethod
    def get(self, key: T, default=None):
        """Get item from the dict or return default if not found."""
        ...

    @abstractmethod
    def items(self):
        """Return items of the dict."""
        ...

    @abstractmethod
    def keys(self):
        """Return keys of the items of the dict."""
        ...

    @abstractmethod
    def pop(self, k: T, d=None):
        """Remove and return item from the dict by given key."""
        ...

    @abstractmethod
    def popitem(self) -> Tuple[T, TT]:
        """Remove item from the dict."""
        ...

    @abstractmethod
    def setdefault(self, key: T, default: TT):
        """Set dict default value for given key."""
        ...

    @abstractmethod
    def update(self, other):
        """Update dict with another dict."""
        ...

    @abstractmethod
    def values(self):
        """Return item values of the dict."""
        ...

    @abstractmethod
    def to_json(self) -> Dict[str, Any]:
        """Serialize TDict to json representation."""
        ...

    @abstractmethod
    def content_to_json(self) -> Dict[str, Any]:
        """Serialize TDict content to json representation."""
        ...

    @abstractmethod
    def from_json(cls, json_data, _locals={}) -> "ABase":
        """Deserialize TDict from json."""
        ...
