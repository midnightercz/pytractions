import json
from typing import List, Generic, TypeVar
import os

from pytraction.base import Res, Base

T = TypeVar('T')


class JSONStore(Base, Generic[T]):
    filename: str

    def store(self, data: T):
        with open(self.filename, 'w') as f:
            json.dump(data.to_json(), f)

    def load(self) -> T:
        with open(self.filename, 'r') as f:
            return json.load(f)
