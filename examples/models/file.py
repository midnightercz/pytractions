from typing import Optional
from pytraction.base import Base
from ..operators import Operator

class File(Base):
    filename: str
    size: int
    atime: int
    mtime: int
    owner: int
    group: int


class FileFilter(Base):
    filename: Optional[Operator[str]]
    size: Optional[Operator[int]]
    atime: Optional[Operator[int]]
    mtime: Optional[Operator[int]]
    owner: Optional[Operator[int]]
    group: Optional[Operator[int]]


