import abc
from typing import List
import os
from pathlib import Path

from pytraction.base import Res, Base

from ..models.file import File


class DirLister(Base):
    @abc.abstractmethod
    def ls(self, path):
        ...


class LocalDirLister(DirLister):
    def ls(self, path) -> List[File]:
        out: List[File] = []
        p = Path(path)
        for f in p.iterdir():
            fstat = os.stat(f)
            fm = File(
                filename=f.name,
                size=fstat.st_size,
                atime=fstat.st_atime,
                mtime=fstat.st_mtime,
            )
            out.append(fm)
        return out
