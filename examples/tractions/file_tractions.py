from typing import Optional, Type

from ..models.file import File, FileFilter, STMD

from pytraction.base import Traction, TList, In, Out


class FilterFile(Traction):
    i_file: In[File]
    i_filters: In[TList[FileFilter]]
    o_file: Out[Optional[File]]

    def run(self):
        self.o_file.data = None
        for ff in self.i_filters.data:
            if ff.filename and not ff.compare(ff.filename):
                return
            if ff.size and ff.size.compare(self.i_file.data.size):
                return
            if ff.atime and ff.atime.compare(self.i_file.data.atime):
                return
            if ff.mtime and ff.mtime.compare(self.i_file.data.mtime):
                return
            if ff.owner and ff.owner.compare(self.i_file.data.owner):
                return
            if ff.group and ff.group.compare(self.i_file.data.group):
                return
        self.o_file.data = self.i_file.data


class STMDFileFilter(STMD):
    _traction: Type = FilterFile
    i_file: In[TList[In[File]]]
    i_filters: In[TList[In[TList[FileFilter]]]]
    o_file: Out[TList[Out[Optional[File]]]]
