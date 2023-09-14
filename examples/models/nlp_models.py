from typing import Optional, Union
from pytraction.base import Base
from ..operators import Operator


class ENDSENTENCE(Base):
    pass


class STOP(Base):
    pass


class PUNC(Base):
    pass


class Doc(Base):
    words: TList[Union[str, SENTENCEEND, STOP, PUNC]]
