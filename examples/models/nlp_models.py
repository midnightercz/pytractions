from typing import Optional, Union
from pytraction.base import Base, TList


class ENDSENTENCE(Base):
    pass


class STOP(Base):
    pass


class PUNC(Base):
    pass


class Doc(Base):
    words: TList[Union[str, ENDSENTENCE, STOP, PUNC]]


class Phrase(Base):
    words: TList[int]
    score: float = 0.0

class TextPhrase(Base):
    words: TList[str]
    score: float = 0.0
class Word(Base):
    wid: int
    count: int = 0
