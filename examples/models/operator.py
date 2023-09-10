import abc
import re

from typing import Generic, TypeVar
from pytraction.base import Base

T = TypeVar('T')


class Operator(Base, Generic[T]):
    v: T

    @abc.abstractmethod
    def compare(self, o: T):
        ...


class EQ(Operator[T], Generic[T]):
    def compare(self, o: T):
        return self.v == o


class NEQ(Operator[T], Generic[T]):
    def compare(self, o: T):
        return self.v != o


class LT(Operator[T], Generic[T]):
    def compare(self, o: T):
        return self.v < o


class GT(Operator[T], Generic[T]):
    def compare(self, o: T):
        return self.v > o


class LTE(Operator[T], Generic[T]):
    def compare(self, o: T):
        return self.v <= o


class GTE(Operator[T], Generic[T]):
    def compare(self, o: T):
        return self.v >= o


class MATCH(Operator[T], Generic[T]):
    r: str
    def compare(self, o: T):
        return re.compile(self.r).match(o)
