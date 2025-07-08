import pytest

from pytractions.base import Field, Port, Base, TList
from pytractions.traction import Traction
from pytractions.exc import NoDefaultError


class Nested(Base):
    """Test class."""

    x: int


class OutContainerNoDef(Base):
    """Out container with no default value."""

    out: str = "x"
    nested: Nested = Field(default_factory=Nested)


class OutContainer(Base):
    """Out container with some default value."""

    out: str = "defaultstr"


def test_traction_out_no_def():
    with pytest.raises(NoDefaultError):

        class TestTractionOutNoDef(Traction):
            """Test Traction."""

            o_out: Port[OutContainerNoDef]

            def _run(self) -> None:
                pass


class TestTraction(Traction):
    """Test Traction."""

    o_out: Port[OutContainer]

    def _run(self) -> None:
        pass


class TestTractionOutList(Traction):
    """Test Traction with list output."""

    o_out: Port[TList[str]]

    def _run(self) -> None:
        pass


def test_out_container_default():
    out = Port[OutContainer]()
    assert out.data is None

    t = TestTraction(uid="test")
    assert t.o_out == OutContainer()


def test_out_tlist():
    t = TestTractionOutList(uid="test")
    assert t.o_out == TList[str]([])
