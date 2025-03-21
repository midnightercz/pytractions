from pytractions.base import Base, Field, TList, TDict, Port, NullPort
from pytractions.traction import Traction, TractionState, TractionStats
from pytractions.tractor import Tractor
from pytractions.stmd import STMD


class TestTraction(Traction):

    i_in: int
    o_out: int

    def _run(self):
        self.o_out = self.i_in + 1


class NestedData(Base):
    j: int = 0


class ComplexData(Base):
    i: int = 0
    nested: NestedData = Field(default_factory=NestedData)


class TestTractionComplex(Traction):

    i_in: ComplexData
    o_out: ComplexData

    def _run(self):
        print("------> test traction run")
        self.o_out.i = self.i_in.i + 1
        print("-1-")
        self.o_out.nested.j += self.i_in.nested.j + 1
        print("-2-")
        print("NESTED J", self.o_out.nested.j)
        self.o_out.nested = NestedData(j=self.o_out.nested.j)
        print("-3-")
        self.o_out.nested.j += 1
        print("-4-")
        print("NESTED J", self.o_out.nested.j)
        print("<-------")


class TestTractionList(Traction):

    i_in: TList[int]
    o_out: TList[int]

    def _run(self):
        for i in self.i_in:
            self.o_out.append(i + 1)


class TestTractionDict(Traction):

    i_in: TDict[str, str]
    o_out: TDict[str, str]

    def _run(self):
        for k, v in self.i_in.items():
            self.o_out[k] = v


class TestTractor(Tractor):
    i_in: Port[ComplexData] = NullPort[ComplexData]()
    t_complex: TestTractionComplex = TestTractionComplex(uid="t_complex",
                                                         i_in=i_in)
    o_tractor_out: Port[ComplexData] = t_complex.o_out


TestTractionSTMD = STMD.wrap(TestTraction)


class Observer:
    def __init__(self):
        self.observed_calls = []

    def _observed(self, *args, extra=None):
        self.observed_calls.append((args, extra))


def test_observed(fixture_isodate_now):
    t = TestTraction(uid="test-traction", i_in=1)
    o = Observer()
    t._observer._observers[id(o)] = (o, "test-traction")
    t.run()
    assert o.observed_calls == [
        (("test-traction.stats", t.stats), {}),
        (("test-traction.stats.started", "1990-01-01T00:00:00.00000Z"), {}),
        (("test-traction.state", TractionState.PREP),
         {"traction_state_changed": {"state": TractionState.PREP,
                                     "traction": "TestTraction[test-traction]"}}
        ),
        (("test-traction.state", TractionState.RUNNING),
         {"traction_state_changed": {"state": TractionState.RUNNING,
                                     "traction": "TestTraction[test-traction]"}}
        ),
        (("test-traction.o_out", 2), {}),
        (("test-traction.o_out", 2), {}),
        (("test-traction.state", TractionState.FINISHED),
         {"traction_state_changed": {"state": TractionState.FINISHED,
                                     "traction": "TestTraction[test-traction]"}}
        ),
        (("test-traction.stats.finished", "1990-01-01T00:00:01.00000Z"), {}),
        (("test-traction.stats.skipped", False), {})
    ]


def test_observed_complex(fixture_isodate_now):
    t = TestTractionComplex(uid="test-traction", i_in=ComplexData(i=1, nested=NestedData(j=1)))
    o = Observer()
    t._observer._observers[id(o)] = (o, "test-traction")
    t.run()
    assert o.observed_calls == [
        (("test-traction.stats", t.stats), {}),
        (("test-traction.stats.started", "1990-01-01T00:00:00.00000Z"), {}),
        (("test-traction.state", TractionState.PREP),
         {"traction_state_changed": {"state": TractionState.PREP,
                                     "traction": "TestTractionComplex[test-traction]"}}
        ),
        (("test-traction.state", TractionState.RUNNING),
         {"traction_state_changed": {"state": TractionState.RUNNING,
                                     "traction": "TestTractionComplex[test-traction]"}}
        ),
        (("test-traction.o_out.i", 2), {}),
        (("test-traction.o_out.nested.j", 2), {}),
        (("test-traction.o_out.nested", NestedData(j=3)), {}),
        (("test-traction.o_out.nested.j", 3), {}),
        (("test-traction.state", TractionState.FINISHED),
         {"traction_state_changed": {"state": TractionState.FINISHED,
                                     "traction": "TestTractionComplex[test-traction]"}}
        ),
        (("test-traction.stats.finished", "1990-01-01T00:00:01.00000Z"), {}),
        (("test-traction.stats.skipped", False), {})
    ]


def test_observed_list(fixture_isodate_now):
    t = TestTractionList(uid="test-traction", i_in=TList[int]([1, 2, 3, 4]))
    o = Observer()
    t._observer._observers[id(o)] = (o, "test-traction")
    t.run()
    assert o.observed_calls == [
        (("test-traction.stats", t.stats), {}),
        (("test-traction.stats.started", "1990-01-01T00:00:00.00000Z"), {}),
        (("test-traction.state", TractionState.PREP),
         {"traction_state_changed": {"state": TractionState.PREP,
                                     "traction": "TestTractionList[test-traction]"}}
        ),
        (("test-traction.state", TractionState.RUNNING),
         {"traction_state_changed": {"state": TractionState.RUNNING,
                                     "traction": "TestTractionList[test-traction]"}}
        ),
        (("test-traction.o_out.[0]", 2), {}),
        (("test-traction.o_out.[1]", 3), {}),
        (("test-traction.o_out.[2]", 4), {}),
        (("test-traction.o_out.[3]", 5), {}),
        (("test-traction.state", TractionState.FINISHED),
         {"traction_state_changed": {"state": TractionState.FINISHED,
                                     "traction": "TestTractionList[test-traction]"}}
        ),
        (("test-traction.stats.finished", "1990-01-01T00:00:01.00000Z"), {}),
        (("test-traction.stats.skipped", False), {})
    ]


def test_observed_dict(fixture_isodate_now):
    t = TestTractionDict(uid="test-traction", i_in=TDict[str, str]({"foo": "bar", "baz": "qux"}))
    o = Observer()
    t._observer._observers[id(o)] = (o, "test-traction")
    t.run()
    assert o.observed_calls == [
        (("test-traction.stats", t.stats), {}),
        (("test-traction.stats.started", "1990-01-01T00:00:00.00000Z"), {}),
        (("test-traction.state", TractionState.PREP),
         {"traction_state_changed": {"state": TractionState.PREP,
                                     "traction": "TestTractionDict[test-traction]"}}
        ),
        (("test-traction.state", TractionState.RUNNING),
         {"traction_state_changed": {"state": TractionState.RUNNING,
                                     "traction": "TestTractionDict[test-traction]"}}
        ),
        (('test-traction.o_out.["foo"]', "bar"), {}),
        (('test-traction.o_out.["baz"]', "qux"), {}),
        (("test-traction.state", TractionState.FINISHED),
         {"traction_state_changed": {"state": TractionState.FINISHED,
                                     "traction": "TestTractionDict[test-traction]"}}
        ),
        (("test-traction.stats.finished", "1990-01-01T00:00:01.00000Z"), {}),
        (("test-traction.stats.skipped", False), {})
    ]


def test_observed_tractor(fixture_isodate_now):
    i_in = ComplexData(i=1, nested=NestedData(j=1))
    t = TestTractor(uid="test-tractor", i_in=i_in)
    o = Observer()
    t._observer._observers[id(o)] = (o, "test-tractor")
    t.run()
    assert o.observed_calls == [
        (("test-tractor.stats", t.stats), {}),
        (("test-tractor.stats.started", "1990-01-01T00:00:00.00000Z"), {}),
        (("test-tractor.state", TractionState.PREP),
         {"traction_state_changed": {"state": TractionState.PREP,
                                     "traction": "TestTractor[test-tractor]"}}
        ),
        (("test-tractor.state", TractionState.RUNNING),
         {"traction_state_changed": {"state": TractionState.RUNNING,
                                     "traction": "TestTractor[test-tractor]"}}
        ),
        (('test-tractor.tractions', t.tractions), {}),
        (('test-tractor.tractions.["t_complex"]', t.tractions['t_complex']), {}),
        (('test-tractor.o_tractor_out', t._raw_o_tractor_out), {}),
        (('test-tractor.tractions.["t_complex"].stats', t.tractions["t_complex"].stats), {}),
        (('test-tractor.tractions.["t_complex"].stats.started', "1990-01-01T00:00:00.00000Z"), {}),
        (('test-tractor.tractions.["t_complex"].state', TractionState.PREP),
         {"traction_state_changed": {"state": TractionState.PREP,
                                     "traction": "TestTractionComplex[test-tractor::t_complex]"}}
        ),
        (('test-tractor.tractions.["t_complex"].state', TractionState.RUNNING),
         {"traction_state_changed": {"state": TractionState.RUNNING,
                                     "traction": "TestTractionComplex[test-tractor::t_complex]"}}
        ),
        (('test-tractor.tractions.["t_complex"].o_out.i', 2), {}),
        (('test-tractor.tractions.["t_complex"].o_out.nested.j', 2), {}),
        (('test-tractor.tractions.["t_complex"].o_out.nested', NestedData(j=3)), {}),
        (('test-tractor.tractions.["t_complex"].o_out.nested.j', 3), {}),
        (('test-tractor.tractions.["t_complex"].state', TractionState.FINISHED),
         {"traction_state_changed": {"state": TractionState.FINISHED,
                                     "traction": "TestTractionComplex[test-tractor::t_complex]"}}
        ),
        (('test-tractor.tractions.["t_complex"].stats.finished', "1990-01-01T00:00:01.00000Z"), {}),
        (('test-tractor.tractions.["t_complex"].stats.skipped', False), {}),
        (('test-tractor.state', TractionState.FINISHED),
         {"traction_state_changed": {"state": TractionState.FINISHED,
                                     "traction": "TestTractor[test-tractor]"}}
        ),
        (('test-tractor.stats.finished', "1990-01-01T00:00:02.00000Z"), {}),
        (('test-tractor.stats.skipped', False), {})
    ]


def test_observed_stmd(fixture_isodate_now):

    i_in1 = TList[int]([1, 2, 3, 4])
    t = TestTractionSTMD(uid="test-traction", i_in=i_in1)
    o = Observer()
    t._observer._observers[id(o)] = (o, "test-stmd")
    t.run()

    assert t.o_out == TList[int]([2, 3, 4, 5])

    assert o.observed_calls == [
        (("test-stmd.stats", t.stats), {}),
        (("test-stmd.stats.started", "1990-01-01T00:00:00.00000Z"), {}),
        (("test-stmd.tractions", t.tractions), {}),
        (("test-stmd.tractions.[0]", t.tractions[0]), {}),
        (("test-stmd.tractions.[1]", t.tractions[1]), {}),
        (("test-stmd.tractions.[2]", t.tractions[2]), {}),
        (("test-stmd.tractions.[3]", t.tractions[3]), {}),

        (("test-stmd.tractions_state", TList[TractionState]([
            TractionState.READY, TractionState.READY,
            TractionState.READY, TractionState.READY
        ])), {}),
        (("test-stmd.o_out.[0]", 0), {}),
        (("test-stmd.o_out.[1]", 0), {}),
        (("test-stmd.o_out.[2]", 0), {}),
        (("test-stmd.o_out.[3]", 0), {}),
        #("test-stmd.state", TractionState.PREP),
        (("test-stmd.state", TractionState.RUNNING),
         {"traction_state_changed": {"state": TractionState.RUNNING,
                                     "traction": "STMDTestTraction[test-traction]"}}
        ),
        (("test-stmd.tractions.[0].stats", TractionStats(
            started="1990-01-01T00:00:00.00000Z",
            finished="1990-01-01T00:00:01.00000Z",
            skipped=False)
         ), {}),
        (("test-stmd.tractions.[0].stats.started", "1990-01-01T00:00:00.00000Z"), {}),
        (("test-stmd.tractions.[0].state", TractionState.PREP),
         {"traction_state_changed": {"state": TractionState.PREP,
                                     "traction": "TestTraction[test-traction:0]"}}
        ),
        (("test-stmd.tractions.[0].state", TractionState.RUNNING),
         {"traction_state_changed": {"state": TractionState.RUNNING,
                                     "traction": "TestTraction[test-traction:0]"}}
        ),
        (("test-stmd.tractions.[0].o_out", 2), {}),
        (("test-stmd.tractions.[0].o_out", 2), {}),
        (("test-stmd.tractions.[0].state", TractionState.FINISHED),
         {"traction_state_changed": {"state": TractionState.FINISHED,
                                     "traction": "TestTraction[test-traction:0]"}}
        ),
        (("test-stmd.tractions.[0].stats.finished", "1990-01-01T00:00:01.00000Z"), {}),
        (("test-stmd.tractions.[0].stats.skipped", False), {}),

        (("test-stmd.tractions.[1].stats", TractionStats(
            started="1990-01-01T00:00:02.00000Z",
            finished="1990-01-01T00:00:03.00000Z",
            skipped=False)
         ), {}),
        (("test-stmd.tractions.[1].stats.started", "1990-01-01T00:00:02.00000Z"), {}),
        (("test-stmd.tractions.[1].state", TractionState.PREP),
         {"traction_state_changed": {"state": TractionState.PREP,
                                     "traction": "TestTraction[test-traction:1]"}}
        ),
        (("test-stmd.tractions.[1].state", TractionState.RUNNING),
         {"traction_state_changed": {"state": TractionState.RUNNING,
                                     "traction": "TestTraction[test-traction:1]"}}
        ),
        (("test-stmd.tractions.[1].o_out", 3), {}),
        (("test-stmd.tractions.[1].o_out", 3), {}),
        (("test-stmd.tractions.[1].state", TractionState.FINISHED),
         {"traction_state_changed": {"state": TractionState.FINISHED,
                                     "traction": "TestTraction[test-traction:1]"}}
        ),
        (("test-stmd.tractions.[1].stats.finished", "1990-01-01T00:00:03.00000Z"), {}),
        (("test-stmd.tractions.[1].stats.skipped", False), {}),

        (("test-stmd.tractions.[2].stats", TractionStats(
            started="1990-01-01T00:00:04.00000Z",
            finished="1990-01-01T00:00:05.00000Z",
            skipped=False)
         ), {}),
        (("test-stmd.tractions.[2].stats.started", "1990-01-01T00:00:04.00000Z"), {}),
        (("test-stmd.tractions.[2].state", TractionState.PREP),
         {"traction_state_changed": {"state": TractionState.PREP,
                                     "traction": "TestTraction[test-traction:2]"}}
        ),
        (("test-stmd.tractions.[2].state", TractionState.RUNNING),
         {"traction_state_changed": {"state": TractionState.RUNNING,
                                     "traction": "TestTraction[test-traction:2]"}}
        ),
        (("test-stmd.tractions.[2].o_out", 4), {}),
        (("test-stmd.tractions.[2].o_out", 4), {}),
        (("test-stmd.tractions.[2].state", TractionState.FINISHED),
         {"traction_state_changed": {"state": TractionState.FINISHED,
                                     "traction": "TestTraction[test-traction:2]"}}
        ),
        (("test-stmd.tractions.[2].stats.finished", "1990-01-01T00:00:05.00000Z"), {}),
        (("test-stmd.tractions.[2].stats.skipped", False), {}),

        (("test-stmd.tractions.[3].stats", TractionStats(
            started="1990-01-01T00:00:06.00000Z",
            finished="1990-01-01T00:00:07.00000Z",
            skipped=False)
         ), {}),
        (("test-stmd.tractions.[3].stats.started", "1990-01-01T00:00:06.00000Z"), {}),
        (("test-stmd.tractions.[3].state", TractionState.PREP),
         {"traction_state_changed": {"state": TractionState.PREP,
                                     "traction": "TestTraction[test-traction:3]"}}
        ),
        (("test-stmd.tractions.[3].state", TractionState.RUNNING),
         {"traction_state_changed": {"state": TractionState.RUNNING,
                                     "traction": "TestTraction[test-traction:3]"}}
        ),
        (("test-stmd.tractions.[3].o_out", 5), {}),
        (("test-stmd.tractions.[3].o_out", 5), {}),
        (("test-stmd.tractions.[3].state", TractionState.FINISHED),
         {"traction_state_changed": {"state": TractionState.FINISHED,
                                     "traction": "TestTraction[test-traction:3]"}}
        ),
        (("test-stmd.tractions.[3].stats.finished", "1990-01-01T00:00:07.00000Z"), {}),
        (("test-stmd.tractions.[3].stats.skipped", False), {}),
        (("test-stmd.o_out.[\"0\"]", 2), {}),
        (("test-stmd.o_out.[\"1\"]", 3), {}),
        (("test-stmd.o_out.[\"2\"]", 4), {}),
        (("test-stmd.o_out.[\"3\"]", 5), {}),

        (('test-stmd.state', TractionState.FINISHED),
         {"traction_state_changed": {"state": TractionState.FINISHED,
                                     "traction": "STMDTestTraction[test-traction]"}}
        ),
        (('test-stmd.stats.finished', "1990-01-01T00:00:08.00000Z"), {}),
        (('test-stmd.stats.skipped', False), {})
    ]
