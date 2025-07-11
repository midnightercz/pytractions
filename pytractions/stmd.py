import dataclasses
import inspect
import logging
from typing import Type, Optional, Union, Any, Dict, ClassVar
from types import prepare_class

from .base import (
    Base,
    Field,
    ANY,
    Port,
    NullPort,
    DefaultOut,
    STMDSingleIn,
    TList,
    TDict,
    _defaultNone,
    is_wrapped,
)
from .traction import Traction, TractionMeta, TractionState, TractionStats
from .executor import ProcessPoolExecutor, ThreadPoolExecutor, LoopExecutor, RayExecutor
from .types import TypeNode
from .utils import isodate_now  # noqa: F401
from .stmd_utils import _init_traction


LOGGER = logging.getLogger(__name__)


class STMDMeta(TractionMeta):
    """STMD metaclass."""

    _SELF_ARGS = ["a_executor", "a_delete_after_finished", "a_allow_unset_inputs"]

    @classmethod
    def _attribute_check(cls, attr, type_, all_attrs):
        """Check attributes when creating the class."""
        type_type_node = TypeNode.from_type(type_, subclass_check=False)
        if attr not in (
            "uid",
            "state",
            "skip",
            "skip_reason",
            "errors",
            "stats",
            "details",
            "traction",
            "tractions",
            "tractions_state",
        ):
            if attr == "d_":
                if type_type_node != TypeNode.from_type(str):
                    raise TypeError(f"Attribute {attr} has to be type str, but is {type_}")
            elif attr.startswith("d_"):
                if type_type_node != TypeNode.from_type(str):
                    raise TypeError(f"Attribute {attr} has to be type str, but is {type_}")
                if attr.replace("d_", "", 1) not in all_attrs["__annotations__"]:
                    raise TypeError(
                        f"Attribute {attr.replace('d_', '', 1)} is not defined for "
                        "description {attr}: {all_attrs}"
                    )
            elif not (
                attr.startswith("i_")
                or attr.startswith("o_")
                or attr.startswith("a_")
                or attr.startswith("r_")
            ):
                raise TypeError(f"Attribute {attr} has start with i_, o_, a_, r_ or d_")

    def __new__(cls, name, bases, attrs):
        """Create new STMD class."""
        annotations = attrs.get("__annotations__", {})
        # check if all attrs are in supported types
        for attr, type_ in annotations.items():
            # skip private attributes
            if attr.startswith("_"):
                continue
            cls._attribute_check(attr, type_, attrs)
            if attr.startswith("i_") and attr not in attrs["_traction"]._fields:
                raise ValueError(
                    f"STMD {cls}{name} has attribute {attr} but traction doesn't have input with "
                    "the same name"
                )
            if attr.startswith("o_") and attr not in attrs["_traction"]._fields:
                raise ValueError(
                    f"STMD {cls}{name} has attribute {attr} but traction doesn't have input with "
                    "the same name"
                )
            if attr.startswith("r_") and attr not in attrs["_traction"]._fields:
                raise ValueError(
                    f"STMD {cls}{name} has attribute {attr} but traction doesn't have resource "
                    "with the same name"
                )
            if (
                attr.startswith("a_")
                and attr not in cls._SELF_ARGS
                and attr not in attrs["_traction"]._fields
            ):
                raise ValueError(
                    f"STMD {cls}{name} has attribute {attr} but traction doesn't have argument "
                    "with the same name"
                )

        if "_traction" not in attrs:
            raise ValueError(f"{name}: Missing _traction: Type[<Traction>] = <Traction> definition")

        # record fields to private attribute
        attrs["_attrs"] = attrs
        attrs["_fields"] = {
            k: v for k, v in attrs.get("__annotations__", {}).items() if not k.startswith("_")
        }

        for f, ftype in attrs["_fields"].items():
            # Do not include outputs in init
            if f.startswith("o_") and f not in attrs:
                if is_wrapped(ftype):
                    _ftype = ftype._params[0]
                else:
                    _ftype = ftype
                if inspect.isclass(_ftype) and issubclass(_ftype, Base):
                    for ff, ft in _ftype._fields.items():
                        df = _ftype.__dataclass_fields__[f]
                        if (
                            df.default is dataclasses.MISSING
                            and df.default_factory is dataclasses.MISSING
                        ):
                            raise TypeError(
                                f"Cannot use {_ftype} for output, as it "
                                f"doesn't have default value for field {ff}"
                            )
                if is_wrapped(ftype):
                    attrs[f] = Field(
                        init=False,
                        default_factory=DefaultOut(type_=ftype._params[0], params=ftype._params),
                    )
                else:
                    attrs[f] = Field(
                        init=False,
                        default_factory=DefaultOut(type_=ftype, params=(ftype,)),
                    )

            # Set all inputs to NullPort after as default
            if f.startswith("i_") and f not in attrs:
                attrs[f] = Field(default_factory=NullPort[ftype._params])

        attrs["_fields"] = {
            k: v for k, v in attrs.get("__annotations__", {}).items() if not k.startswith("_")
        }
        cls._before_new(name, attrs, bases)
        ret = super().__new__(cls, name, bases, attrs)

        return ret

    @classmethod
    def _before_new(cls, name, attrs, bases):
        """Adjust class attributes before class creation."""
        outputs_map = []
        inputs_map = {}
        resources_map = {}
        args_map = {}
        for f, fo in attrs.items():
            if f.startswith("i_"):
                inputs_map[f] = id(fo)
                outputs_map.append(id(fo))
            if f.startswith("r_"):
                resources_map[f] = id(fo)
            if f.startswith("a_"):
                args_map[f] = id(fo)

        attrs["_inputs_map"] = inputs_map
        attrs["_resources_map"] = resources_map
        attrs["_args_map"] = args_map


_loop_executor = LoopExecutor(executor_type="loop_executor")


class STMD(Traction, metaclass=STMDMeta):
    """STMD class."""

    _TYPE: ClassVar[str] = "STMD"
    uid: str
    state: str = TractionState.READY
    skip: bool = False
    skip_reason: Optional[str] = ""
    errors: TList[str] = Field(default_factory=TList[str])
    stats: TractionStats = Field(default_factory=TractionStats)
    details: TDict[str, str] = Field(default_factory=TDict[str, str])
    _traction: Type[Traction] = Traction
    a_delete_after_finished: Port[bool] = Port[bool](data=True)
    a_allow_unset_inputs: Port[bool] = Port[bool](data=False)
    a_executor: Port[Union[ProcessPoolExecutor, ThreadPoolExecutor, LoopExecutor, RayExecutor]] = (
        Port[Union[ProcessPoolExecutor, ThreadPoolExecutor, LoopExecutor, RayExecutor]](
            data=_loop_executor
        )
    )
    tractions: TList[Union[Traction, None]] = Field(default_factory=TList[Optional[Traction]])
    tractions_state: TList[TractionState] = Field(default_factory=TList[TractionState])

    _wrap_cache: ClassVar[Dict[str, Type[Any]]] = {}

    @classmethod
    def wrap(cls, clstowrap, single_inputs={}):
        """Wrap Traction class into STMD class."""
        if not inspect.isclass(clstowrap) and not issubclass(clstowrap, Traction):
            raise ValueError("Can only wrap Traction classes")

        if f"STMD{clstowrap.__name__}" in cls._wrap_cache:
            return cls._wrap_cache[f"STMD{clstowrap.__name__}"]

        attrs = {}
        annotations = {"_traction": Type[Traction]}
        for k, v in clstowrap._fields.items():
            if k.startswith("i_"):
                if k in single_inputs:
                    annotations[k] = STMDSingleIn[v._params[0]]
                else:
                    annotations[k] = Port[TList[v._params[0]]]
            elif k.startswith("o_"):
                annotations[k] = Port[TList[v._params[0]]]
            elif k.startswith("a_") or k.startswith("r_"):
                if clstowrap.__dataclass_fields__[k].default:
                    attrs[k] = clstowrap.__dataclass_fields__[k].default
                annotations[k] = v

        meta, ns, kwds = prepare_class(f"STMD{clstowrap.__name__}", [cls], attrs)
        kwds["__qualname__"] = f"STMD{clstowrap.__name__}"
        kwds["_traction"] = clstowrap
        kwds["__annotations__"] = annotations

        ret = meta(kwds["__qualname__"], (cls,), kwds)
        cls._wrap_cache[kwds["__qualname__"]] = ret
        return ret

    def _prep_tractions(self, first_in, args, inputs, resources, outputs):
        """Prepare tractions for the run."""
        self.tractions = TList[Union[Traction, None]]([])
        for x in range(len(first_in)):
            uid = "%s:%d" % (self.uid, x)
            self.tractions.append(
                _init_traction(self._traction, inputs[x], resources, args, x, uid, self._observer)
            )

        if self.state == TractionState.READY:
            self.tractions_state = TList[TractionState]([])
            self.tractions_state.extend(TList[TractionState]([TractionState.READY] * len(first_in)))

            # clear outputs and set each of them to deault value
            for o in outputs:
                o_type = getattr(self, "_raw_" + o).data._params[0]
                getattr(self, "_raw_" + o).data.clear()
                for i in range(len(first_in)):
                    getattr(self, "_raw_" + o).data.append(o_type())

    def run(self) -> "STMD":
        """Run the STMD class."""
        if self.state not in (TractionState.READY, TractionState.ERROR):
            return self

        self._reset_stats()
        self.stats.started = isodate_now()

        first_in_field = None
        for f, ftype in self._fields.items():
            if f.startswith("i_"):
                # check if input is not STMDSingleIn type and if it's not None or is NullPort
                # if so, mark it as first input which we can iterate over.
                if TypeNode.from_type(ftype, subclass_check=False) != TypeNode.from_type(
                    STMDSingleIn[ANY], subclass_check=False
                ) and (
                    not isinstance(getattr(self, f), _defaultNone)
                    or TypeNode.from_type(type(getattr(self, "_raw_" + f)), subclass_check=False)
                    == TypeNode.from_type(NullPort[ANY], subclass_check=False)
                ):
                    first_in_field = f
                    break

        if not first_in_field:
            raise RuntimeError(f"[{self.uid}] Cannot have STMD with only STMDSingleIn inputs")

        LOGGER.info(
            f"Running STMD {self.fullname} on {self.a_executor} "
            f"on {len(getattr(self, first_in_field))} items"
        )

        outputs = {}
        for f in self._fields:
            if f.startswith("o_"):
                outputs[f] = getattr(self, f)

        inputs = []
        for i in range(len(getattr(self, first_in_field))):
            inputs.append({})
            for f, ftype in self._fields.items():
                if f.startswith("i_"):
                    # Raise error if allow_unset_inputs is False and input is missing
                    if not self.a_allow_unset_inputs and (
                        getattr(self, f) is None or isinstance(getattr(self, f), _defaultNone)
                    ):
                        raise ValueError(f"{self.fullname}: No input data for '{f}'")
                    elif getattr(self, f) is None or isinstance(getattr(self, f), _defaultNone):
                        continue
                    if TypeNode.from_type(
                        self._fields[f], subclass_check=False
                    ) != TypeNode.from_type(STMDSingleIn[ANY]) and len(getattr(self, f)) != len(
                        getattr(self, first_in_field)
                    ):
                        raise ValueError(
                            f"{self.__class__}: Input {f} has length"
                            f" {len(getattr(self, f))} but "
                            f"others have length {len(getattr(self, first_in_field))}"
                        )

                    if TypeNode.from_type(
                        self._fields[f], subclass_check=False
                    ) == TypeNode.from_type(STMDSingleIn[ANY]):
                        inputs[i][f] = Port.__class_getitem__(*getattr(self, "_raw_" + f)._params)(
                            data=getattr(self, f)
                        )
                    else:
                        inputs[i][f] = Port.__class_getitem__(
                            *getattr(self, "_raw_" + f)._params[0]._params
                        )(data=getattr(self, f)[i])
        args = {}
        for f, ftype in self._fields.items():
            if f.startswith("a_"):
                # do not copy stmd special args if those are not in traction
                if f in ("a_executor", "a_delete_after_finished", "a_allow_unset_inputs"):
                    if f not in self._traction._fields:
                        continue
                args[f] = getattr(self, f)

        resources = {}
        for f, ftype in self._fields.items():
            if f.startswith("r_"):
                resources[f] = getattr(self, f)

        self._prep_tractions(getattr(self, first_in_field), args, inputs, resources, outputs)
        self.state = TractionState.RUNNING

        self.a_executor.init()
        for i in range(0, len(getattr(self, first_in_field))):
            if self.tractions_state[i] in (
                TractionState.READY,
                TractionState.ERROR,
            ):
                uid = "%s:%d" % (self.uid, i)
                self.a_executor.execute(
                    i, uid, self._traction, inputs[i], args, resources, self._observer
                )

        uids = ["%s:%d" % (self.uid, i) for i in range(len(getattr(self, first_in_field)))]
        for uid, (state, stats) in self.a_executor.get_states(uids).items():
            index = uids.index(uid)
            self.tractions_state[index] = state
            self.tractions[index].state = state
            self.tractions[index].stats = stats

        for uid, out in self.a_executor.get_outputs(uids).items():
            index = uids.index(uid)
            for o in out:
                getattr(self, o)[index] = out[o]

        if any(s in (TractionState.ERROR) for s in self.tractions_state):
            self.state = TractionState.ERROR
        elif any(s in (TractionState.FAILED) for s in self.tractions_state):
            self.state = TractionState.FAILED
        else:
            self.state = TractionState.FINISHED
        self._finish_stats()
        self.stats = self.stats
        return self
