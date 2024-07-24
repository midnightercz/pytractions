import dataclasses
import inspect
import logging
from typing import Type, Optional, Union, Any, Dict, ClassVar
from types import prepare_class

from .base import (
    Base,
    ANY,
    Res,
    Arg,
    In,
    Out,
    NoData,
    DefaultOut,
    STMDSingleIn,
    TractionState,
    TractionMeta,
    TList,
    TDict,
    Traction,
    TractionStats,
    on_update_empty,
    OnUpdateCallable,
)
from .executor import ProcessPoolExecutor, ThreadPoolExecutor, LoopExecutor
from .types import TypeNode
from .utils import isodate_now  # noqa: F401


LOGGER = logging.getLogger(__name__)


class STMDMeta(TractionMeta):
    """STMD metaclass."""

    _SELF_ARGS = ["a_executor", "a_delete_after_finished"]

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
            if attr.startswith("i_"):
                child_input_type = all_attrs["_traction"]._fields[attr]._params[0]
                if type_type_node != TypeNode.from_type(
                    In[TList[child_input_type]]
                ) and TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(
                    STMDSingleIn[child_input_type]
                ):
                    raise TypeError(
                        f"Attribute {attr} has to be type In[TList[ANY]] or STMDSingleIn[ANY], "
                        f"but is {type_}"
                    )
            elif attr.startswith("o_"):
                child_input_type = all_attrs["_traction"]._fields[attr]._params[0]
                if type_type_node != TypeNode.from_type(Out[TList[child_input_type]]):
                    raise TypeError(
                        f"Attribute {attr} has to be type Out[TList[ANY]], but is {type_}"
                    )
            elif attr.startswith("a_"):
                if type_type_node != TypeNode.from_type(Arg[ANY]):
                    raise TypeError(f"Attribute {attr} has to be type Arg[ANY], but is {type_}")
            elif attr.startswith("r_"):
                if type_type_node != TypeNode.from_type(Res[ANY]):
                    raise TypeError(f"Attribute {attr} has to be type Res[ANY], but is {type_}")
            elif attr == "d_":
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
            else:
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
            raise ValueError("Missing _traction: Type[<Traction>] = <Traction> definition")

        # record fields to private attribute
        attrs["_attrs"] = attrs
        attrs["_fields"] = {
            k: v for k, v in attrs.get("__annotations__", {}).items() if not k.startswith("_")
        }

        for f, ftype in attrs["_fields"].items():
            # Do not include outputs in init
            if f.startswith("o_") and f not in attrs:
                if inspect.isclass(ftype._params[0]) and issubclass(ftype._params[0], Base):
                    for ff, ft in ftype._params[0]._fields.items():
                        df = ftype._params[0].__dataclass_fields__[f]
                        if (
                            df.default is dataclasses.MISSING
                            and df.default_factory is dataclasses.MISSING
                        ):
                            raise TypeError(
                                f"Cannot use {ftype._params[0]} for output, as it "
                                f"doesn't have default value for field {ff}"
                            )
                attrs[f] = dataclasses.field(
                    init=False,
                    default_factory=DefaultOut(type_=ftype._params[0], params=(ftype._params)),
                )

            # Set all inputs to NoData after as default
            if f.startswith("i_") and f not in attrs:
                attrs[f] = dataclasses.field(default_factory=NoData[ftype._params])

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


_loop_executor = LoopExecutor()


class STMD(Traction, metaclass=STMDMeta):
    """STMD class."""

    _TYPE: str = "STMD"
    uid: str
    state: str = TractionState.READY
    skip: bool = False
    skip_reason: Optional[str] = ""
    errors: TList[str] = dataclasses.field(default_factory=TList[str])
    stats: TractionStats = dataclasses.field(default_factory=TractionStats)
    details: TDict[str, str] = dataclasses.field(default_factory=TDict[str, str])
    _traction: Type[Traction] = Traction
    a_delete_after_finished: Arg[bool] = Arg[bool](a=True)
    a_executor: Arg[Union[ProcessPoolExecutor, ThreadPoolExecutor, LoopExecutor]] = Arg[
        Union[ProcessPoolExecutor, ThreadPoolExecutor, LoopExecutor]
    ](a=_loop_executor)
    tractions: TList[Union[Traction, None]] = dataclasses.field(
        default_factory=TList[Optional[Traction]]
    )
    tractions_state: TList[TractionState] = dataclasses.field(default_factory=TList[TractionState])

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
                    annotations[k] = In[TList[v._params[0]]]
            if k.startswith("o_"):
                annotations[k] = Out[TList[v._params[0]]]
            if k.startswith("a_") or k.startswith("r_"):
                annotations[k] = v

        meta, ns, kwds = prepare_class(f"STMD{clstowrap.__name__}", [cls], attrs)
        kwds["__qualname__"] = f"STMD{clstowrap.__name__}"
        kwds["_traction"] = clstowrap
        kwds["__annotations__"] = annotations

        ret = meta(kwds["__qualname__"], (cls,), kwds)
        cls._wrap_cache[kwds["__qualname__"]] = ret
        return ret

    def _prep_tractions(self, first_in, outputs):
        """Prepare tractions for the run."""
        if self.state == TractionState.READY:
            self.tractions_state.clear()
            self.tractions_state.extend(
                TList[TractionState]([TractionState.READY] * len(first_in.data))
            )

            for o in outputs:
                o_type = getattr(self, o).data._params[0]
                for i in range(len(first_in.data)):
                    getattr(self, o).data.append(o_type())

    def run(
        self,
        on_update: Optional[OnUpdateCallable] = None,
    ) -> "STMD":
        """Run the STMD class."""
        _on_update: OnUpdateCallable = on_update or on_update_empty

        if self.state not in (TractionState.READY, TractionState.ERROR):
            return self

        LOGGER.info(f"Running STMD {self.fullname}")
        self._reset_stats()
        self.stats.started = isodate_now()

        first_in_field = None
        for f, ftype in self._fields.items():
            if f.startswith("i_"):
                if TypeNode.from_type(ftype, subclass_check=False) != TypeNode.from_type(
                    STMDSingleIn[ANY]
                ):
                    first_in_field = f
                    break

        if not first_in_field:
            raise RuntimeError("Cannot have STMD with only STMDSingleIn inputs")

        outputs = {}
        for f in self._fields:
            if f.startswith("o_"):
                outputs[f] = getattr(self, f)

        inputs = []
        for i in range(len(getattr(self, first_in_field).data)):
            inputs.append({})
            for f, ftype in self._fields.items():
                if f.startswith("i_"):
                    if getattr(self, f).data is None:
                        raise ValueError(f"{self.fullname}: No input data for '{f}'")
                    if TypeNode.from_type(
                        self._fields[f], subclass_check=False
                    ) != TypeNode.from_type(STMDSingleIn[ANY]) and len(
                        getattr(self, f).data
                    ) != len(
                        getattr(self, first_in_field).data
                    ):
                        raise ValueError(
                            f"{self.__class__}: Input {f} has length"
                            f" {len(getattr(self, f).data)} but "
                            f"others have length {len(getattr(self, first_in_field).data)}"
                        )

                    if TypeNode.from_type(
                        self._fields[f], subclass_check=False
                    ) == TypeNode.from_type(STMDSingleIn[ANY]):
                        inputs[i][f] = In.__class_getitem__(*getattr(self, f)._params)(
                            data=getattr(self, f).data
                        )
                    else:
                        inputs[i][f] = In.__class_getitem__(*getattr(self, f)._params[0]._params)(
                            data=getattr(self, f).data[i]
                        )
        args = {}
        for f, ftype in self._fields.items():
            if f.startswith("a_"):
                # do not copy stmd special args if those are not in traction
                if f in ("a_executor", "a_delete_after_finished"):
                    if f not in self._traction._fields:
                        continue
                args[f] = getattr(self, f)

        resources = {}
        for f, ftype in self._fields.items():
            if f.startswith("r_"):
                resources[f] = getattr(self, f)

        self._prep_tractions(getattr(self, first_in_field), outputs)
        self.state = TractionState.RUNNING
        _on_update(self)

        self.a_executor.a.init()
        for i in range(0, len(getattr(self, first_in_field).data)):
            if self.tractions_state[i] in (
                TractionState.READY,
                TractionState.ERROR,
            ):
                uid = "%s:%d" % (self.uid, i)
                self.a_executor.a.execute(
                    uid, self._traction, inputs[i], args, resources, on_update=_on_update
                )

        uids = ["%s:%d" % (self.uid, i) for i in range(len(getattr(self, first_in_field).data))]
        for uid, out in self.a_executor.a.get_outputs(uids).items():
            index = uids.index(uid)
            for o in out:
                getattr(self, o).data[index] = out[o]
        self.a_executor.a.shutdown()

        self.state = TractionState.FINISHED
        _on_update(self)
        self._finish_stats()
        return self
