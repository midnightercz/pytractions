from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging
from typing import Optional, Dict, Any, Tuple, List, ClassVar

from .base import (
    Base,
    Field,
    TList,
    TDict,
    ANY,
    TypeNode,
    Port,
    NullPort,
    STMDSingleIn,
)
from .traction import Traction, TractionState, TractionStats, TractionMeta, TractionFailedError
from .exc import UninitiatedResource, WrongInputMappingError, WrongArgMappingError
from .utils import isodate_now


LOGGER = logging.getLogger(__name__)


class TractorMeta(TractionMeta):
    """Tractor metaclass."""

    @classmethod
    def _attribute_check(cls, attr, type_, all_attrs):
        type_type_node = TypeNode.from_type(type_, subclass_check=False)

        if attr not in (
            "uid",
            "state",
            "skip",
            "skip_reason",
            "errors",
            "stats",
            "details",
            "tractions",
        ):
            type_type_node = TypeNode.from_type(type_, subclass_check=False)
            if attr.startswith("t_"):
                if TypeNode.from_type(type_, subclass_check=True) != TypeNode.from_type(Traction):
                    raise TypeError(f"Attribute {attr} has to be type Traction, but is {type_}")
            elif attr == "d_":
                if type_type_node != TypeNode.from_type(str):
                    raise TypeError(f"Attribute {attr} has to be type str, but is {type_}")
            elif attr.startswith("d_"):
                if type_type_node != TypeNode.from_type(str):
                    raise TypeError(f"Attribute {attr} has to be type str, but is {type_}")
                if attr.replace("d_", "", 1) not in all_attrs["__annotations__"]:
                    raise TypeError(
                        f"Attribute {attr.replace('d_', '', 1)} is not defined for description "
                        f"{attr}: {all_attrs}"
                    )
            elif not (
                attr.startswith("_")
                or attr.startswith("i_")
                or attr.startswith("o_")
                or attr.startswith("a_")
                or attr.startswith("r_")
                or attr.startswith("t_")
                or attr.startswith("d_")
            ):
                raise TypeError(f"Attribute {attr} has start with i_, o_, a_, r_ or t_")

    @classmethod
    def _process_output(
        cls,
        traction,
        output_name,
        raw_output,
        output,
        outputs_map,
        all_outputs,
        traction_waves,
        output_waves,
    ):
        to_process = []
        all_outputs.append(id(output))
        outputs_map[id(raw_output)] = (traction, output_name)
        output_waves[id(raw_output)] = traction_waves[traction]

        to_process.append((output, [traction, output_name, "data"]))
        while to_process:
            current, current_mapping = to_process.pop(0)

            all_outputs.append(id(current))
            outputs_map[id(current)] = current_mapping
            output_waves[id(current)] = traction_waves[traction]
            if isinstance(current, Base):
                for f in current._fields:
                    to_process.append((getattr(current, f), current_mapping + [f]))

    @classmethod
    def _overwrite_attributes_from_base_class(cls, base, _attr, destination):
        if hasattr(base, _attr):
            for k, v in getattr(base, _attr).items():
                if k not in destination:
                    destination[k] = v

    @classmethod
    def _gather_outputs_from_base_class(cls, base, outputs_all):
        if hasattr(base, "_known_output_ids"):
            for v in base._known_output_ids:
                outputs_all.append(v)

    @classmethod
    def _validate_input_type(cls, f, fo):
        if TypeNode.from_type(type(fo)) != TypeNode.from_type(
            STMDSingleIn[ANY]
        ) and TypeNode.from_type(type(fo)) != TypeNode.from_type(type(Port[ANY])):
            raise ValueError(f"Tractor input {f} has to be type Port[ANY] but is {type(fo)}")

    @classmethod
    def _before_new(cls, name, attrs, bases):
        # mapping which holds tractor inputs + tractions outputs
        outputs_map = {}
        # inputs map to map (traction, input_name) -> (traction/tractor, output_name)
        io_map = {}
        resources = {}
        resources_map = {}
        t_outputs_map = {}
        args_map = {}
        margs_map = {}
        args = {}
        margs = {}
        output_waves = {}
        traction_waves = {}

        for dst_o, _attr in [
            (outputs_map, "_outputs_map"),
            (io_map, "_io_map"),
            (resources, "_resources"),
            (resources_map, "_resources_map"),
            (outputs_map, "_outputs_map"),
            (t_outputs_map, "_t_outputs_map"),
            (args_map, "_args_map"),
            (margs_map, "_margs_map"),
            (args, "_args"),
            (margs, "_margs"),
            (output_waves, "_output_waves"),
            (traction_waves, "_traction_waves"),
        ]:
            for base in bases:
                cls._overwrite_attributes_from_base_class(base, _attr, dst_o)

        known_output_ids = []
        for base in bases:
            cls._gather_outputs_from_base_class(base, known_output_ids)

        tractions = []
        for t in attrs["_fields"]:
            if not t.startswith("t_"):
                continue
            traction = attrs[t]
            if isinstance(traction, Field):
                traction_fields = traction.default._fields
                _traction = traction.default
            else:
                traction_fields = traction._fields
                _traction = traction
            tractions.append(_traction)

        for f, fo in attrs.items():
            if f.startswith("i_"):
                cls._validate_input_type(f, fo)

                # If owner is one of the child tractions, set it to None
                # as the input owner should be tractor
                if fo._owner in tractions:
                    fo._owner = None

                outputs_map[id(fo)] = ("#", f)
                output_waves[id(fo)] = 0
                known_output_ids.append(id(fo))
            if f.startswith("o_"):
                known_output_ids.append(id(fo))
            if f.startswith("r_"):
                resources[id(fo)] = f
            if f.startswith("a_"):
                args[id(fo)] = f
            #     if isinstance(fo, MultiArg):
            #         for maf in fo._fields:
            #             mafo = getattr(fo, maf)
            #             margs[id(mafo)] = (f, maf)
            #             if id(mafo) in args:
            #                 margs_map[("#", f, maf)] = args[id(mafo)]

        # Process tractions
        for t in attrs["_fields"]:
            if not t.startswith("t_"):
                continue
            wave = 0
            traction = attrs[t]
            # in the case of using inputs from parent
            if isinstance(traction, Field):
                traction_fields = traction.default._fields
                _traction = traction.default
            else:
                traction_fields = traction._fields
                _traction = traction
            # build io map from traction inputs
            for tf in traction_fields:
                raw_tfo = object.__getattribute__(_traction, tf)
                tfo = getattr(_traction, tf)
                if tf.startswith("i_"):
                    cls._validate_input_type(tf, raw_tfo)
                    if TypeNode.from_type(
                        type(raw_tfo), subclass_check=False
                    ) != TypeNode.from_type(NullPort[ANY]):
                        if id(raw_tfo._owner) != id(traction):
                            # if input is not default input of the tractor,
                            # check wether is mapped to known output
                            if (
                                id(raw_tfo) not in known_output_ids
                                and id(tfo) not in known_output_ids
                            ):
                                raise WrongInputMappingError(
                                    f"Input {name}.{_traction.__class__}[{_traction.uid}]->{tf} "
                                    "is mapped to output which is not known yet"
                                )
                    if id(raw_tfo) in outputs_map:
                        io_map[(t, tf)] = outputs_map[id(raw_tfo)]
                        wave = max(output_waves[id(raw_tfo)], wave)
                    elif id(tfo) in outputs_map:
                        io_map[(t, tf)] = outputs_map[id(tfo)]
                        wave = max(output_waves[id(tfo)], wave)
                    elif id(raw_tfo._owner) == id(_traction):
                        pass  # don't do anything if input is default traction input
                    else:
                        raise WrongInputMappingError(
                            f"Input {name}.{_traction.__class__}[{_traction.uid}]->{tf}"
                            " is mapped to "
                            "Port which is not tractor input or output of any traction."
                        )

            # END: build io map from traction inputs

            # Set traction wave to be one more than the highest wave of its inputs
            traction_waves[t] = wave + 1

            for tf in _traction._fields:
                tfo = getattr(_traction, tf)
                raw_tfo = object.__getattribute__(_traction, tf)

                if tf.startswith("o_"):
                    cls._process_output(
                        t,
                        tf,
                        raw_tfo,
                        tfo,
                        outputs_map,
                        known_output_ids,
                        traction_waves,
                        output_waves,
                    )
                elif tf.startswith("r_"):
                    if id(tfo) in resources:
                        resources_map[(t, tf)] = resources[id(tfo)]
                    if id(raw_tfo) in resources:
                        resources_map[(t, tf)] = resources[id(raw_tfo)]
                    else:
                        raise ValueError(f"Resources {t}.{tf} is not mapped to any parent resource")

                elif tf.startswith("a_"):
                    if id(raw_tfo) in args:
                        args_map[(t, tf)] = args[id(raw_tfo)]

                    elif id(tfo) in args:
                        args_map[(t, tf)] = args[id(tfo)]

                    elif id(raw_tfo) in margs:
                        args_map[(t, tf)] = margs[id(raw_tfo)]

                    elif id(tfo) in margs:
                        args_map[(t, tf)] = margs[id(tfo)]

                    # elif TypeNode.from_type(type(tfo), subclass_check=True) == TypeNode.from_type(
                    #     MultiArg
                    # ):
                    #     for maf, mafo in raw_tfo._fields.items():
                    #         if id(mafo) in args:
                    #             margs_map[(t, tf, maf)] = args[id(mafo)]
                    #
                    elif id(raw_tfo) in resources_map or id(raw_tfo) in outputs_map:
                        raise WrongArgMappingError(
                            f"{t}.{tf} is argument and cannot be mapped to "
                            "inputs, outputs or resources"
                        )

        for f, fo in attrs.items():
            if f.startswith("o_"):
                if id(fo) in outputs_map:
                    path = outputs_map[id(fo)]
                    if len(path) == 3 and path[2] == "data":
                        t_outputs_map[f] = path[0:2]
                    else:
                        t_outputs_map[f] = path

        attrs["_t_outputs_map"] = t_outputs_map
        attrs["_output_waves"] = output_waves
        attrs["_outputs_map"] = outputs_map
        attrs["_resources"] = resources
        attrs["_resources_map"] = resources_map
        attrs["_args_map"] = args_map
        attrs["_margs_map"] = margs_map
        attrs["_io_map"] = io_map
        attrs["_args"] = args
        attrs["_margs"] = margs
        attrs["_known_output_ids"] = known_output_ids
        attrs["_traction_waves"] = traction_waves


class Tractor(Traction, metaclass=TractorMeta):
    """Tractor class."""

    _TYPE: ClassVar[str] = "TRACTOR"
    _CUSTOM_TYPE_TO_JSON: ClassVar[bool] = True
    _CUSTOM_TO_JSON: ClassVar[bool] = True
    uid: str
    state: str = "ready"
    skip: bool = False
    skip_reason: Optional[str] = ""
    errors: TList[str] = Field(default_factory=TList[str])
    stats: TractionStats = Field(default_factory=TractionStats)
    details: TDict[str, str] = Field(default_factory=TDict[str, str])
    tractions: TDict[str, Traction] = Field(default_factory=TDict[str, Traction], init=False)
    _initialized: bool = Field(default=False, init=False, repr=False)

    def _init_traction_input(self, traction_name, traction):
        init_fields = {}
        for ft, field in traction.__dataclass_fields__.items():
            # set all inputs for the traction created at the end of this method
            # to outputs of traction copy in self.tractions
            if ft.startswith("i_"):
                if (traction_name, ft) in self._io_map:
                    source, o_name = self._io_map[(traction_name, ft)]
                    if source == "#":
                        init_fields[ft] = getattr(self, o_name)
                    else:
                        init_fields[ft] = getattr(self.tractions[source], o_name)
        return init_fields

    def _init_traction(self, traction_name, traction):
        init_fields = {}

        for ft, field in traction.__dataclass_fields__.items():
            # set all inputs for the traction created at the end of this method
            # to outputs of traction copy in self.tractions
            if ft.startswith("i_"):
                if (traction_name, ft) in self._io_map:
                    source, *o_path = self._io_map[(traction_name, ft)]
                    if source == "#":
                        out = self
                    else:
                        out = self.tractions[source]
                    if not isinstance(o_path, list):
                        o_path = [o_path]

                    if len(o_path) == 2 and o_path[1] == "data":
                        o_path = o_path[0:1]

                    if len(o_path) <= 2:
                        for o_name in o_path:
                            out = object.__getattribute__(out, o_name)
                    else:
                        for o_name in o_path:
                            out = object.__getattribute__(out, o_name)

                        n_out = Port[type(out)](data=None)
                        n_out._ref = object.__getattribute__(self.tractions[source], o_path[0])
                        n_out._data_proxy = o_path[1:]
                        out = n_out

                    init_fields[ft] = out

            elif ft.startswith("r_"):
                self_field = self._resources_map[(traction_name, ft)]
                init_fields[ft] = object.__getattribute__(self, self_field)
            elif ft.startswith("a_"):
                # First check if whole argument can be found in map
                # of global tractor arguments
                if (traction_name, ft) in self._args_map:
                    self_field = self._args_map[(traction_name, ft)]
                    if isinstance(self_field, tuple):
                        init_fields[ft] = getattr(getattr(self, self_field[0]), self_field[1])
                    else:
                        init_fields[ft] = object.__getattribute__(self, self_field)
                # # handle MultiArg type
                # elif TypeNode.from_type(field.type, subclass_check=True) == TypeNode.from_type(
                #     MultiArg
                # ):
                #     ma_init_fields = {}
                #     for maf, _ in field.type._fields.items():
                #         if (traction_name, ft, maf) in self._margs_map:
                #             self_field = self._margs_map[(traction_name, ft, maf)]
                #             ma_init_fields[maf] = getattr(self, self_field)
                #     init_fields[ft] = field.type(**ma_init_fields)
                # if argument is not found in arg mapping, use default value
                elif (traction_name, ft) not in self._args_map:
                    init_fields[ft] = getattr(traction, ft)
            elif ft == "uid":
                # change uid to be tractor.uid::traction.uid
                init_fields[ft] = "%s::%s" % (self.uid, getattr(traction, ft))
            # if field doesn't start with _ include it in init_fields to
            # initialize the traction copy
            elif field.init:
                if ft.startswith("_"):
                    continue
                init_fields[ft] = getattr(traction, ft)

        LOGGER.info("Init traction %s", traction_name)
        return traction.__class__(**init_fields)

    # def __post_init__(self):
    #    super().__post_init__()
    #    self._before_run()

    def _before_run(self):
        """Tractor setup before run."""
        self._elementary_outs = {}
        self.tractions = TDict[str, Traction]({})

        for f in self._fields:
            # Init all tractions to self.tractions
            if f.startswith("t_"):
                traction = getattr(self, f)
                new_traction = self._init_traction(f, traction)
                if hasattr(new_traction, "_before_run"):
                    new_traction._before_run()
                    new_traction._initialized = True
                self.tractions[f] = new_traction

        for f in self._fields:
            # set tractor output to outputs of copied tractions
            if f.startswith("o_"):
                # regular __setattr__ don't overwrite whole output model but just
                # data in it to keep connection, so need to use _no_validate_setattr
                t, *tf_path = self._t_outputs_map[f]
                if t == "#":
                    out = self
                else:
                    out = self.tractions[t]
                if not isinstance(tf_path, list):
                    tf_path = [tf_path]
                for o_name in tf_path:
                    out = object.__getattribute__(out, o_name)

                self._observed(f, out)
                self._no_validate_setattr_(f, out)
                self._no_validate_setattr_("_raw_" + f, out)
            # elif f.startswith("a_"):
            #     # for MulArgs which are mapped to args, overwrite them
            #     fo = getattr(self, f)
            #     # if isinstance(fo, MultiArg):
            #     #     for maf in fo._fields:
            #     #         if ("#", f, maf) in self._margs_map:
            #     #             setattr(fo, maf, getattr(self, self._margs_map[("#", f, maf)]))
            elif f.startswith("i_"):
                self._no_validate_setattr_("_raw_" + f, object.__getattribute__(self, f))

    def resubmit_from(self, traction_name: str):
        """Run tractor from specific traction."""
        reset_started = False
        self.state = TractionState.READY
        for tname, traction in self.tractions.items():
            if tname == traction_name:
                reset_started = True
            if reset_started:
                traction.state = TractionState.READY
            else:
                traction.state = TractionState.FINISHED

    def _run(self) -> "Tractor":  # pragma: no cover
        if not self._initialized:
            self._before_run()
            self._initialized = True

        # Check for uninitialized resources
        for f in self._fields:
            if f.startswith("r_"):
                fo = getattr(self, f)
                if isinstance(fo, NullPort):
                    raise UninitiatedResource(f"{f}")

        for tname, traction in self.tractions.items():
            traction.run()
            if traction.state == TractionState.ERROR:
                LOGGER.error(f"traction {traction.fullname} ERROR")
                self.state = TractionState.ERROR
                traction.stats = traction.stats
                return self
            elif traction.state == TractionState.FAILED:
                LOGGER.error(f"traction {traction.fullname} FAILED")
                self.state = TractionState.FAILED
                traction.stats = traction.stats
                return self
            else:
                LOGGER.info(f"Traction {traction.fullname} FINISHED")
        return self

    def run(self) -> "Tractor":
        """Run the tractor."""
        self._reset_stats()
        if self.state == TractionState.READY:
            self.stats.started = isodate_now()

            self.state = TractionState.PREP
            self._pre_run()
        try:
            if self.state not in (TractionState.PREP, TractionState.ERROR):
                return self
            if not self.skip:
                self.state = TractionState.RUNNING
                self._run()
        except TractionFailedError:
            self.state = TractionState.FAILED
        except Exception as e:
            self.state = TractionState.ERROR
            self.errors.append(str(e))
            raise
        finally:
            self.state = TractionState.FINISHED
            self._finish_stats()
            self.stats = self.stats
        return self

    @classmethod
    def type_to_json(cls) -> Dict[str, Any]:
        """Return tractor type to json."""
        pre_order: Dict[str, Any] = {}
        # stack is list of (current_cls_to_process, current_parent, current_key, current_default)
        stack: List[Tuple[Base, Dict[str, Any], str]] = [(cls, pre_order, "root", None)]
        while stack:
            current, current_parent, parent_key, current_default = stack.pop(0)
            if (
                hasattr(current, "_CUSTOM_TYPE_TO_JSON")
                and current._CUSTOM_TYPE_TO_JSON
                and current != cls
            ):
                current_parent[parent_key] = current.type_to_json()
            elif hasattr(current, "_fields"):
                current_parent[parent_key] = {"$type": TypeNode.from_type(current).to_json()}
                if hasattr(current, "_TYPE"):
                    current_parent[parent_key]["_TYPE"] = current._TYPE
                for f, ftype in current._fields.items():
                    if type(current.__dataclass_fields__[f].default) in (str, int, float, None):
                        stack.append(
                            (
                                ftype,
                                current_parent[parent_key],
                                f,
                                current.__dataclass_fields__[f].default,
                            )
                        )
                    else:
                        stack.append((ftype, current_parent[parent_key], f, None))
            else:
                current_parent[parent_key] = {
                    "$type": TypeNode.from_type(current).to_json(),
                    "default": current_default,
                }
                if hasattr(current, "_TYPE"):
                    current_parent[parent_key]["_TYPE"] = current._TYPE

        pre_order["root"]["_TYPE"] = cls._TYPE
        pre_order["root"]["$io_map"] = [[list(k), list(v)] for k, v in cls._io_map.items()]
        pre_order["root"]["$resource_map"] = [[list(k), v] for k, v in cls._resources_map.items()]
        return pre_order["root"]

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> "Tractor":
        """Deserialize tractor from json."""
        args = {}
        outs = {}
        tractions = {}
        traction_outputs = {}
        for f, ftype in cls._fields.items():
            if not cls.__dataclass_fields__[f].init:
                continue
            if f.startswith("i_") and isinstance(json_data[f], str):
                continue
            elif f.startswith("t_"):
                args[f] = ftype.from_json(json_data[f])
                tractions[f] = args[f]
                for tf in tractions[f]._fields:
                    if tf.startswith("o_"):
                        traction_outputs.setdefault(tractions[f].fullname, {})[tf] = getattr(
                            tractions[f], tf
                        )
                for tf, tfval in json_data[f].items():
                    if (
                        tf.startswith("i_") or tf.startswith("r_") or tf.startswith("a_")
                    ) and isinstance(tfval, str):
                        traction_name, o_name = tfval.split("#")
                        setattr(tractions[f], tf, traction_outputs[traction_name][o_name])
            elif f.startswith("i_"):
                if json_data[f].get("$data"):
                    args[f] = ftype.from_json(json_data[f])
            elif (
                f.startswith("a_")
                or f.startswith("r_")
                or f in ("errors", "stats", "details", "tractions")
            ):
                # skip if there are no data to load
                if json_data[f].get("$data"):
                    args[f] = ftype.from_json(json_data[f])
            elif f.startswith("o_"):
                outs[f] = ftype.from_json(json_data[f])
            elif f == "tractions":
                continue
            else:
                args[f] = json_data[f]
        ret = cls(**args)
        for o, oval in outs.items():
            getattr(ret, o).data = oval.data
        return ret


class MultiTractor(Tractor, metaclass=TractorMeta):
    """Multitractor version of tractor."""

    _TYPE: ClassVar[str] = "MULTITRACTOR"
    uid: str
    a_pool_size: Port[int] = Port[int](data=5)
    state: str = "ready"
    skip: bool = False
    skip_reason: Optional[str] = ""
    errors: TList[str] = Field(default_factory=TList[str])
    stats: TractionStats = Field(default_factory=TractionStats)
    details: TList[str] = Field(default_factory=TList[str])
    tractions: TList[Traction] = Field(default_factory=TList[Traction], init=False)
    a_use_processes: Port[bool] = Port[bool](data=False)

    def _traction_runner(self, t_name, traction, on_update=None):
        traction = self._init_traction(t_name, traction)
        LOGGER.info(f"Running traction {traction.full_name}")
        traction.run(on_update=on_update)
        return traction

    def _run(self):  # pragma: no cover
        traction_groups: Dict[int, Dict[str, Traction]] = {}
        for t in self._fields:
            if not t.startswith("t_"):
                continue
            traction_groups.setdefault(self._traction_waves[t], {})[t] = self._tractions[t]

        if self.a_use_processes:
            executor_class = ProcessPoolExecutor
        else:
            executor_class = ThreadPoolExecutor

        with executor_class(max_workers=self.a_pool_size.a) as executor:
            for w, tractions in traction_groups.items():
                ft_results = {}
                for t_name, traction in tractions.items():
                    res = executor.submit(self._traction_runner, t_name, traction)
                    ft_results[res] = t_name
                for ft in as_completed(ft_results):
                    t_name = ft_results[ft]
                    nt = ft.result()
                    self._tractions[t_name] = nt

        for f in self._fields:
            if f.startswith("o_"):
                # regular __setattr__ don't overwrite whole output model but just
                # data in it to keep connection, so need to use _no_validate_setattr
                t, tf = self._outputs_map[f]
                self._no_validate_setattr_(f, getattr(self._tractions[t], tf))
                self._no_validate_setattr_("_raw_" + f, getattr(self._tractions[t], tf))
        return self


class LoopTractorEnd(Exception):
    """Exception raised when loop tractor should end."""

    pass


class LoopTractor(Tractor):
    """Loop tractor class.

    Tractor runs all traction in loop until LoopTractorEnd exception is raised.
    """

    def _run(self) -> "LoopTractor":  # pragma: no cover
        # Check for uninitialized resources
        for f in self._fields:
            if f.startswith("r_"):
                fo = getattr(self, f)
                if isinstance(fo, NullPort):
                    raise UninitiatedResource(f"{f}")

        while True:
            try:
                for tname, traction in self.tractions.items():
                    traction.state = TractionState.READY
                    traction.run()
                    if traction.state == TractionState.ERROR:
                        LOGGER.error(f"Traction {traction.fullname} ERROR")
                        self.state = TractionState.ERROR
                        return self
                    elif traction.state == TractionState.FAILED:
                        LOGGER.error(f"Traction {traction.fullname} FAILED")
                        self.state = TractionState.FAILED
                        return self
                    else:
                        LOGGER.info(f"Traction {traction.fullname} FINISHED")
            except LoopTractorEnd:
                break
        return self
