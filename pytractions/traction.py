import abc
import dataclasses
import enum
import logging
import inspect

from typing import (
    Dict,
    Union,
    Optional,
    ForwardRef,
    ClassVar,
    Callable,
    Any,
)

from .base import (
    Base,
    BaseMeta,
    Field,
    Port,
    STMDSingleIn,
    DefaultOut,
    TList,
    TDict,
    NullPort,
    is_wrapped,
    extract_from_optional,
    _defaultBool,
    type_to_default_type,
    Tree,
    UnionItemHandlerSchema,
    ListItemHandlerSchema,
    DictItemHandlerSchema,
    EnumItemHandlerSchema,
    BasicItemHandlerSchema,
    PortItemHandlerSchema,
    BaseItemHandlerSchema,
    ItemHandler,
)
from .types import TypeNode
from .utils import ANY, isodate_now
from .exc import NoDefaultError, TractionFailedError

_Traction = ForwardRef("Traction")


class TractionMeta(BaseMeta):
    """Traction metaclass."""

    @classmethod
    def _attribute_check(cls, attr, type_, all_attrs):
        if attr not in (
            "uid",
            "state",
            "skip",
            "skip_reason",
            "errors",
            "stats",
            "details",
        ):
            """Check attribute on class creation."""
            type_type_node = TypeNode.from_type(type_, subclass_check=False)
            if attr == "d_":
                if type_type_node != TypeNode.from_type(str):
                    raise TypeError(f"Attribute {attr} has to be type str, but is {type_}")
            elif attr.startswith("d_"):
                if type_type_node != TypeNode.from_type(str):
                    raise TypeError(f"Attribute {attr} has to be type str, but is {type_}")
                if attr != "d_" and attr.replace("d_", "", 1) not in all_attrs["__annotations__"]:
                    raise TypeError(
                        f"Attribute {attr.replace('d_', '', 1)} is not defined for description "
                        f"{attr}: {all_attrs}"
                    )
            elif not (
                attr.startswith("i_")
                or attr.startswith("o_")
                or attr.startswith("a_")
                or attr.startswith("r_")
            ):
                raise TypeError(f"Attribute {attr} has start with i_, o_, a_, r_ or d_")

            if (
                not attr.startswith("i_")
                and not attr.startswith("a_")
                and not attr.startswith("r_")
                and not attr.startswith("o_")
                and not attr.startswith("d_")
            ):
                if not isinstance(type_, Field):
                    raise TypeError(f"Attribute {attr} has start with i_, o_, a_, r_ or d_")

    def __new__(cls, name, bases, attrs):
        """Create new traction class."""
        annotations = attrs.get("__annotations__", {})
        # check if all attrs are in supported types
        for attr, type_ in annotations.items():
            # skip private attributes
            if attr.startswith("_"):
                continue
            cls._attribute_check(attr, type_, attrs)

        attrs = cls._attributes_preparation(name, attrs, bases)

        # record fields to private attribute
        attrs["_attrs"] = attrs
        attrs["_fields"] = {
            k: v for k, v in attrs.get("__annotations__", {}).items() if not k.startswith("_")
        }

        for f, ftype in list(attrs["_fields"].items()):
            # Do not include outputs in init
            if f.startswith("a_") or f.startswith("r_"):
                if TypeNode.from_type(ftype, subclass_check=False) != TypeNode.from_type(Port[ANY]):
                    attrs["_fields"][f] = Port[ftype]

            if f.startswith("i_"):
                if TypeNode.from_type(ftype) != TypeNode.from_type(
                    STMDSingleIn[ANY]
                ) and TypeNode.from_type(ftype) != TypeNode.from_type(Port[ANY]):
                    attrs["_fields"][f] = Port[ftype]
                if f.startswith("i_") and f not in attrs:
                    attrs[f] = Field(default_factory=NullPort[attrs["_fields"][f]._params])

            if f.startswith("o_"):
                if TypeNode.from_type(ftype, subclass_check=False) != TypeNode.from_type(Port[ANY]):
                    attrs["_fields"][f] = Port[ftype]

                ftype_final = ftype._params[0] if is_wrapped(ftype) else ftype

                if inspect.isclass(ftype_final) and issubclass(ftype_final, Base):
                    stack = [(ftype_final, str(ftype_final))]
                    while stack:
                        ftype, path = stack.pop(0)
                        for ff, fft in ftype._fields.items():
                            fft = extract_from_optional(fft)
                            df = ftype.__dataclass_fields__[ff]
                            if (
                                df.default is dataclasses.MISSING
                                and df.default_factory is dataclasses.MISSING
                            ):
                                raise NoDefaultError(
                                    f"Cannot use {path} for output, as it "
                                    f"doesn't have default value for field '{ff}'"
                                )
                            if issubclass(fft, Base):
                                stack.append((fft, path + "." + ff))

                if f not in attrs:
                    attrs[f] = Field(
                        init=False,
                        default_factory=DefaultOut(
                            type_=ftype_final, params=(attrs["_fields"][f]._params)
                        ),
                    )

        ret = super().__new__(cls, name, bases, attrs)

        return ret


class TractionStats(Base):
    """Model class for traction stats."""

    started: str = ""
    finished: str = ""
    skipped: bool = False


class TractionState(str, enum.Enum):
    """Enum-like class to store step state."""

    READY = "ready"
    PREP = "prep"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"
    ERROR = "error"


OnUpdateCallable = Callable[[_Traction], None]
OnErrorCallable = Callable[[_Traction], None]


class TractionItemHandlerSchema(ItemHandler):
    """Serialization handler for base subclasses."""

    def match(self, item):
        """Match Base subclasses."""
        try:
            if issubclass(item.data_type, Traction):
                return True
        except Exception:
            import sys

            print(f"Error matching item {item.data_type} in {item.path}", file=sys.stderr)
            raise

    def process(self, tree, item):
        """Process Base subclasses."""
        item.result[item.parent_index] = {
            "type": "object",
            "title": str(item.data),
            "properties": {},
        }
        required = []
        if item.data_type._fields.get("d_"):
            item.result[item.parent_index]["description"] = item.data_type.__dataclass_fields__.get(
                "d_", {}
            ).default

        for f, ftype in item.data_type._fields.items():
            if not (f.startswith("i_") or f.startswith("a_") or f.startswith("r_")):
                continue

            if (
                item.data_type.__dataclass_fields__.get(f)
                and item.data.__dataclass_fields__[f].init is False
            ):
                # Skip if field is not initialized
                continue
            default = item.data_type.__dataclass_fields__.get(f, {}).default
            extra = {}
            if not isinstance(default, dataclasses._MISSING_TYPE):
                if not isinstance(default, Port):
                    if hasattr(default, "content_to_json"):
                        extra["default"] = default.content_to_json()
                    else:
                        extra["default"] = default
                else:
                    extra["default"] = default
            if "d_" + f in item.data_type._fields:
                extra["description"] = item.data_type.__dataclass_fields__.get("d_" + f, {}).default
            _f = item.data._SERIALIZE_REPLACE_FIELDS.get(f, f)
            tree.add_to_process(
                data=f,
                data_type=ftype,
                parent_index=_f,
                result=item.result[item.parent_index]["properties"],
                extra=extra,
            )
            required.append(f)
        item.result[item.parent_index]["required"] = required


class ToJsonSchemaTree(Tree):
    """Tree for serialization."""

    handlers = [
        UnionItemHandlerSchema(),
        ListItemHandlerSchema(),
        DictItemHandlerSchema(),
        EnumItemHandlerSchema(),
        BasicItemHandlerSchema(),
        PortItemHandlerSchema(),
        TractionItemHandlerSchema(),
        BaseItemHandlerSchema(),
    ]


class Traction(Base, metaclass=TractionMeta):
    """Class represeting basic processing element.

    Traction works with data provided on defined inputs, using provided resources and arguments and
    store output data to defined outputs.

    Traction subclasses can have defined only 5 type of user attributes:
    inputs
        every input name needs to start with ``i_``
    outputs
        every output name needs to start with ``o_``
    resources
        every resource name needs to start with ``r_``
    arguments
        every argument name needs to start with ``a_``
    documentation
        every documentation argument needs to start with ``d_``, also rest of the
        name must be already defined field. For example `i_in1` can be described in
       ``d_i_in1``. With only ``d_`` is used as the field name, it should be used as
       description of whole traction.

    example of Traction subclass

    .. code-block::

        class AVG(Traction):
            a_len: Arg[int]
            a_timeframe: Arg[str]
            r_ticker_client: Res[TickerClient]
            o_avg: Out[float]

            d_a_len: str = "Size of the window for calculation."
            d_a_timeframe: str = "Timeframe which used for calculation"
            d_r_ticker_client: str = "Ticker client which provides market data"
            d_o_avg: str = "Average value of fetched candles for selected timeframe and window"
            d_: str = "Traction used to fetch last spx500 candles and calculates average
                       of their close values"

            def run(self, on_update: Optional[OnUpdateCallable]=None):
                ret = self.r_ticker_client.r.fetch_spx_data(self.a_timeframe.a)
                closes = [x['close'] for x in ret[:self.a_len.a]]
                self.o_avg.data = sum(closes)/self.a_len.a

        tc = TickerClient(...)
        avg = AVG(uid='spx-avg',
                  a_len=Arg[int](a=10),
                  a_timeframe=Arg[str](a='1H'),
                  r_ticker_client=Res[TickerClient](r=tc)
        )
        avg.run()
        print(avg.o_avg.data)

    In the following example, output is set to Out member data. However it's also
    possible to set output like this:

    .. code-block::

        self.o_avg = Out[float](data=1.0)

    Traction class will internally set only data of the output, reference to the output
    itself will not be overwritten
    """

    _TYPE: ClassVar[str] = "TRACTION"
    _CUSTOM_TYPE_TO_JSON: ClassVar[bool] = False
    _CUSTOM_TO_JSON: ClassVar[bool] = True

    uid: str
    "Unique identifier of the current traction."
    state: TractionState = TractionState.READY
    "Indicator of current state of the traction."
    skip: bool = False
    "Flag indicating if execution of the traction was skipped."
    skip_reason: Optional[str] = ""
    "Can be se to explain why the execution of the traction was skipped."
    errors: TList[str] = Field(default_factory=TList[str])
    """List of errors which occured during the traction execution. Inherited class should add errors
    here manually"""
    stats: TractionStats = Field(default_factory=TractionStats)
    "Collection of traction stats"
    details: TDict[str, str] = Field(default_factory=TDict[str, str])
    "List of details of the execution of the Traction."
    "Inherited class can add details here manually"

    _elementary_outs: Dict[str, Any] = Field(default_factory=dict)

    @property
    def log(self):
        """Return logger for the traction."""
        if not hasattr(self, "_log") or not self._log:
            self._log = logging.getLogger(self.uid)
        return self._log

    def __post_init__(self):
        """Adjust class instance after initialization."""
        for f in self._fields:
            if f.startswith("a_") or f.startswith("r_"):
                self._no_validate_setattr_("_raw_" + f, super().__getattribute__(f))
                if TypeNode.from_type(
                    super().__getattribute__(f).__class__, subclass_check=True
                ) == TypeNode.from_type(NullPort[ANY]):
                    continue
                if not super().__getattribute__(f)._owner:
                    super().__getattribute__(f)._name = f
                    super().__getattribute__(f)._owner = self

            elif f.startswith("o_") or f.startswith("i_"):
                self._no_validate_setattr_("_raw_" + f, super().__getattribute__(f))
                if TypeNode.from_type(
                    super().__getattribute__(f).__class__, subclass_check=True
                ) == TypeNode.from_type(NullPort[ANY]):
                    continue
                if not super().__getattribute__(f)._owner:
                    super().__getattribute__(f)._name = f
                    super().__getattribute__(f)._owner = self

    def __getattribute_orig__(self, name: str) -> Any:
        """Get attribute of the class instance - unmodified version."""
        return super().__getattribute__(name)

    def __getattribute__(self, name: str) -> Any:
        """Get attribute of the class instance.

        with special handler for inputs.
        """
        if name.startswith("a_"):
            default_convertor = type_to_default_type.get(type(super().__getattribute__(name).data))
            return (
                default_convertor(super().__getattribute__(name).data)
                if default_convertor
                else super().__getattribute__(name).data
            )
        if name.startswith("r_"):
            default_convertor = type_to_default_type.get(type(super().__getattribute__(name).data))
            return (
                default_convertor(super().__getattribute__(name).data)
                if default_convertor
                else super().__getattribute__(name).data
            )
        if name.startswith("o_"):
            default_convertor = type_to_default_type.get(type(super().__getattribute__(name).data))
            if default_convertor:
                if name not in self._elementary_outs:
                    ret = default_convertor(super().__getattribute__(name).data)
                    self._elementary_outs[name] = ret
                else:
                    self._elementary_outs[name]._val = super().__getattribute__(name).data
                return self._elementary_outs[name]
            else:
                return super().__getattribute__(name).data

        if name.startswith("i_"):
            default_convertor = type_to_default_type.get(type(super().__getattribute__(name).data))
            if name not in self._fields:
                _class = super().__getattribute__("__class__")
                raise AttributeError(f"{_class} doesn't have attribute {name}")

            return (
                default_convertor(super().__getattribute__(name).data)
                if default_convertor
                else super().__getattribute__(name).data
            )

        return super().__getattribute__(name)

    def _extra_observed_events(self, attr, value, extra):
        # print("PATH", attr)
        if attr in ("state", "stats", "uid"):
            extra["traction_extra"] = {}
            if hasattr(self, "uid"):
                extra["traction_extra"]["traction"] = self.fullname
            if hasattr(self, "uid"):
                extra["traction_extra"]["uid"] = self.uid
            if hasattr(self, "state"):
                extra["traction_extra"]["state"] = self.state
            if hasattr(self, "stats"):
                extra["traction_extra"]["stats"] = self.stats.content_to_json()
        return extra

    def _validate_setattr_(self, name: str, value: Any) -> None:
        """Set attribute to the instance with type validation and inputs handling."""
        if not name.startswith("_"):  # do not check for private attrs
            if name not in self._fields and not self._config.allow_extra:
                raise AttributeError(f"{self.__class__} doesn't have attribute {name}")

        wrapped = True
        if (
            name.startswith("i_")
            or name.startswith("o_")
            or name.startswith("a_")
            or name.startswith("r_")
        ):
            default_attr = self.__dataclass_fields__[name].default
            vtype = value.__class__
            tt1 = TypeNode.from_type(
                vtype, subclass_check=True, type_aliases=[(type(True), _defaultBool)]
            )
            if is_wrapped(vtype):
                tt2 = TypeNode.from_type(
                    self._fields[name], type_aliases=[(type(True), _defaultBool)]
                )
            else:
                tt2 = TypeNode.from_type(
                    self._fields[name]._params[0], type_aliases=[(type(True), _defaultBool)]
                )
                # Value is not wrapped in Arg, In, Out or Res
                wrapped = False
            if tt1 != tt2:
                raise TypeError(
                    f"Cannot set attribute {self.__class__}.{name} to type {vtype}, "
                    f"expected  {tt2.to_type()}"
                )

        if name.startswith("i_"):
            # Need to check with hasattr first to make sure inputs can be initialized
            # print("SETTR", name)
            if hasattr(self, name):
                # Allow overwrite default input values
                if super().__getattribute__(name) == default_attr or TypeNode.from_type(
                    super().__getattribute__(name)
                ) == TypeNode.from_type(NullPort[ANY]):
                    if wrapped:
                        self._update_observers(name, value)
                        # print("I_ TRACTION OBSERVED WRAPPED", name)
                        self._observed(name, value)
                        self._no_validate_setattr_(name, value)
                        self._no_validate_setattr_("_raw_" + name, value)
                    else:
                        wrapped_val = self._fields[name](data=value)
                        self._update_observers(name, wrapped_val)
                        # print("I_ TRACTION OBSERVED UNWRAPPED", name)
                        self._observed(name, wrapped_val)
                        self._no_validate_setattr_(name, wrapped_val)
                        self._no_validate_setattr_("_raw_" + name, wrapped_val)
                    return
                connected = (
                    TypeNode.from_type(type(getattr(self, "_raw_" + name)), subclass_check=False)
                    != TypeNode.from_type(NullPort[ANY])
                    and TypeNode.from_type(
                        type(getattr(self, "_raw_" + name)), subclass_check=False
                    )
                    != Port[ANY]
                    and TypeNode.from_type(
                        type(getattr(self, "_raw_" + name)), subclass_check=False
                    )
                    != TypeNode.from_type(NullPort[ANY])
                )
                if connected:
                    raise AttributeError(f"Input {name} is already connected")

            # in the case input is not set, initialize it
            elif not hasattr(self, name):
                if wrapped:
                    self._no_validate_setattr_(name, value)
                    self._no_validate_setattr_("_raw_" + name, value)
                    if not object.__getattribute__(value, "_ref"):
                        super().__getattribute__(name)._ref = value
                else:
                    self._no_validate_setattr_(name, self._fields[name](data=value))
            return

        elif name.startswith("o_"):
            # print("SETTR", name)
            if not hasattr(self, name):
                # output is set for the first time
                if wrapped:
                    self._no_validate_setattr_(name, self._fields[name](data=value.data))
                else:
                    self._no_validate_setattr_(name, self._fields[name](data=value))

            if not self.__getattribute_orig__(name)._owner:
                self.__getattribute_orig__(name)._owner = self
                self.__getattribute_orig__(name)._name = name
            # Do not overwrite whole output container, rather just copy update data
            if wrapped:
                # print("O_ TRACTION OBSERVED WRAPPED", name)
                self._observed(name, value)
                self.__getattribute_orig__(name).data = value.data
                self._update_observers(name, self.__getattribute_orig__(name))
                # print("O_ TRACTION OBSERVED WRAPPED END", name)
            else:
                # print("O_ TRACTION OBSERVED UNWRAPPED", name)
                self._observed(name, value)
                self.__getattribute_orig__(name).data = value
                self._update_observers(name, self.__getattribute_orig__(name))
                # print("O_ TRACTION OBSERVED UNWRAPPED END", name)
            return
        elif name.startswith("a_"):
            if hasattr(self, name):
                # Allow overwrite default input values
                if super().__getattribute__(name) == default_attr or TypeNode.from_type(
                    super().__getattribute__(name)
                ) == TypeNode.from_type(NullPort[ANY]):
                    if wrapped:
                        self._no_validate_setattr_(name, value)
                        self._no_validate_setattr_("_raw_" + name, value)
                    else:
                        wrapped_val = self._fields[name](data=value)
                        self._no_validate_setattr_(name, wrapped_val)
                        self._no_validate_setattr_("_raw_" + name, wrapped_val)
                    return
            # in the case input is not set, initialize it
            elif not hasattr(self, name):
                if wrapped:
                    self._no_validate_setattr_(name, value)
                    self._no_validate_setattr_("_raw_" + name, value)
                else:
                    self._no_validate_setattr_(name, self._fields[name](data=value))
            return
        elif name.startswith("r_"):
            if not hasattr(self, name):
                if wrapped:
                    self._no_validate_setattr_(name, value)
                else:
                    self._no_validate_setattr_(name, self._fields[name](data=value))
            else:
                if super().__getattribute__(name) == default_attr:
                    if wrapped:
                        self._no_validate_setattr_(name, value)
                    else:
                        self._no_validate_setattr_(name, self._fields[name](data=value))
                else:
                    if wrapped:
                        if TypeNode.from_type(vtype, subclass_check=True) == TypeNode.from_type(
                            NullPort[ANY]
                        ):
                            self._no_validate_setattr_(name, value)
                        else:
                            self.__getattribute_orig__(name).data = value.data
                    else:
                        self.__getattribute_orig__(name).data = value
            self._observed(name, value)
            return

        super().__setattr__(name, value)

    def add_details(self, detail):
        """Add details about traction run."""
        self.details[isodate_now()] = detail

    @property
    def fullname(self) -> str:
        """Full name of traction instance. It's composition of class name and instance uid."""
        return f"{self.__class__.__name__}[{self.uid}]"

    def to_json(self) -> Dict[str, Any]:
        """Serialize class instance to json representation."""
        ret = {"$data": {}}
        for f in self._fields:
            if f.startswith("i_") or f.startswith("a_") or f.startswith("r_"):
                if (
                    hasattr(getattr(self, "_raw_" + f), "_owner")
                    and getattr(self, "_raw_" + f)._owner
                    and getattr(self, "_raw_" + f)._owner != self
                ):
                    ret["$data"][f] = (
                        getattr(self, "_raw_" + f)._owner.fullname
                        + "#"
                        + getattr(self, "_raw_" + f)._name
                    )
                else:
                    i_json = getattr(self, "_raw_" + f).to_json()
                    ret["$data"][f] = i_json
            elif f.startswith("o_"):
                ret["$data"][f] = object.__getattribute__(self, f).to_json()
            elif isinstance(getattr(self, f), (enum.Enum)):
                ret["$data"][f] = getattr(self, f).value
            elif isinstance(getattr(self, f), (int, str, bool, float, type(None))):
                ret["$data"][f] = getattr(self, f)
            else:
                ret["$data"][f] = getattr(self, f).to_json()

        ret["$type"] = TypeNode.from_type(self.__class__).to_json()
        # ret['$data']["name"] = self.__class__.__name__
        # ret['$data']["type"] = self._TYPE
        return ret

    def _getstate_to_json(self) -> Dict[str, Any]:
        ret = {}
        for f in self._fields:
            if isinstance(getattr(self, f), (int, str, bool, float, type(None))):
                ret[f] = getattr(self, f)
            else:
                ret[f] = getattr(self, f).to_json()
        return ret

    def run(
        self,
    ) -> "Traction":
        """Start execution of the Traction.

        * When traction is in `TractionState.READY` it runs the
        user defined _pre_run method where user can do some
        preparation before the run itself, potentially set `skip`
        attribute to True to skip the execution. After that, traction
        state is set to TractionState.PREP

        * When traction is in TractionState.PREP or TractionState.ERROR, if skip is set to True
          skipped attribute is set to True, and execution is finished.

        * When skip is not set to True, state is set to TractionState.RUNNING
          and user defined _run method is executed.
        If an exception is raised during the execution:
          * If exception is TractionFailedError, state is set to FAILED. This means
            traction failed with defined failure and it's not possible to rerun it

          * If unexpected exception is raised, traction state is set to ERROR which is
            state from which it's possible to rerun the traction.

         At the end of the execution traction stats are updated.
        """
        self._reset_stats()
        if self.state == TractionState.READY:
            self.stats.started = isodate_now()

            self.state = TractionState.PREP
            self.log.debug(f"Starting traction {self.fullname} pre_run")
            self._pre_run()
        try:
            if self.state not in (TractionState.PREP, TractionState.ERROR):
                self.log.debug(f"Skipping traction {self.fullname} as state is {self.state}")
                return self
            if not self.skip:
                self.state = TractionState.RUNNING
                self.log.debug(f"Running traction {self.fullname}")
                self._run()
        except TractionFailedError:
            self.state = TractionState.FAILED
        except Exception as e:
            self.state = TractionState.ERROR
            self.errors.append(str(e))
            raise
        else:
            self.state = TractionState.FINISHED
        finally:
            self.log.debug(f"Traction {self.fullname} finished")
            self._finish_stats()
            self.stats = self.stats
        return self

    def _pre_run(self) -> None:
        """Execute code needed before step run.

        In this method, all neccesary preparation of data can be done.
        It can be also used to determine if step should run or not by setting
        self.skip to True and providing self.skip_reason string with explanation.
        """
        pass

    def _reset_stats(self) -> None:
        """Reset stats of the traction."""
        self.stats = TractionStats(
            started="",
            finished="",
            skipped=False,
        )

    def _finish_stats(self) -> None:
        self.stats.finished = isodate_now()
        self.stats.skipped = self.skip

    @abc.abstractmethod
    def _run(self, on_update: Optional[OnUpdateCallable] = None) -> None:  # pragma: no cover
        """Run code of the step.

        Method expects raise StepFailedError if step code fails due data error
        (incorrect configuration or missing/wrong data). That ends with step
        state set to failed.
        If error occurs due to uncaught exception in this method, step state
        will be set to error
        """
        raise NotImplementedError

    @classmethod
    def from_json(cls, json_data, _locals={}) -> "Traction":
        """Deserialize class instance from json representation."""
        args = {}
        outs = {}
        data = json_data["$data"]
        type_cls = TypeNode.from_json(json_data["$type"], _locals=_locals).to_type()
        for f, ftype in cls._fields.items():
            if f.startswith("i_") and isinstance(data[f], str):
                continue
            elif (
                f.startswith("a_")
                or f.startswith("i_")
                or f.startswith("r_")
                or f in ("errors", "stats", "details")
            ):
                if f.startswith("r_"):
                    args[f] = (
                        TypeNode.from_json(data[f]["$type"], _locals=_locals)
                        .to_type()
                        .from_json(data[f], _locals=_locals)
                    )
                else:
                    args[f] = ftype.from_json(data[f], _locals=_locals)
            elif f.startswith("i_"):
                args[f] = ftype.from_json(data[f], _locals=_locals)
            elif f.startswith("o_"):
                outs[f] = ftype.from_json(data[f], _locals=_locals)
            elif f == "tractions":
                args[f] = TList[Union[Traction, None]].from_json(data[f], _locals=_locals)
            elif f == "tractions_state":
                args[f] = TList[TractionState].from_json(data[f], _locals=_locals)
            elif f == "state":
                args[f] = TractionState(data[f])
            else:
                args[f] = data[f]
        ret = type_cls(**args)
        for o, oval in outs.items():
            setattr(ret, o, oval)
        return ret

    @classmethod
    def _to_json_schema(cls) -> Dict[str, Any]:
        """Return json schema representation of the class."""
        result = {}
        tree = ToJsonSchemaTree(result)
        tree.add_to_process(data=cls, data_type=cls, parent_index="root", result=result)
        tree.process()
        return result["root"]
