from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import dataclasses
from typing import Optional, Dict, Any
from typing_extensions import Self


from .base import (
    Traction, TractionStats, TractionState, TList, TractionMeta, Arg, MultiArg, In, Out, Res, ANY, TypeNode,
    OnUpdateCallable, OnErrorCallable, on_update_empty, TractionFailedError,
    NoData, isodate_now

)


class TractorMeta(TractionMeta):
    @classmethod
    def _attribute_check(cls, attr, type_, all_attrs):
        if attr not in ('uid', 'state', 'skip', 'skip_reason', 'errors', 'stats', 'details', 'tractions'):
            if attr.startswith("i_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(In[ANY]):
                    raise TypeError(f"Attribute {attr} has to be type In[ANY], but is {type_}")
            elif attr.startswith("o_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(Out[ANY]):
                    raise TypeError(f"Attribute {attr} has to be type Out[ANY], but is {type_}")
            elif attr.startswith("a_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(Arg[ANY]) and\
                   TypeNode.from_type(type_, subclass_check=True) != TypeNode.from_type(MultiArg):
                    raise TypeError(f"Attribute {attr} has to be type Arg[ANY] or MultiArg, but is {type_}")
            elif attr.startswith("r_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(Res[ANY]):
                    raise TypeError(f"Attribute {attr} has to be type Res[ANY], but is {type_}")
            elif attr.startswith("t_"):
                if TypeNode.from_type(type_, subclass_check=True) != TypeNode.from_type(Traction):
                    raise TypeError(f"Attribute {attr} has to be type Traction, but is {type_}")
            elif attr.startswith("d_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(str):
                    raise TypeError(f"Attribute {attr} has to be type str, but is {type_}")
                if attr.replace("d_", "", 1) not in all_attrs['__annotations__']:
                    raise TypeError(f"Attribute {attr.replace('d_', '', 1)} is not defined for description {attr}: {all_attrs}")
            else:
                raise TypeError(f"Attribute {attr} has start with i_, o_, a_, r_ or t_")

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

        for dst_o, _attr in [(outputs_map, '_outputs_map'),
                             (io_map, '_io_map'),
                             (resources, '_resources'),
                             (resources_map, '_resources_map'),
                             (outputs_map, '_outputs_map'),
                             (t_outputs_map, '_t_outputs_map'),
                             (args_map, '_args_map'),
                             (margs_map, '_margs_map'),
                             (args, '_args'),
                             (margs, '_margs'),
                             (output_waves, '_output_waves'),
                             (traction_waves, '_traction_waves'),
                            ]:
            for base in bases:
                if hasattr(base, _attr):
                    for k, v in getattr(base, _attr).items():
                        if k not in dst_o:
                            dst_o[k] = v

        outputs_all = []
        for base in bases:
            if hasattr(base, "_outputs_all"):
                for v in base._outputs_all:
                    outputs_all.append(v)

        for f, fo in attrs.items():
            if f.startswith("i_"):
                outputs_map[id(fo)] = ("#", f)
                output_waves[id(fo)] = 0
                outputs_all.append(id(fo))
            if f.startswith("o_"):
                outputs_all.append(id(fo))
            if f.startswith("r_"):
                resources[id(fo)] = f
            if f.startswith("a_"):
                args[id(fo)] = f
                if isinstance(fo, MultiArg):
                    for maf in fo._fields:
                        mafo = getattr(fo, maf)
                        margs[id(mafo)] = (f, maf)
                        if id(mafo) in args:
                            margs_map[("#", f, maf)] = args[id(mafo)]

        for t in attrs['_fields']:
            if not t.startswith("t_"):
                continue
            traction = attrs[t]
            wave = 0
            # in the case of using inputs from parent
            if isinstance(traction, dataclasses.Field):
                traction_fields = traction.default._fields
                _traction = traction.default
            else:
                traction_fields = traction._fields
                _traction = traction
            for tf in traction_fields:
                tfo = getattr(_traction, tf)
                if tf.startswith("i_"):
                    if TypeNode.from_type(type(tfo), subclass_check=False) != TypeNode.from_type(NoData[ANY]):
                        if id(getattr(_traction, tf)) not in outputs_all:
                            raise ValueError(f"Input {_traction.__class__}[{_traction.uid}]->{tf} is mapped to output which is not known yet")
                    if id(tfo) in outputs_map:
                        io_map[(t, tf)] = outputs_map[id(tfo)]
                        wave = max(output_waves[id(tfo)], wave)
            traction_waves[t] = wave + 1

            for tf in _traction._fields:
                tfo = getattr(_traction, tf)

                #if tf.startswith("a_"):
                #    print("ID", id(tfo), tf)

                if tf.startswith("o_"):
                    outputs_all.append(id(tfo))
                    outputs_map[id(tfo)] = (t, tf)
                    output_waves[id(tfo)] = traction_waves[t]
                elif tf.startswith("i_"):
                    if TypeNode.from_type(type(tfo), subclass_check=False) != TypeNode.from_type(NoData[ANY]):
                        if id(getattr(_traction, tf)) not in outputs_all:
                            raise ValueError(f"Input {_traction.__class__}[{_traction.uid}]->{tf} is mapped to output which is not known yet")
                    if id(tfo) in outputs_map:
                        io_map[(t, tf)] = outputs_map[id(tfo)]
                elif tf.startswith("r_"):
                    if id(tfo) in resources:
                        resources_map[(t, tf)] = resources[id(tfo)]
                    else:
                        raise ValueError(f"Resources {t}.{tf} is not map to any parent resource")

                elif tf.startswith("a_") and id(tfo) in args:
                    args_map[(t, tf)] = args[id(tfo)]

                elif tf.startswith("a_") and id(tfo) in margs:
                    ##print("!!!!!!!!", tf, margs)
                    args_map[(t, tf)] = margs[id(tfo)]

                elif tf.startswith("a_") and TypeNode.from_type(type(tfo), subclass_check=True) == TypeNode.from_type(MultiArg):
                    #print("Multiarg", t, tf)
                    for maf, mafo in tfo._fields.items():
                        if id(mafo) in args:
                            margs_map[(t, tf, maf)] = args[id(mafo)]

        #print("OUTPUTS MAP", outputs_map)
        for f, fo in attrs.items():
            #print("F", f, id(fo))#, fo)
            if f.startswith("o_"):
                if id(fo) in outputs_map:
                    t_outputs_map[f] = outputs_map[id(fo)]


        attrs['_t_outputs_map'] = t_outputs_map
        attrs['_output_waves'] = output_waves
        attrs['_outputs_map'] = outputs_map
        attrs['_resources'] = resources
        attrs['_resources_map'] = resources_map
        attrs['_args_map'] = args_map
        attrs['_margs_map'] = margs_map
        attrs['_io_map'] = io_map
        attrs['_args'] = args
        attrs['_margs'] = margs
        attrs['_outputs_all'] = outputs_all
        attrs['_traction_waves'] = traction_waves


class Tractor(Traction, metaclass=TractorMeta):
    _TYPE: str = "TRACTOR"
    _CUSTOM_TYPE_TO_JSON: bool = True
    uid: str
    state: str = "ready"
    skip: bool = False
    skip_reason: Optional[str] = ""
    errors: TList[str] = dataclasses.field(default_factory=TList[str])
    stats: TractionStats = dataclasses.field(default_factory=TractionStats)
    details: TList[str] = dataclasses.field(default_factory=TList[str])
    tractions: TList[Traction] = dataclasses.field(default_factory=TList[Traction], init=False)

    def _init_traction(self, traction_name, traction):
        init_fields = {}
        for ft, field in traction.__dataclass_fields__.items():
            # set all inputs for the traction to outputs of traction copy
            # created bellow
            if ft.startswith("i_"):
                if (traction_name, ft) in self._io_map:
                    source, o_name = self._io_map[(traction_name, ft)]
                    if source == "#":
                        init_fields[ft] = getattr(self, o_name)
                    else:
                        init_fields[ft] = getattr(self._tractions[source], o_name)
            elif ft.startswith("r_"):
                self_field = self._resources_map[(traction_name, ft)]
                init_fields[ft] = getattr(self, self_field)
            elif ft.startswith("a_"):
                # First check if whole argument can be found in map
                if (traction_name, ft) in self._args_map:
                    #print("TRACTION ARG", traction_name, ft)
                    self_field = self._args_map[(traction_name, ft)]
                    if isinstance(self_field, tuple):
                        init_fields[ft] = getattr(getattr(self, self_field[0]), self_field[1])
                    else:
                        init_fields[ft] = getattr(self, self_field)

                elif TypeNode.from_type(field.type, subclass_check=True) == TypeNode.from_type(MultiArg):
                    ma_init_fields = {}
                    for maf, _ in field.type._fields.items():
                        #print(ma_init_fields, self._margs_map)
                        if (traction_name, ft, maf) in self._margs_map:
                            self_field = self._margs_map[(traction_name, ft, maf)]
                            ma_init_fields[maf] = getattr(self, self_field)
                    init_fields[ft] = field.type(**ma_init_fields)
                    #print("multiarg traction init", traction_name, ft, init_fields[ft])
                elif (traction_name, ft) not in self._args_map:
                    #print("default traction init", traction_name, ft, getattr(traction, ft))
                    init_fields[ft] = getattr(traction, ft)
            elif ft == 'uid':
                init_fields[ft] = "%s::%s" % (self.uid, getattr(traction, ft))
            # if field doesn't start with _ include it in init_fields to
            # initialize the traction copy
            elif field.init:
                if ft.startswith("_"):
                    continue
                init_fields[ft] = getattr(traction, ft)
        return traction.__class__(**init_fields)

    def __post_init__(self):
        self._tractions = {}
        self.tractions.clear()
        for f in self._fields:
            # Copy all tractions
            if f.startswith("t_"):
                traction = getattr(self, f)
                new_traction = self._init_traction(f, traction)
                self._tractions[f] = new_traction

                # also put new traction in tractions list used in run
                self.tractions.append(new_traction)
        for f in self._fields:
            if f.startswith("o_"):
                # regular __setattr__ don't overwrite whole output model but just
                # data in it to keep connection, so need to use _no_validate_setattr
                t, tf = self._t_outputs_map[f]
                self._no_validate_setattr_(f, getattr(self._tractions[t], tf))
            if f.startswith("a_"):
                # for MulArgs which are mapped to args, overwrite them
                fo = getattr(self, f)
                if isinstance(fo, MultiArg):
                    #print("post init ma", f, self._margs_map)
                    for maf in fo._fields:
                        if ("#", f, maf) in self._margs_map:
                            setattr(fo, maf, getattr(self, self._margs_map[("#", f, maf)]))

        #print("--")
        #print("Traction waves", self._traction_waves)

    def _run(self, on_update: Optional[OnUpdateCallable] = None) -> Self:  # pragma: no cover
        for n, traction in enumerate(self.tractions):
            traction.run(on_update=on_update)
            setattr(self, list(self._tractions.keys())[n], traction)
            if on_update:
                on_update(self)
            if traction.state == TractionState.ERROR:
                self.state = TractionState.ERROR
                return self
            if traction.state == TractionState.FAILED:
                self.state = TractionState.FAILED
                return self
        return self

    def run(
        self,
        on_update: Optional[OnUpdateCallable] = None,
        on_error: Optional[OnErrorCallable] = None,
    ) -> Self:
        _on_update: OnUpdateCallable = on_update or on_update_empty
        _on_error: OnErrorCallable = on_error or on_update_empty
        self._reset_stats()
        if self.state == TractionState.READY:
            self.stats.started = isodate_now()

            self.state = TractionState.PREP
            self._pre_run()
            _on_update(self)  # type: ignore
        try:
            if self.state not in (TractionState.PREP, TractionState.ERROR):
                return self
            if not self.skip:
                self.state = TractionState.RUNNING
                _on_update(self)  # type: ignore
                self._run(on_update=_on_update)
        except TractionFailedError:
            self.state = TractionState.FAILED
        except Exception as e:
            self.state = TractionState.ERROR
            self.errors.append(str(e))
            _on_error(self)
            raise
        else:
            self.state = TractionState.FINISHED
        finally:
            self._finish_stats()
            _on_update(self)  # type: ignore
        return self

    @classmethod
    def type_to_json(cls) -> Dict[str, Any]:
        pre_order: Dict[str, Any] = {}
        # stack is list of (current_cls_to_process, current_parent, current_key, current_default)
        stack: List[Tuple[Base, Dict[str, Any], str]] = [(cls, pre_order, "root", None)]
        while stack:
            current, current_parent, parent_key, current_default = stack.pop(0)
            if hasattr(current, "_CUSTOM_TYPE_TO_JSON") and current._CUSTOM_TYPE_TO_JSON and current != cls:
                current_parent[parent_key] = current.type_to_json()
            elif hasattr(current, "_fields"):
                current_parent[parent_key] = {"$type": TypeNode.from_type(current).to_json()}
                for f, ftype in current._fields.items():
                    if type(current.__dataclass_fields__[f].default) in (str, int, float, None):
                        stack.append((ftype, current_parent[parent_key], f, current.__dataclass_fields__[f].default))
                    else:
                        stack.append((ftype, current_parent[parent_key], f, None))
            else:
                current_parent[parent_key] = {"$type": TypeNode.from_type(current).to_json(), "default": current_default}
        pre_order['root']['$io_map'] = [[list(k), list(v)] for k, v in cls._io_map.items()]
        pre_order['root']['$resource_map'] = [[list(k), v] for k, v in cls._resources_map.items()]
        return pre_order['root']

    @classmethod
    def from_json(cls, json_data) -> Self:
        args = {}
        outs = {}
        tractions = {}
        traction_outputs = {}
        #print(json_data)
        for f, ftype in cls._fields.items():
            if f.startswith("i_") and isinstance(json_data[f], str):
                continue
            elif f.startswith("t_"):
                #print("TRACTOR FROM JSON F", f, ftype)
                args[f] = ftype.from_json(json_data[f])
                tractions[f] = args[f]
                for tf in tractions[f]._fields:
                    if tf.startswith("o_"):
                        traction_outputs.setdefault(tractions[f].fullname,{})[tf] = getattr(tractions[f], tf)
                for tf, tfval in json_data[f].items():
                    if tf.startswith("i_") and isinstance(tfval, str):
                        traction_name, o_name = tfval.split("#")
                        setattr(tractions[f], tf, traction_outputs[traction_name][o_name])
            elif f.startswith("a_") or f.startswith("i_") or f.startswith("r_") or f in ("errors", "stats", "details", "tractions"):
                # skip if there are no data to load
                if json_data[f].get("$data"):
                    #print("TRACTOR LOAD F", f, json_data[f].get("$data"))
                    args[f] = ftype.from_json(json_data[f])
            elif f.startswith("o_"):
                outs[f] = ftype.from_json(json_data[f])
            else:
                args[f] = json_data[f]
        ret = cls(**args)
        for o, oval in outs.items():
            getattr(ret, o).data = oval.data
        return ret


class MultiTractor(Tractor, metaclass=TractorMeta):
    _TYPE: str = "MULTITRACTOR"
    uid: str
    state: str = "ready"
    skip: bool = False
    skip_reason: Optional[str] = ""
    errors: TList[str] = dataclasses.field(default_factory=TList[str])
    stats: TractionStats = dataclasses.field(default_factory=TractionStats)
    details: TList[str] = dataclasses.field(default_factory=TList[str])
    tractions: TList[Traction] = dataclasses.field(default_factory=TList[Traction], init=False)
    a_pool_size: Arg[int]
    a_use_processes: Arg[bool] = Arg[bool](a=False)

    def _traction_runner(self, t_name, traction, on_update=None):
        traction = self._init_traction(t_name, traction)
        traction.run(on_update=on_update)
        return traction

    def _run(self, on_update: Optional[OnUpdateCallable] = None):  # pragma: no cover
        _on_update: OnUpdateCallable = lambda step: None
        if on_update:
            _on_update = on_update

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
                #print("Running wave", w)
                ft_results = {}
                for t_name, traction in tractions.items():
                    res = executor.submit(self._traction_runner, t_name, traction, on_update=on_update)
                    ft_results[res] = t_name
                _on_update(self)
                for ft in as_completed(ft_results):
                    t_name = ft_results[ft]
                    nt = ft.result()
                    self._tractions[t_name] = nt
                _on_update(self)

        for f in self._fields:
            if f.startswith("o_"):
                # regular __setattr__ don't overwrite whole output model but just
                # data in it to keep connection, so need to use _no_validate_setattr
                t, tf = self._outputs_map[f]
                self._no_validate_setattr_(f, getattr(self._tractions[t], tf))
        return self