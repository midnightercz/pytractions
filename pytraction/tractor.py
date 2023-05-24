from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import dataclasses
from typing import Optional, Dict
from typing_extensions import Self


from .base import (
    Traction, TractionStats, TractionState, TList, TractionMeta, Arg, In, Out, Res, ANY, TypeNode,
    OnUpdateCallable, OnErrorCallable, on_update_empty, TractionFailedError,
    NoData, isodate_now

)


class TractorMeta(TractionMeta):
    @classmethod
    def _attribute_check(cls, attr, type_):
        if attr not in ('uid', 'state', 'skip', 'skip_reason', 'errors', 'stats', 'details', 'tractions'):
            if attr.startswith("i_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(In[ANY]):
                    raise TypeError(f"Attribute {attr} has to be type In[ANY], but is {type_}")
            elif attr.startswith("o_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(Out[ANY]):
                    raise TypeError(f"Attribute {attr} has to be type Out[ANY], but is {type_}")
            elif attr.startswith("a_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(Arg[ANY]):
                    raise TypeError(f"Attribute {attr} has to be type Arg[ANY], but is {type_}")
            elif attr.startswith("r_"):
                if TypeNode.from_type(type_, subclass_check=False) != TypeNode.from_type(Res[ANY]):
                    raise TypeError(f"Attribute {attr} has to be type Res[ANY], but is {type_}")
            elif attr.startswith("t_"):
                if TypeNode.from_type(type_, subclass_check=True) != TypeNode.from_type(Traction):
                    raise TypeError(f"Attribute {attr} has to be type Traction, but is {type_}")
            else:
                raise TypeError(f"Attribute {attr} has start with i_, o_, a_, r_ or t_")

    @classmethod
    def _before_new(cls, attrs):
        # mapping which holds tractor inputs + tractions outputs
        outputs_map = {}
        # inputs map to map (traction, input_name) -> (traction/tractor, output_name)
        io_map = {}
        outputs_all = []
        resources = {}
        resources_map = {}
        # map for tractor outputs o -> (traction, o_name)
        t_outputs_map = {}
        args = {}
        args_map = {}

        output_waves = {}
        traction_waves = {}

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

        for t in attrs['_fields']:
            if not t.startswith("t_"):
                continue
            traction = attrs[t]
            wave = 0
            for tf in traction._fields:
                tfo = getattr(traction, tf)
                if tf.startswith("i_"):
                    if TypeNode.from_type(type(tfo), subclass_check=False) != TypeNode.from_type(NoData[ANY]):
                        if id(getattr(traction, tf)) not in outputs_all:
                            raise ValueError(f"Input {traction.__class__}[{traction.uid}]->{tf} is mapped to output which is not known yet")
                    if id(tfo) in outputs_map:
                        io_map[(t, tf)] = outputs_map[id(tfo)]
                        wave = max(output_waves[id(tfo)], wave)
            traction_waves[t] = wave + 1

            for tf in traction._fields:
                tfo = getattr(traction, tf)
                if tf.startswith("o_"):
                    outputs_all.append(id(tfo))
                    outputs_map[id(tfo)] = (t, tf)
                    output_waves[id(tfo)] = traction_waves[t]
                elif tf.startswith("i_"):
                    if TypeNode.from_type(type(tfo), subclass_check=False) != TypeNode.from_type(NoData[ANY]):
                        if id(getattr(traction, tf)) not in outputs_all:
                            raise ValueError(f"Input {traction.__class__}[{traction.uid}]->{tf} is mapped to output which is not known yet")
                    if id(tfo) in outputs_map:
                        io_map[(t, tf)] = outputs_map[id(tfo)]
                elif tf.startswith("r_"):
                    resources_map[(t, tf)] = resources[id(tfo)]
                elif tf.startswith("a_") and id(tfo) in args:
                    args_map[(t, tf)] = args[id(tfo)]

        for f, fo in attrs.items():
            if f.startswith("o_"):
                t_outputs_map[f] = outputs_map[id(fo)]

        attrs['_outputs_map'] = t_outputs_map
        attrs['_resources_map'] = resources_map
        attrs['_args_map'] = args_map
        attrs['_io_map'] = io_map
        attrs['_traction_waves'] = traction_waves


class Tractor(Traction, metaclass=TractorMeta):
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
            elif ft.startswith("a_") and (traction_name, ft) in self._args_map:
                self_field = self._args_map[(traction_name, ft)]
                init_fields[ft] = getattr(self, self_field)

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
                t, tf = self._outputs_map[f]
                self._no_validate_setattr_(f, getattr(self._tractions[t], tf))
        #print("Traction waves", self._traction_waves)

    def _run(self, on_update: Optional[OnUpdateCallable] = None) -> Self:  # pragma: no cover
        for traction in self.tractions:
            traction.run(on_update=on_update)
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

    def _traction_runner(self, traction, on_update=None):
        traction = self._init_traction()
        traction.run(on_update=on_update)
        return traction

    def _run(self, on_update: Optional[OnUpdateCallable] = None) -> Self:  # pragma: no cover
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

        for w, tractions in traction_groups.items():
            with executor_class(max_workers=self.a_pool_size.a) as executor:
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
