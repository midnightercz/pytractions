import logging

from pytractions.base import Base

from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor
from concurrent.futures import as_completed

LOGGER = logging.getLogger(__name__)


class Executor(Base):
    """Executor abstract class."""

    pass


def _copy_traction(traction, inputs, resources, args, uid="0"):
    """After tractor is created, all tractions needs to be copied.

    This way it's possible to use same Tractor class multiple times
    """
    init_fields = {}
    for ft, field in traction.__dataclass_fields__.items():
        # set all inputs for the traction to outputs of traction copy
        # created bellow

        if ft.startswith("r_"):
            init_fields[ft] = resources[ft]

        elif ft.startswith("a_"):
            init_fields[ft] = args[ft]

        elif ft.startswith("i_") and ft in inputs:
            init_fields[ft] = inputs[ft]

    init_fields["uid"] = uid
    # create copy of existing traction
    ret = traction(**init_fields)
    return ret


def _execute_traction(uid, traction, inputs, args, resources, on_update=None):
    """Eexecute traction and return outputs."""
    traction = _copy_traction(traction, inputs, resources, args, uid=uid)
    traction.run(on_update=on_update)
    outputs = {}
    for o in traction._fields:
        if o.startswith("o_"):
            outputs[o] = getattr(traction, o)
    return outputs


class ProcessPoolExecutor(Executor):
    """Execute tractions in parallel using pythons concurrent ProcessPoolExecutor."""

    pool_size: int = 1

    def __post_init__(self):
        """Initialize executor."""
        self._outputs = {}
        self._outputs_by_uid = {}
        self._inited = False

    def init(self):
        """Start the executor."""
        if not self._inited:
            self._executor = _ProcessPoolExecutor(max_workers=self.pool_size)

    def shutdown(self):
        """Shutdown the executor."""
        self._executor.shutdown()

    def execute(self, uid, traction, inputs, args, resources, on_update=None):
        """Execute the traction with given inputs args and resources."""
        res = self._executor.submit(
            _execute_traction,
            uid,
            traction,
            inputs,
            args,
            resources,
        )
        self._outputs[res] = uid
        self._outputs_by_uid[uid] = res

    def get_outputs(self, uids):
        """Fetch outputs of executed tractions."""
        outs = []
        completed = {}
        for uid in uids:
            outs.append(self._outputs_by_uid[uid])
        for ft in as_completed(outs):
            uid = self._outputs[ft]
            completed[uid] = ft.result()
        return completed

    def clear_output(self, uid):
        """Clear stored outputs."""
        ft = self._outputs_by_uid[uid]
        self._outputs_by_uid[uid] = None
        self._outputs[ft] = None


class ThreadPoolExecutor(Executor):
    """Execute tractions in parallel using pythons concurrent ThreadPoolExecutor."""

    pool_size: int = 1

    def __post_init__(self):
        """Initialize executor."""
        self._outputs = {}
        self._outputs_by_uid = {}
        self._inited = False

    def init(self):
        """Start the executor."""
        if not self._inited:
            self._executor = _ThreadPoolExecutor(max_workers=self.pool_size)

    def shutdown(self):
        """Shutdown the executor."""
        self._executor.shutdown()

    def execute(self, uid, traction, inputs, args, resources, on_update=None):
        """Execute the traction with given inputs args and resources."""
        res = self._executor.submit(_execute_traction, uid, traction, inputs, args, resources)
        self._outputs[res] = uid
        self._outputs_by_uid[uid] = res

    def get_outputs(self, uids):
        """Fetch outputs of executed tractions."""
        outs = []
        completed = {}
        for uid in uids:
            outs.append(self._outputs_by_uid[uid])
        for ft in as_completed(outs):
            uid = self._outputs[ft]
            completed[uid] = ft.result()
        return completed

    def clear_output(self, uid):
        """Clear stored outputs."""
        ft = self._outputs_by_uid[uid]
        self._outputs_by_uid[uid] = None
        self._outputs[ft] = None


class LoopExecutor(Executor):
    """Execute tractions in sequentially in for loop."""

    def __post_init__(self):
        """Initialize executor."""
        self._outputs = {}

    def execute(self, uid, traction, inputs, args, resources, on_update=None):
        """Execute the traction with given inputs args and resources."""
        res = _execute_traction(uid, traction, inputs, args, resources, on_update=on_update)
        self._outputs[uid] = res

    def get_outputs(self, uids):
        """Fetch outputs of executed tractions."""
        outs = {}
        for uid in uids:
            outs[uid] = self._outputs[uid]
        return outs

    def clear_output(self, uid):
        """Clear stored outputs."""
        self._outputs[uid] = None

    def init(self):
        """Start the executor."""
        return None

    def shutdown(self):
        """Shutdown the executor."""
        return None
