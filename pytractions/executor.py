import logging
import multiprocessing
import traceback
from typing import Literal

import ray

from pytractions.base import Base, TDict, JSON_COMPATIBLE
from pytractions.stmd_utils import _init_traction
from pytractions.traction import TractionState, TractionStats

from concurrent.futures import ThreadPoolExecutor as _ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor as _ProcessPoolExecutor
from concurrent.futures import as_completed

LOGGER = logging.getLogger(__name__)


class Executor(Base):
    """Executor abstract class."""

    pass


class TractionResult(Base):
    """Results of traction running in executor."""

    outputs: TDict[str, JSON_COMPATIBLE]
    state: TractionState
    stats: TractionStats


def _execute_traction(index, uid, traction_cls, inputs, args, resources, observer):
    """Eexecute traction and return outputs."""
    traction = _init_traction(
        traction_cls, inputs, resources, args, index, uid=uid, observer=observer
    )
    traction.run()
    outputs: TDict[str, JSON_COMPATIBLE] = TDict[str, JSON_COMPATIBLE]({})
    for o in traction._fields:
        if o.startswith("o_"):
            outputs[o] = getattr(traction, "_raw_" + o).data
    return TractionResult(outputs=outputs, state=traction.state, stats=traction.stats)


class ProcessPoolExecutor(Executor):
    """Execute tractions in parallel using pythons concurrent ProcessPoolExecutor."""

    pool_size: int = multiprocessing.cpu_count()
    executor_type: Literal["process_pool_executor"]

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

    def execute(self, index, uid, traction_cls, inputs, args, resources, observer=None):
        """Execute the traction with given inputs args and resources."""
        res = self._executor.submit(
            _execute_traction,
            index,
            uid,
            traction_cls,
            inputs,
            args,
            resources,
            observer,
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
            exc = ft.exception()
            if exc:
                for line in traceback.format_exception(exc):
                    LOGGER.error("Error in traction execution: %s", line)
            uid = self._outputs[ft]
            completed[uid] = ft.result().outputs
        return completed

    def get_states(self, uids):
        """Fetch outputs of executed tractions."""
        states = []
        completed = {}
        for uid in uids:
            states.append(self._outputs_by_uid[uid])
        for ft in as_completed(states):
            uid = self._outputs[ft]
            if ft.exception():
                completed[uid] = (TractionState.ERROR, ft.result().stats)
            else:
                completed[uid] = (ft.result().state, ft.result().stats)
        return completed

    def clear_output(self, uid):
        """Clear stored outputs."""
        ft = self._outputs_by_uid[uid]
        self._outputs_by_uid[uid] = None
        self._outputs[ft] = None


class ThreadPoolExecutor(Executor):
    """Execute tractions in parallel using pythons concurrent ThreadPoolExecutor."""

    pool_size: int = 1
    executor_type: Literal["thread_pool_executor"]

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

    def execute(self, index, uid, traction, inputs, args, resources, observer=None):
        """Execute the traction with given inputs args and resources."""
        res = self._executor.submit(
            _execute_traction, index, uid, traction, inputs, args, resources, observer
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
            exc = ft.exception()
            if exc:
                for line in traceback.format_exception(exc):
                    LOGGER.error("Error in traction execution: %s", line)
            uid = self._outputs[ft]
            completed[uid] = ft.result().outputs
        return completed

    def get_states(self, uids):
        """Fetch outputs of executed tractions."""
        states = []
        completed = {}
        for uid in uids:
            states.append(self._outputs_by_uid[uid])
        for ft in as_completed(states):
            uid = self._outputs[ft]
            if ft.exception():
                print("Error in traction execution: %s", ft.exception())
                completed[uid] = (TractionState.ERROR, ft.result().stats)
            else:
                completed[uid] = (ft.result().state, ft.result().stats)
        return completed

    def clear_output(self, uid):
        """Clear stored outputs."""
        ft = self._outputs_by_uid[uid]
        self._outputs_by_uid[uid] = None
        self._outputs[ft] = None


class LoopExecutor(Executor):
    """Execute tractions in sequentially in for loop."""

    executor_type: Literal["loop_executor"]

    def __post_init__(self):
        """Initialize executor."""
        self._outputs = {}

    def execute(self, index, uid, traction, inputs, args, resources, observer):
        """Execute the traction with given inputs args and resources."""
        try:
            res = _execute_traction(index, uid, traction, inputs, args, resources, observer)
        except Exception:
            LOGGER.error("Error in traction execution: %s", traceback.format_exc())
            res = TractionResult(
                outputs=TDict[str, JSON_COMPATIBLE]({}),
                state=TractionState.ERROR,
                stats=TractionStats(),
            )
        self._outputs[uid] = res

    def get_states(self, uids):
        """Fetch outputs of executed tractions."""
        completed = {}
        for uid in uids:
            completed[uid] = (self._outputs[uid].state, self._outputs[uid].stats)
        return completed

    def get_outputs(self, uids):
        """Fetch outputs of executed tractions."""
        outs = {}
        for uid in uids:
            outs[uid] = self._outputs[uid].outputs
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


@ray.remote
def _ray_execute_traction(index, uid, traction_cls, inputs, args, resources, observer=None):
    """Eexecute traction and return outputs."""
    traction = _init_traction(
        traction_cls, inputs, resources, args, index, uid=uid, observer=observer
    )
    traction.run()
    outputs: TDict[str, JSON_COMPATIBLE] = TDict[str, JSON_COMPATIBLE]({})
    for o in traction._fields:
        if o.startswith("o_"):
            outputs[o] = getattr(traction, "_raw_" + o).data
    return TractionResult(state=traction.state, stats=traction.stats, outputs=outputs).to_json()


class RayExecutor(Executor):
    """Execute tractions in sequentially in for loop."""

    pool_size: int = multiprocessing.cpu_count()

    def __post_init__(self):
        """Initialize executor."""
        self._outputs = {}

    def execute(self, index, uid, traction, inputs, args, resources, observer):
        """Execute the traction with given inputs args and resources."""
        res = _ray_execute_traction.remote(index, uid, traction, inputs, args, resources, observer)
        self._outputs[uid] = res

    def get_outputs(self, uids):
        """Fetch outputs of executed tractions."""
        _outs = []
        for uid in uids:
            _outs.append(self._outputs[uid])

        outputs = {}
        _serialized_outputs_ = ray.get(_outs)

        d_outputs = []
        for out in _serialized_outputs_:
            d_out = Base.from_json(out)
            d_outputs.append(d_out)

        for out, uid in zip(d_outputs, uids):
            outputs[uid] = out.outputs

        return outputs

    def get_states(self, uids):
        """Fetch outputs of executed tractions."""
        _outs = []
        for uid in uids:
            _outs.append(self._outputs[uid])

        states = {}
        _serialized_outputs_ = ray.get(_outs)
        d_states = []
        for out in _serialized_outputs_:
            d_out = Base.from_json(out)
            d_states.append(d_out)

        for out, uid in zip(d_states, uids):
            states[uid] = (out.state, out.stats)

        return states

    def clear_output(self, uid):
        """Clear stored outputs."""
        self._outputs[uid] = None

    def init(self):
        """Start the executor."""
        ray.init(num_cpus=self.pool_size, ignore_reinit_error=True)

    def shutdown(self):
        """Shutdown the executor."""
        ray.shutdown()
        return None
