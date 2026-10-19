"""One integration owner for the staged per-model cone and row cache.

Imported only after the root runner has acquired the shared model-process lock.
This module changes no qualified baseline source and no package-wide method.
"""
from contextlib import contextmanager
from pathlib import Path

from evidence import ROOT, import_path


def native_hooks():
    from omlx.patches.deepseek_v41 import language, storage
    from omlx.patches.deepseek_v41.quantization import QuantizedProjection
    return {
        "language_call": language.LanguageModel.__call__,
        "language_prefill": language.LanguageModel._omlx_prefill,
        "language_append": language.LanguageModel.dspark_append_context,
        "block": language.Block.__call__,
        "projection_call": QuantizedProjection.__call__,
        "projection": QuantizedProjection.project_quantized,
        "gate": language.Gate.__call__, "moe": language.MoE.__call__,
        "hc": language.hc_mixes,
        "disk_read": storage.DiskEngramEmbedding._read_rows,
        "disk_close": storage.DiskEngramEmbedding.close,
    }


def parameter_identity(model):
    from mlx.utils import tree_flatten
    return tuple((key, id(value), tuple(value.shape), str(value.dtype))
                 for key, value in tree_flatten(model.parameters()))


def drained(target):
    prefetch = getattr(target, "_engram_prefetch", None)
    if prefetch is not None and (prefetch._closed or prefetch._pending is not None):
        raise RuntimeError("Require an open drained prefetcher between model requests")


class Candidate:
    def __init__(self, model, *, record_events=True):
        self.model, self.target = model, model.language_model
        self.original_hooks = native_hooks()
        self.original_parameters = parameter_identity(model)
        self.cone = self.rows = None
        self.closed = False
        drained(self.target)
        cone_module = import_path("_serving_v3_cone", ROOT / "work/serving-cone-v3/cone_serving.py")
        cache_module = import_path("_serving_v3_rows", ROOT / "work/serving-cache-v3/serving_cache.py")
        try:
            self.cone = cone_module.install(self.target, enabled=False, record_events=record_events)
            self.rows = cache_module.install(model, enabled=False, budget_per_table=256 * 2**20)
            self.verify()
        except BaseException:
            self.close()
            raise

    def verify(self):
        if native_hooks() != self.original_hooks:
            raise RuntimeError("Candidate changed a process-wide inference method")
        if parameter_identity(self.model) != self.original_parameters:
            raise RuntimeError("Candidate changed model parameter names or tensor identity")

    def set_enabled(self, enabled):
        if type(enabled) is not bool or self.closed:
            raise ValueError("Open controller requires an explicit boolean mode")
        drained(self.target)
        self.rows.set_enabled(enabled)
        self.cone.set_enabled(enabled)

    def snapshot(self, *, clear_cone=False):
        return {"cone": self.cone.snapshot(clear=clear_cone), "rows": self.rows.snapshot()}

    def close(self):
        if self.closed:
            return
        # Reverse attachment order; native model.close may already have released
        # the row arrays. Both owners promise idempotent detach.
        error = None
        for owner in (self.rows, self.cone):
            if owner is not None:
                try:
                    owner.close()
                except BaseException as exc:
                    error = error or exc
        try:
            self.verify()
        except BaseException as exc:
            error = error or exc
        else:
            if ((self.rows is None or self.rows.status in ("detached", "closed", "failed_install"))
                    and (self.cone is None or self.cone.closed)):
                self.model = self.target = None
                self.original_parameters = None
                self.closed = True
        if error is not None:
            raise error


@contextmanager
def serving_prefill_entry(target, adapter):
    """Exercise the real VLM adapter while retaining the frozen bench loop.

    The adapter calls target.__call__, so this instance-only bridge cannot
    recurse into target._omlx_prefill. Both A/B arms use the same entry route.
    """
    namespace = object.__getattribute__(target, "__dict__")
    present = "_omlx_prefill" in namespace
    prior = namespace.get("_omlx_prefill")
    object.__setattr__(target, "_omlx_prefill", adapter._omlx_prefill)
    try:
        yield
    finally:
        if present:
            object.__setattr__(target, "_omlx_prefill", prior)
        else:
            object.__delattr__(target, "_omlx_prefill")
