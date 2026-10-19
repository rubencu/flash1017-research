"""CPU-only fixtures: native method bodies via AST, tiny immutable NumPy rows.

No runtime/MLX import, weight file, GPU allocation or serving call. Concurrent
fixtures exercise the actual native one-worker prefetch manager and its
pending=None-before-Future.result behavior, using events rather than timing
assumptions to order the critical transitions.
"""
import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import gc
import hashlib
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest
import weakref

import numpy as np

from serving_cache import install, MAX_TABLE_BYTES, UPSTREAM_SHA256, dependency_fingerprints

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/flash1017-lab/_runtime/omlx/patches/deepseek_v41/storage.py"
TREE = ast.parse(SOURCE.read_text())
node = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "DiskEngramEmbedding")
methods = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in
           ("_read_rows", "close", "selected_bytes", "make_resident")]
native = {}
exec(compile(ast.Module(body=methods, type_ignores=[]), str(SOURCE), "exec"), native)


class Reader:
    def __init__(self, values):
        self.fixture_values, self.closed, self.calls = values, False, []
        self.entered = self.release = None
        self.fail_key = self.callback = None
        self.header = {k: {"shape": list(a.shape), "data_offsets": [0, a.nbytes]}
                       for k, (a, _) in values.items()}

    def read(self, key, rows=None, *, metal_backed=False):
        if self.closed:
            raise RuntimeError("fixture reader closed")
        if self.callback:
            self.callback()
        if self.entered is not None:
            self.entered.set()
            if not self.release.wait(5):
                raise TimeoutError("fixture read not released")
        if key == self.fail_key:
            raise OSError("fixture component read failed")
        a, dtype = self.fixture_values[key]
        rows = np.arange(a.shape[0]) if rows is None else np.asarray(rows, dtype=np.intp)
        self.calls.append((key, rows.copy()))
        if rows.size and (rows.min() < 0 or rows.max() >= a.shape[0]):
            raise IndexError("Engram row outside table")
        return a[rows].copy(), dtype

    def close(self):
        self.closed = True


class ModuleLike(dict):
    """Mimic Module's mapping-style parameter storage, without importing MLX."""
    def __setattr__(self, name, value):
        if isinstance(value, (np.ndarray, dict, list, tuple)):
            if hasattr(self, name) and name not in self:
                delattr(self, name)
            self[name] = value
        else:
            object.__setattr__(self, name, value)
            self.pop(name, None)

    def __getattr__(self, name):
        value = self.get(name, None)
        if value is not None:
            return value
        return object.__getattribute__(self, name)

    def __delattr__(self, name):
        if self.get(name, None) is not None:
            del self[name]
        else:
            object.__delattr__(self, name)


class Embedding(ModuleLike):
    _read_rows = native["_read_rows"]
    selected_bytes = native["selected_bytes"]
    make_resident = native["make_resident"]
    close = native["close"]

    def __init__(self, count=4096, components=3, salt=0):
        self._lock = threading.RLock()
        self._closed = False
        self._resident = self._prefetched = None
        rows = np.arange(count, dtype=np.uint32)
        self.fixture_values = {"weight": ((rows[:, None] * 31 + np.arange(4, dtype=np.uint32) + salt), "U32")}
        if components >= 2:
            self.fixture_values["scale"] = ((rows[:, None] * 13 + np.arange(2, dtype=np.uint32) + salt).astype(np.uint16), "BF16")
        if components >= 3:
            self.fixture_values["bias"] = ((rows[:, None] * 7 + np.arange(2, dtype=np.uint32) + salt).astype(np.float32), "F32")
        self._weights = self._scales = Reader(self.fixture_values)
        self._weight_key = "weight"
        self._scale_key = "scale" if components >= 2 else None
        self._bias_key = "bias" if components >= 3 else None

    def expected(self, ids):
        return [(a[ids.reshape(-1)].copy(), dtype) for a, dtype in self.fixture_values.values()]

    def consume(self, host):
        pending, self._prefetched = self._prefetched, None
        if pending is not None:
            requested, future = pending
            result = future.result()
            return result if np.array_equal(requested, host) else self._read_rows(host)
        return self._read_rows(host)


prefetch_node = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == "EngramPrefetch")
prefetch_globals = dict(ThreadPoolExecutor=ThreadPoolExecutor, contextmanager=contextmanager,
                        np=np, DiskEngramEmbedding=Embedding, PREFETCH_BYTES=64 * 2**20)
exec(compile(ast.Module(body=[prefetch_node], type_ignores=[]), str(SOURCE), "exec"), prefetch_globals)
Prefetch = prefetch_globals["EngramPrefetch"]


class Language(ModuleLike):
    def __init__(self, tables):
        self.layers = [SimpleNamespace(engram=SimpleNamespace(embed=e)) for e in tables]
        self._engram_prefetch = Prefetch()
        self.parameter = np.arange(8, dtype=np.float32)

    def __call__(self):
        return "native language"


class Model(ModuleLike):
    def __init__(self, components=3, tables=2):
        self.language_model = Language([Embedding(components=components, salt=i * 777) for i in range(tables)])
        self.close_failure = False
        self.closes = 0

    @property
    def tables(self):
        return [layer.engram.embed for layer in self.language_model.layers]

    @property
    def prefetch(self):
        return self.language_model._engram_prefetch

    def close(self):
        self.closes += 1
        if self.close_failure:
            raise OSError("fixture early offload close failure")
        try:
            self.prefetch.close()
        finally:
            for e in self.tables:
                e.close()


def same(actual, expected):
    assert len(actual) == len(expected)
    for (a, ad), (b, bd) in zip(actual, expected):
        assert ad == bd and a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes()


def spawn(function):
    result = {"started": threading.Event(), "done": threading.Event()}
    def run():
        result["started"].set()
        try:
            result["value"] = function()
        except BaseException as exc:
            result["error"] = exc
        finally:
            result["done"].set()
    thread = threading.Thread(target=run, daemon=True)
    result["thread"] = thread
    thread.start()
    assert result["started"].wait(2)
    return result


def finish(task):
    assert task["done"].wait(4), "thread did not finish (possible lifecycle deadlock)"
    task["thread"].join(1)
    if "error" in task:
        raise task["error"]
    return task.get("value")


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.models = []
        self.controllers = []

    def tearDown(self):
        for controller in self.controllers:
            controller.close()
        for model in self.models:
            model.close_failure = False
            model.close()
        self.assertIs(Embedding._read_rows, native["_read_rows"])
        self.assertIs(Embedding.close, native["close"])
        self.assertEqual(prefetch_globals["PREFETCH_BYTES"], 64 * 2**20)
        self.assertNotIn("mlx", sys.modules)
        self.assertNotIn("mlx.core", sys.modules)

    def make(self, *, components=3, budget=4096, enabled=True):
        model = Model(components=components)
        self.models.append(model)
        controller = install(model, embedding_type=Embedding, enabled=enabled, budget_per_table=budget)
        self.controllers.append(controller)
        return model, controller

    def test_exact_bytes_components_duplicates_order_empty(self):
        for components in (1, 2, 3):
            model, controller = self.make(components=components)
            e = model.tables[0]
            for ids in (np.array([[7, 1, 7], [3, 1, 3]]),
                        np.array([[3, 7, 1], [7, 1, 1]]), np.empty((0, 3), dtype=np.int64)):
                same(e._read_rows(ids), e.expected(ids))
            self.assertEqual(e._weights.calls[0][1].tolist(), [1, 3, 7])
            self.assertGreater(controller.snapshot()["live_tables"][0]["unique_hits"], 0)
            self.assertLessEqual(controller.snapshot()["retained_bytes"], 2 * 4096)

    def test_random_pressure_decay_and_collisions(self):
        model, c = self.make(budget=512)
        e = model.tables[0]
        same(e._read_rows(np.array([1])), e.expected(np.array([1])))
        state = c._states[0]
        capacity = len(state.cache.tags)
        _, saved = state.cache.lookup(np.array([1]))
        retained = [(a.copy(), d) for a, d in saved]
        other = 1 + capacity
        self.assertFalse(state.cache.lookup(np.array([other]))[0][0])
        for _ in range(5):
            ids = np.full(40, other)
            same(e._read_rows(ids), e.expected(ids))
        self.assertEqual(int(state.cache.tags[1 % capacity]), other)
        same(saved, retained)
        rng = np.random.default_rng(1017)
        for _ in range(260):
            ids = rng.integers(0, 4096, (5, 17))
            same(e._read_rows(ids), e.expected(ids))
            self.assertLessEqual(c.snapshot()["retained_bytes"], 1024)
        self.assertGreaterEqual(state.cache.calls, 256)

    def test_bounds_and_component_error_do_not_admit(self):
        model, c = self.make()
        e = model.tables[0]
        e._read_rows(np.array([1, 2]))
        before = c._states[0].cache.admitted
        for ids in ([-1], [4096], [1, -1], [2**64 - 1]):
            with self.assertRaises(IndexError):
                e._read_rows(np.array(ids, dtype=np.uint64 if ids[0] > 2**63 else np.int64))
        e._weights.fail_key = "scale"
        with self.assertRaises(OSError):
            e._read_rows(np.array([1, 900]))
        self.assertEqual(c._states[0].cache.admitted, before)
        e._weights.fail_key = None
        ids = np.array([900, 1, 900])
        same(e._read_rows(ids), e.expected(ids))

    def test_disabled_keeps_history_and_native_duplicate_reads(self):
        model, c = self.make()
        e = model.tables[0]
        ids = np.array([7, 1, 7, 2, 2])
        e._read_rows(ids)
        first = c.snapshot()
        c.set_enabled(False)
        e._read_rows(ids)
        after = c.snapshot()
        self.assertEqual(first["retained_bytes"], after["retained_bytes"])
        self.assertEqual(first["live_tables"][0]["unique_queries"], after["live_tables"][0]["unique_queries"])
        self.assertEqual(e._weights.calls[-1][1].tolist(), ids.tolist())
        c.set_enabled(True)
        same(e._read_rows(ids), e.expected(ids))
        self.assertEqual(c.snapshot()["live_tables"][0]["counts"]["all_hit_calls"], 1)
        c.set_enabled(False, clear=True)
        self.assertEqual(c.snapshot()["retained_bytes"], 0)

    def test_invalid_budget_bool_table_count_and_default_native_type(self):
        model = Model()
        self.models.append(model)
        for bad in (-1, MAX_TABLE_BYTES + 1, True, 1.0, "2"):
            with self.assertRaises(ValueError):
                install(model, embedding_type=Embedding, budget_per_table=bad)
        with self.assertRaises(TypeError):
            install(model, embedding_type=Embedding, enabled=1)
        with self.assertRaises(TypeError):
            install(model)
        for n in (0, 1, 3):
            wrong = Model(tables=n)
            self.models.append(wrong)
            with self.assertRaises(ValueError):
                install(wrong, embedding_type=Embedding)
        for budget in (0, 1, 31, 64):
            m, c = self.make(budget=budget)
            ids = np.array([1, 2, 1])
            same(m.tables[0]._read_rows(ids), m.tables[0].expected(ids))
            self.assertLessEqual(c.snapshot()["retained_bytes"], 2 * budget)

    def test_allocation_failure_releases_and_disables_table(self):
        model, c = self.make()
        e = model.tables[0]
        e._read_rows(np.array([1]))
        state = c._states[0]
        def fail(*args):
            raise MemoryError("fixture cache allocation")
        state.cache.remember = fail
        ids = np.array([1, 2, 1])
        same(e._read_rows(ids), e.expected(ids))
        self.assertEqual(c.snapshot()["retained_bytes"], 0)
        self.assertTrue(state.disabled)
        same(e._read_rows(ids), e.expected(ids))

    def test_noninteger_native_fallback_and_resident_transition(self):
        model, c = self.make()
        e = model.tables[0]
        ids = np.array([1., 2.])
        same(e._read_rows(ids), native["_read_rows"](e, ids))
        e._read_rows(np.array([1, 2]))
        self.assertGreater(c.snapshot()["retained_bytes"], 0)
        e.make_resident()
        self.assertEqual(c.snapshot()["retained_bytes"], 0)
        ids = np.array([2, 1, 2])
        same(e._read_rows(ids), e.expected(ids))
        self.assertTrue(c.snapshot()["live_tables"][0]["resident"])

    def test_native_prefetch_consume_and_mismatch(self):
        model, c = self.make()
        e = model.tables[0]
        ids = np.array([3, 7, 3])
        with model.prefetch.forward():
            model.prefetch.submit(e, ids)
            same(e.consume(ids), e.expected(ids))
            model.prefetch.submit(e, ids)
            other = np.array([7, 2])
            same(e.consume(other), e.expected(other))
        self.assertIsNone(model.prefetch._pending)
        self.assertIsNone(e._prefetched)

    def test_forward_drain_waits_worker_without_outer_lock_deadlock(self):
        model, c = self.make()
        e, p = model.tables[0], model.prefetch
        e._weights.entered, e._weights.release = threading.Event(), threading.Event()
        at_finally = threading.Event()
        def forward():
            with p.forward():
                p.submit(e, np.array([3, 4]))
                self.assertTrue(e._weights.entered.wait(2))
                at_finally.set()
        task = spawn(forward)
        try:
            self.assertTrue(at_finally.wait(2))
            # Native drain clears pending before awaiting the still-running read.
            deadline = time.monotonic() + 2
            while p._pending is not None and time.monotonic() < deadline:
                time.sleep(.001)
            self.assertIsNone(p._pending)
            self.assertFalse(task["done"].is_set())
            transition = spawn(lambda: c.set_enabled(False))
            self.assertFalse(transition["done"].wait(.03))
        finally:
            e._weights.release.set()
        finish(task)
        finish(transition)
        self.assertFalse(c.enabled)
        self.assertGreater(c.snapshot()["retained_bytes"], 0)

    def test_detach_during_direct_drain_pending_none_gap(self):
        model, c = self.make()
        e, p = model.tables[0], model.prefetch
        e._weights.entered, e._weights.release = threading.Event(), threading.Event()
        p.submit(e, np.array([3]))
        self.assertTrue(e._weights.entered.wait(2))
        draining = spawn(p.drain)
        try:
            deadline = time.monotonic() + 2
            while p._pending is not None and time.monotonic() < deadline:
                time.sleep(.001)
            self.assertIsNone(p._pending)
            detach = spawn(c.close)
            self.assertFalse(detach["done"].wait(.03))
        finally:
            e._weights.release.set()
        finish(draining)
        finish(detach)
        self.assertEqual(c.snapshot()["status"], "detached")
        self.assertEqual(c.snapshot()["retained_bytes"], 0)

    def test_cancelled_forward_exception_drains_and_recovers(self):
        model, c = self.make()
        e = model.tables[0]
        with self.assertRaisesRegex(RuntimeError, "request cancelled"):
            with model.prefetch.forward():
                model.prefetch.submit(e, np.array([1, 2]))
                raise RuntimeError("request cancelled")
        self.assertIsNone(model.prefetch._pending)
        self.assertIsNone(e._prefetched)
        with model.prefetch.forward():
            ids = np.array([8, 3, 8])
            model.prefetch.submit(e, ids)
            same(e.consume(ids), e.expected(ids))
        c.set_enabled(False)
        c.set_enabled(True)

    def test_control_inside_forward_rejected_and_recovery(self):
        model, c = self.make()
        with model.prefetch.forward():
            for action in (c.close, lambda: c.set_enabled(False), model.close,
                           model.prefetch.close, model.tables[0].close,
                           model.tables[0].make_resident):
                with self.assertRaisesRegex(RuntimeError, "drained forward"):
                    action()
            with self.assertRaisesRegex(RuntimeError, "Nested"):
                with model.prefetch.forward():
                    pass
        c.close()

    def test_read_callback_controls_rejected_instead_of_deadlock(self):
        model, c = self.make()
        e = model.tables[0]
        for action in (c.close, c.snapshot, lambda: c.set_enabled(False), model.close):
            e._weights.callback = action
            with self.assertRaisesRegex(RuntimeError, "row-read callback"):
                e._read_rows(np.array([234]))
        e._weights.callback = None
        same(e._read_rows(np.array([234])), e.expected(np.array([234])))

    def test_direct_read_close_waits_without_closing_live_reader(self):
        model, c = self.make()
        e = model.tables[0]
        e._weights.entered, e._weights.release = threading.Event(), threading.Event()
        task = spawn(lambda: e._read_rows(np.array([3])))
        self.assertTrue(e._weights.entered.wait(2))
        closing = spawn(e.close)
        try:
            self.assertFalse(closing["done"].wait(.03))
            self.assertFalse(e._weights.closed)
        finally:
            e._weights.release.set()
        same(finish(task), e.expected(np.array([3])))
        finish(closing)
        self.assertEqual(c.snapshot()["retained_bytes"], 0)
        with self.assertRaises(RuntimeError):
            e._read_rows(np.array([3]))

    def test_model_close_with_queued_read_and_early_native_error(self):
        model, c = self.make()
        e = model.tables[0]
        e._weights.entered, e._weights.release = threading.Event(), threading.Event()
        model.prefetch.submit(e, np.array([3]))
        self.assertTrue(e._weights.entered.wait(2))
        model.close_failure = True
        closing = spawn(model.close)
        try:
            self.assertFalse(closing["done"].wait(.03))
        finally:
            e._weights.release.set()
        with self.assertRaisesRegex(OSError, "early offload"):
            finish(closing)
        self.assertTrue(model.prefetch._closed)
        self.assertTrue(all(e._closed and e._weights.closed for e in model.tables))
        self.assertEqual(c.snapshot()["status"], "closed")
        self.assertEqual(c.snapshot()["retained_bytes"], 0)
        c.close()

    def test_prefetch_read_error_on_detach_restores_and_preserves_model(self):
        model, c = self.make()
        e = model.tables[0]
        e._weights.fail_key = "scale"
        model.prefetch.submit(e, np.array([1]))
        # Force the queued Future into running/completed state so cancel cannot
        # make this error-disposition test scheduler-dependent.
        try:
            model.prefetch._pending[1].result(2)
        except OSError:
            pass
        with self.assertRaises(OSError):
            c.close()
        self.assertEqual(c.snapshot()["status"], "detached")
        self.assertFalse(e._closed)
        self.assertNotIn("_read_rows", vars(e))
        e._weights.fail_key = None
        same(e._read_rows(np.array([1])), e.expected(np.array([1])))

    def test_two_models_isolated_no_global_or_parameter_tree_changes(self):
        a, ca = self.make()
        b, cb = self.make()
        ids = np.array([7, 1, 7])
        for e in a.tables + b.tables:
            same(e._read_rows(ids), e.expected(ids))
            self.assertNotIn("_read_rows", e)
            self.assertNotIn("_flash1017_serving_row_cache", e)
        ca.close()
        self.assertEqual(ca.snapshot()["retained_bytes"], 0)
        self.assertGreater(cb.snapshot()["retained_bytes"], 0)
        self.assertNotIn("close", a)
        self.assertNotIn("_flash1017_serving_row_cache", a)
        same(b.tables[0]._read_rows(ids), b.tables[0].expected(ids))

    def test_existing_instance_method_restored_and_stale_reader_native(self):
        model = Model()
        self.models.append(model)
        e = model.tables[0]
        original = e._read_rows
        calls = []
        def prior(ids):
            calls.append(1)
            return original(ids)
        object.__setattr__(e, "_read_rows", prior)
        c = install(model, embedding_type=Embedding, budget_per_table=4096)
        self.controllers.append(c)
        stale = e._read_rows
        stale(np.array([1]))
        c.close()
        self.assertIs(e._read_rows, prior)
        same(stale(np.array([2])), e.expected(np.array([2])))
        self.assertEqual(c.snapshot()["retained_bytes"], 0)
        self.assertEqual(len(calls), 2)

    def test_double_attach_rejected_and_reattach_after_close(self):
        model, c = self.make()
        with self.assertRaises(RuntimeError):
            install(model, embedding_type=Embedding)
        c.close()
        fresh = install(model, embedding_type=Embedding, budget_per_table=1024)
        self.controllers.append(fresh)
        self.assertEqual(fresh.snapshot()["retained_bytes"], 0)
        model.close()
        fresh.close()

    def test_loader_inference_shutdown_threads_can_migrate(self):
        model, c = self.make()
        e = model.tables[0]
        def work():
            with model.prefetch.forward():
                model.prefetch.submit(e, np.array([8, 4]))
                same(e.consume(np.array([8, 4])), e.expected(np.array([8, 4])))
        finish(spawn(work))
        finish(spawn(lambda: c.set_enabled(False)))
        finish(spawn(work))
        finish(spawn(c.close))
        self.assertEqual(c.snapshot()["status"], "detached")

    def test_language_dynamic_subclass_lifo_unchanged(self):
        model = Model()
        self.models.append(model)
        language = model.language_model
        original_type, parameter = type(language), language.parameter
        candidate_type = type("FixtureConeLanguage", (original_type,),
                              {"__call__": lambda self: "cone language"})
        object.__setattr__(language, "__class__", candidate_type)
        c = install(model, embedding_type=Embedding, budget_per_table=4096)
        self.controllers.append(c)
        self.assertEqual(language(), "cone language")
        self.assertIs(language.parameter, parameter)
        c.close()
        self.assertIs(type(language), candidate_type)
        self.assertEqual(language(), "cone language")
        object.__setattr__(language, "__class__", original_type)
        self.assertEqual(language(), "native language")
        self.assertIs(language.parameter, parameter)

    def test_close_drops_model_references_even_when_controller_retained(self):
        model = Model()
        c = install(model, embedding_type=Embedding, budget_per_table=4096)
        ref = weakref.ref(model)
        eref = weakref.ref(model.tables[0])
        model.tables[0]._read_rows(np.array([1]))
        model.close()
        del model
        gc.collect()
        self.assertIsNone(ref())
        self.assertIsNone(eref())
        self.assertEqual(c.snapshot()["status"], "closed")
        self.assertEqual(c.snapshot()["retained_bytes"], 0)

    def test_detach_reports_external_mutation_without_overwriting_it(self):
        model, c = self.make()
        e = model.tables[0]
        original = native["_read_rows"]
        replacement = lambda ids: original(e, ids)
        object.__setattr__(e, "_read_rows", replacement)
        with self.assertRaisesRegex(RuntimeError, "Owned instance attribute"):
            c.close()
        self.assertIs(e._read_rows, replacement)
        self.assertEqual(c.snapshot()["retained_bytes"], 0)
        self.assertNotIn("close", vars(model))

    def test_source_pin_and_no_mlx_import(self):
        hashes = dependency_fingerprints()
        self.assertEqual(hashes[str(Path(__file__).with_name("hot_rows_upstream.py"))], UPSTREAM_SHA256)
        self.assertEqual(len(hashes), 4)
        self.assertNotIn("mlx.core", sys.modules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
