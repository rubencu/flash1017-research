"""Control-flow and ownership fixtures; no MLX import or numerical claim."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import gc
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest
import weakref

import cone_serving as candidate


class Tensor:
    def __init__(self, shape, dtype="bf16"):
        self.shape, self.ndim, self.dtype = tuple(shape), len(shape), dtype
    def __getitem__(self, key):
        shape = list(self.shape)
        for axis, part in enumerate(key if isinstance(key, tuple) else (key,)):
            if isinstance(part, slice):
                shape[axis] = len(range(*part.indices(shape[axis])))
        return Tensor(shape, self.dtype)


class Module(dict):
    """Match MLX Module's Python dict-subclass layout, not its computation."""
    pass


class Projection(Module):
    def __init__(self, *, quantize_input=True):
        self.quantize_input, self.mode, self.bits, self.group_size = quantize_input, "mxfp8", 8, 32
        self.input_dims = 5120
        self.weight = Tensor((512, 1280))
        self.calls = []
        self.fail = False
    def project_quantized(self, x, indices=None, sorted_indices=False, block_plan=None):
        self.calls.append(x.shape[1])
        if self.fail:
            raise ValueError("projection failure")
        return Tensor((*x.shape[:2], 512), x.dtype)
    def __call__(self, x, indices=None, sorted_indices=False, block_plan=None):
        return self.project_quantized(x, indices, sorted_indices, block_plan)


def hc_mixes(*args):
    return "native-hc"


class Gate(Module):
    def __init__(self, config):
        self._config = config
        self.rows = []
    def __call__(self, x, image_mask):
        self.rows.append(x.shape[1])
        return Tensor((*x.shape[:2], 6)), Tensor((*x.shape[:2], 6))


class MoE(Module):
    def __init__(self, config):
        self.gate = Gate(config)
    def __call__(self, x, image_mask):
        self.gate(x, image_mask)
        return x


class Block(Module):
    def __init__(self, config, i):
        self._config, self.layer = config, i
        self.attn = SimpleNamespace(wkv=Projection(), indexer=SimpleNamespace(weights_proj=Projection(quantize_input=False)))
        self.ffn = MoE(config)
    def __call__(self, h, pre, cache, shared, start, image_mask, ced_tail=None):
        hc_mixes(h, Tensor((24, 20480), "f32"), None, None, self._config)
        if ced_tail is not None:
            h, pre = h[:, -ced_tail:], pre[:, -ced_tail:]
        self.attn.wkv(h)
        if self.layer in (20, 24, 28, 32, 36):
            self.attn.indexer.weights_proj(h)
        return self.ffn(h, image_mask), pre


class Cache:
    def __init__(self, start=0):
        self.offset, self.left_padding, self.lengths = start, None, None
    def size(self):
        return self.offset


class LanguageModel(Module):
    def __init__(self):
        self._config = SimpleNamespace(
            n_layers=40, window_size=128, kv_source_layers=(2,8,14,20),
            index_source_layers=(2,8,14,20,24,28,32,36), engram_layer_ids=(1,14),
            dspark_target_layer_ids=(37,38,39), candidate_source_layer=20,
            candidate_block_size=8, candidate_topk_blocks=2048, ced_prefill=False,
            ced_layout_supported=lambda: True, n_routed_experts=384, n_activated_experts=6,
            hc_mult=4)
        self.layers = [Block(self._config, i) for i in range(40)]
        self.mtp = [SimpleNamespace(attn=SimpleNamespace(wkv=Projection())) for _ in range(3)]
        self._omlx_dspark_decode_enabled = True
        self.calls, self.appends = [], []
        self.fail = self.swallow = self.omit_prime = self.reenter = False
        self.enter_event = self.exit_event = None
    def _omlx_prefill(self, input_ids, cache=None, **kwargs):
        return self(input_ids, cache=cache, _ced_prefill=True, **kwargs)
    def __call__(self, input_ids, cache=None, inputs_embeds=None, token_types=None, **kwargs):
        if self.enter_event:
            self.enter_event.set()
            assert self.exit_event.wait(2)
        if self.reenter:
            return self(input_ids, cache=cache)
        if self.fail:
            raise ValueError("forward failure")
        cache = cache or [Cache() for _ in range(40)]
        n, start = input_ids.shape[1], cache[0].size()
        self.calls.append((n, self._config.ced_prefill, inputs_embeds, token_types, dict(kwargs)))
        h = pre = Tensor((1, n, 5120))
        shared = {}
        for i, block in enumerate(self.layers):
            ced = self._config.ced_prefill and kwargs.get("_ced_prefill", False) and i >= 20
            absolute_start = start + n - 128 if ced and i > 20 else start
            h, pre = block(h, pre, cache[i], shared, absolute_start, None, ced_tail=128 if ced else None)
            cache[i].offset = start + n
        if self._omlx_dspark_decode_enabled and not self.omit_prime:
            try:
                self.dspark_append_context(Tensor((1,h.shape[1],15360)), [Cache() for _ in range(3)],
                                           start_offset=start+n-h.shape[1])
            except Exception:
                if not self.swallow:
                    raise
        return Tensor((1,h.shape[1],129280))
    def dspark_append_context(self, hidden, cache, *, start_offset=None):
        self.appends.append((hidden.shape[1], start_offset))
        for stage in self.mtp:
            out = stage.attn.wkv(Tensor((1,hidden.shape[1],5120)))
            assert out.shape[1] == hidden.shape[1]
        return "native-append"
    def forward_spec(self, ids, main_hidden, cache, **kwargs):
        return self.mtp[0].attn.wkv(main_hidden)
    def configure_mtp(self, enabled, depth=5):
        self._omlx_dspark_decode_enabled = bool(enabled)


class Adapter:
    def __init__(self, model):
        self.model = model
    def _omlx_prefill(self, ids, cache=None, **kwargs):
        # Mirrors VLMModelAdapter, which bypasses inner _omlx_prefill.
        return self.model(ids, cache=cache, _ced_prefill=True, **kwargs)


def bindings():
    def concat(arrays, axis=1):
        shape = list(arrays[0].shape)
        shape[axis] = sum(x.shape[axis] for x in arrays)
        return Tensor(shape, arrays[0].dtype)
    return SimpleNamespace(
        mx=SimpleNamespace(concatenate=concat, zeros=lambda shape,dtype: Tensor(shape,dtype),
                           bfloat16="bf16", float32="f32"),
        language=SimpleNamespace(LanguageModel=LanguageModel, Block=Block, Gate=Gate, MoE=MoE, hc_mixes=hc_mixes),
        hyper_connection=SimpleNamespace(), QuantizedProjection=Projection)


def cache(start=0):
    return [Cache(start) for _ in range(40)]


class Contracts(unittest.TestCase):
    def install(self, model=None, **kwargs):
        model = model or LanguageModel()
        owner = candidate.install(model, bindings=bindings(), record_events=True, **kwargs)
        self.addCleanup(owner.close)
        return model, owner

    def test_import_provenance_no_mlx_and_source_pins(self):
        self.assertNotIn("mlx.core", sys.modules)
        self.assertEqual(len(candidate.provenance()), 3)

    def test_direct_and_adapter_prefill_rows_endpoints(self):
        model, owner = self.install()
        for entry, n, start in ((model._omlx_prefill,8192,0), (Adapter(model)._omlx_prefill,8191,24576)):
            result = entry(Tensor((1,n)), cache=cache(start))
            self.assertEqual(result.shape, (1,128,129280))
            trace = owner.snapshot()["trace"][-1]
            self.assertEqual(trace["cone"][0]["end"], start+n)
            event = trace["prime"][0]
            self.assertTrue(event["append_complete"])
            self.assertEqual(event["append_start"], start+n-128)
            self.assertEqual([x["stage"] for x in event["stages"]], [0,1,2])
            for stage in model.mtp:
                self.assertEqual(stage.attn.wkv.calls[-1], n)
            self.assertFalse(model._config.ced_prefill)

    def test_disabled_native_toggle_and_ordinary_decode(self):
        model, owner = self.install(enabled=False)
        self.assertEqual(Adapter(model)._omlx_prefill(Tensor((1,8192)),cache=cache()).shape[1],8192)
        owner.set_enabled(True)
        self.assertEqual(model._omlx_prefill(Tensor((1,8192)),cache=cache()).shape[1],128)
        self.assertEqual(model(Tensor((1,3)),cache=cache()).shape[1],3)
        self.assertEqual(model.mtp[-1].attn.wkv.calls[-1],3)
        self.assertEqual(model.forward_spec(None,Tensor((1,5,5120)),None).shape[1],5)

    def test_short_vision_unknown_missing_cache_fallbacks(self):
        model, owner = self.install()
        for n, kwargs, c in ((2667,{},cache()), (8192,{"inputs_embeds":Tensor((1,8192,5120))},cache()),
                             (8192,{"token_types":Tensor((1,8192))},cache()),
                             (8192,{"new_option":True},cache()),(8192,{},None)):
            self.assertEqual(model._omlx_prefill(Tensor((1,n)), cache=c, **kwargs).shape[1],n)
        self.assertEqual(owner.snapshot()["cone_counts"].get("cone_calls",0),0)

    def test_sequential_thread_migration(self):
        model, owner = self.install()
        with ThreadPoolExecutor(1) as a, ThreadPoolExecutor(1) as b:
            ids = {a.submit(threading.get_ident).result(),b.submit(threading.get_ident).result()}
            self.assertEqual(len(ids),2)
            for worker in (a,b,a,b):
                self.assertEqual(worker.submit(model._omlx_prefill,Tensor((1,8192)),cache=cache()).result().shape[1],128)

    def test_overlap_and_mutation_rejected_before_native_call(self):
        model, owner = self.install()
        model.enter_event, model.exit_event = threading.Event(), threading.Event()
        with ThreadPoolExecutor(1) as worker:
            future = worker.submit(model._omlx_prefill,Tensor((1,8192)),cache=cache())
            self.assertTrue(model.enter_event.wait(2))
            for action in (lambda:model(Tensor((1,1)),cache=cache()),lambda:owner.set_enabled(False), owner.close, owner.snapshot,
                           lambda:model.forward_spec(None,Tensor((1,2,5120)),None)):
                with self.assertRaises(RuntimeError): action()
            model.exit_event.set()
            self.assertEqual(future.result().shape[1],128)

    def test_reentrant_call_failure_cleanup_and_recovery(self):
        model, owner = self.install()
        model.reenter = True
        with self.assertRaises(RuntimeError): model._omlx_prefill(Tensor((1,8192)),cache=cache())
        model.reenter = False
        self.assertFalse(model._config.ced_prefill)
        self.assertEqual(model._omlx_prefill(Tensor((1,8192)),cache=cache()).shape[1],128)

    def test_swallowed_prime_error_and_missing_capture_fail_visible(self):
        model, owner = self.install()
        model.swallow, model.mtp[1].attn.wkv.fail = True, True
        with self.assertRaisesRegex(RuntimeError,"priming failed"):
            model._omlx_prefill(Tensor((1,8192)),cache=cache())
        model.mtp[1].attn.wkv.fail = False
        model.omit_prime = True
        with self.assertRaisesRegex(RuntimeError,"did not complete"):
            model._omlx_prefill(Tensor((1,8192)),cache=cache())
        model.omit_prime = False
        self.assertEqual(model._omlx_prefill(Tensor((1,8192)),cache=cache()).shape[1],128)

    def test_no_dspark_allowed_and_native_direct_append(self):
        model, owner = self.install()
        model.configure_mtp(False)
        self.assertEqual(model._omlx_prefill(Tensor((1,8192)),cache=cache()).shape[1],128)
        model.dspark_append_context(Tensor((1,128,15360)),[Cache() for _ in range(3)],start_offset=0)
        self.assertEqual(model.mtp[0].attn.wkv.calls[-1],128)

    def test_unrelated_model_and_class_globals_untouched(self):
        before = (LanguageModel.__call__, Block.__call__, Gate.__call__, MoE.__call__, Projection.__call__, Projection.project_quantized, hc_mixes)
        first, owner = self.install()
        second, other = self.install()
        unrelated = LanguageModel()
        self.assertEqual(unrelated._omlx_prefill(Tensor((1,8192)),cache=cache()).shape[1],8192)
        with ThreadPoolExecutor(2) as pool:
            futures=[pool.submit(m._omlx_prefill,Tensor((1,8192)),cache=cache()) for m in (first,second)]
            self.assertEqual([f.result().shape[1] for f in futures],[128,128])
        after = (LanguageModel.__call__, Block.__call__, Gate.__call__, MoE.__call__, Projection.__call__, Projection.project_quantized, hc_mixes)
        self.assertEqual(before,after)
        self.assertNotIn("_serving_cone_v3_owner",first)

    def test_class_restore_parameter_identity_and_bounded_trace(self):
        model, owner = self.install()
        weight = model.layers[20].attn.wkv.weight
        for _ in range(20): model(Tensor((1,1)),cache=cache())
        self.assertEqual(len(owner.snapshot()["trace"]),16)
        owner.snapshot(clear=True)
        self.assertEqual(owner.snapshot()["trace"],[])
        owner.close()
        self.assertIs(type(model),LanguageModel)
        self.assertIs(type(model.layers[20]),Block)
        self.assertIs(type(model.layers[20].attn.wkv),Projection)
        self.assertIs(weight,model.layers[20].attn.wkv.weight)
        self.assertFalse(hasattr(model,"_serving_cone_v3_owner"))
        self.assertEqual(model._omlx_prefill(Tensor((1,8192)),cache=cache()).shape[1],8192)

    def test_layout_guard_duplicate_and_native_ced_guard(self):
        model, owner = self.install()
        with self.assertRaises(RuntimeError): candidate.install(model,bindings=bindings())
        model._config.ced_prefill = True
        with self.assertRaises(RuntimeError): model(Tensor((1,1)),cache=cache())
        model._config.ced_prefill = False
        invalid=LanguageModel()
        invalid._config.window_size=64
        with self.assertRaises(ValueError):candidate.install(invalid,bindings=bindings())

    def test_foreign_class_refuses_without_partial_teardown(self):
        model, owner = self.install()
        installed = type(model)
        foreign = type("Foreign",(installed,),{"__slots__":()})
        object.__setattr__(model,"__class__",foreign)
        before = [type(obj) for obj,_,_ in owner._saved]
        with self.assertRaisesRegex(RuntimeError,"class changed"): owner.close()
        self.assertEqual(before,[type(obj) for obj,_,_ in owner._saved])
        self.assertFalse(owner.closed)
        object.__setattr__(model,"__class__",installed)
        owner.close()

    def test_strict_bool_options(self):
        model, owner = self.install()
        for invalid in (0,1,None,"false"):
            with self.assertRaises(TypeError):owner.set_enabled(invalid)
            with self.assertRaises(TypeError):candidate.install(LanguageModel(),enabled=invalid,bindings=bindings())

    def test_failure_cleanup_preserves_request_cache_boundary(self):
        model, owner = self.install()
        model.fail = True
        failed_cache=cache(1000)
        with self.assertRaises(ValueError):model._omlx_prefill(Tensor((1,8192)),cache=failed_cache)
        self.assertEqual(failed_cache[0].size(),1000)
        self.assertIsNone(owner.prime.frame)
        self.assertIsNone(owner.shape.frame)
        self.assertIsNone(owner.cone.frame)
        self.assertFalse(model._config.ced_prefill)
        model.fail = False
        self.assertEqual(model._omlx_prefill(Tensor((1,8192)),cache=cache(29000)).shape[1],128)

    def test_batched_padded_limited_capture_verify_fallback(self):
        model, owner = self.install()
        padded=cache()
        padded[0].left_padding=SimpleNamespace(reshape=lambda _:SimpleNamespace(tolist=lambda:[1]))
        limited=cache()
        limited[0].lengths=Tensor((1,))
        for ids,c,kw in ((Tensor((2,8192)),cache(),{}), (Tensor((1,8192)),padded,{}),
                         (Tensor((1,8192)),limited,{}),
                         (Tensor((1,8192)),cache(),{"return_hidden":True}),
                         (Tensor((1,8192)),cache(),{"return_dspark_hidden":True}),
                         (Tensor((1,8192)),cache(),{"target_verify":True})):
            self.assertEqual(model._omlx_prefill(ids,cache=c,**kw).shape[1],8192)
        self.assertEqual(owner.cone.counts["cone_calls"],0)

    def test_close_releases_model_and_keeps_plain_diagnostics(self):
        model, owner = self.install()
        model._omlx_prefill(Tensor((1,8192)),cache=cache())
        reference = weakref.ref(model)
        owner.close()
        del model
        gc.collect()
        self.assertIsNone(reference())
        summary = owner.snapshot()
        self.assertTrue(summary["closed"])
        self.assertTrue(summary["model_references_released"])
        self.assertEqual(summary["prime_counts"]["padded_appends"],1)
        self.assertEqual(summary["owned_instances"],0)

    def test_prefill_trace_survives_decode_and_reports_eviction(self):
        model, owner = self.install()
        for _ in range(17): model._omlx_prefill(Tensor((1,2668)),cache=cache())
        for _ in range(20): model(Tensor((1,1)),cache=cache())
        summary=owner.snapshot()
        self.assertEqual(len(summary["prefill_trace"]),16)
        self.assertEqual(summary["counts"]["prefill_trace_evictions"],1)
        self.assertTrue(all(t["prime"][0]["append_complete"] for t in summary["prefill_trace"]))
        owner.snapshot(clear=True)
        self.assertEqual(owner.snapshot()["prefill_trace"],[])


if __name__ == "__main__":
    unittest.main()
