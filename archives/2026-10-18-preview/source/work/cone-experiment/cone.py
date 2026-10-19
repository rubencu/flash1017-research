"""Scoped, single-thread experimental decoder dependency-cone prefill.

This is NOT the native approximate 128-row CED policy. A large cache-only
chunk retains W + (n_layers-layer)*(W-1) input rows at each decoder layer.
All encoder rows still produce midpoint global K/V and index keys. Only chunks
which alone contain the complete dependency cone are eligible; no extra hidden
state crosses a call or cache boundary. Ordinary scoring, explicit capture,
verification, short chunks, padding, and multimodal inputs use full-depth calls.

The dependency argument does not establish floating-point or token equality:
the row shapes change dispatch/reduction choices. This file is an isolated
prototype, does not edit runtime files, and must never be installed in serving
without separate numerical, cache, DSpark, and performance qualification.
"""

from collections import Counter
from contextlib import contextmanager
import threading

from omlx.patches.deepseek_v41 import language


_INSTALL_LOCK = threading.Lock()


def required_rows(config, layer):
    # Retain W valid final logits, not merely one. The smaller bound
    # W+(n_layers-1-layer)*(W-1) suffices for caches and one final logit.
    return int(config.window_size) + (int(config.n_layers) - layer) * (int(config.window_size) - 1)


class ConeExperiment:
    """One-model scoped shim; rejects concurrent/reentrant use of that model."""

    def __init__(self, model, *, allow_tiny_layout=False, record_events=True):
        if not isinstance(model, language.LanguageModel):
            raise TypeError("Pass the DeepSeek V4.1 language_model instance")
        self.model = model
        self.config = model._config
        self.allow_tiny_layout = bool(allow_tiny_layout)
        self.record_events = bool(record_events)
        self.events = []
        self.counts = Counter()
        self.owner_thread = None
        self.frame = None
        self.active = False
        self.mid = int(self.config.n_layers) // 2
        self.required = required_rows(self.config, self.mid)
        self.minimum_cache_cone = self.required - (int(self.config.window_size) - 1)
        self.blocks = {id(block): i for i, block in enumerate(model.layers)}
        self.layout_reason = self._layout_reason()

    def _layout_reason(self):
        c = self.config
        if not c.ced_layout_supported():
            return "unsupported-native-layout"
        if int(c.window_size) <= 1:
            return "degenerate-window"
        if len(self.model.layers) != c.n_layers or any(type(x) is not language.Block for x in self.model.layers):
            return "custom-block-implementation"
        if c.candidate_source_layer != self.mid:
            return "candidate-source-not-midpoint"
        if c.candidate_block_size <= 0 or c.candidate_topk_blocks <= 0:
            return "missing-candidate-blocks"
        if any(i < self.mid or i >= c.n_layers for i in c.dspark_target_layer_ids):
            return "unsupported-capture-layers"
        if not self.allow_tiny_layout:
            if not (
                c.n_layers == 40 and c.window_size == 128
                and tuple(c.kv_source_layers) == (2, 8, 14, 20)
                and tuple(c.index_source_layers) == (2, 8, 14, 20, 24, 28, 32, 36)
                and tuple(c.engram_layer_ids) == (1, 14)
                and tuple(c.dspark_target_layer_ids) == (37, 38, 39)
                and c.candidate_block_size == 8 and c.candidate_topk_blocks == 2048
            ):
                return "not-pinned-1017-layout"
        return None

    def _assert_owner(self):
        if threading.get_ident() != self.owner_thread:
            raise RuntimeError("Cone prototype permits only its installation thread")

    def _fallback_reason(self, input_ids, cache, inputs_embeds, token_types, kwargs):
        if self.layout_reason:
            return self.layout_reason
        if not kwargs.get("_ced_prefill", False):
            return "not-cache-only-prefill"
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            return "not-singleton-text"
        if input_ids.shape[1] < self.required:
            return "short-chunk"
        if inputs_embeds is not None or token_types is not None:
            return "multimodal-or-token-types"
        if kwargs.get("return_hidden") or kwargs.get("return_dspark_hidden"):
            return "explicit-full-hidden"
        if kwargs.get("target_verify") or kwargs.get("n_confirmed") or kwargs.get("mtp_verify_states") is not None:
            return "verification"
        allowed = {"_ced_prefill", "skip_lm_head", "return_hidden", "return_dspark_hidden",
                   "target_verify", "n_confirmed", "mtp_verify_states"}
        if any(k not in allowed for k in kwargs):
            return "unknown-forward-option"
        if cache is not None:
            if len(cache) != self.config.n_layers:
                return "cache-layout"
            for item in cache:
                # Natural singleton cache merges retain a zero left-padding
                # array. Permit exactly that case, but never infer validity
                # from an opaque batched/length-limited cache.
                padding = getattr(item, "left_padding", None)
                if padding is not None and any(int(x) != 0 for x in padding.reshape(-1).tolist()):
                    return "padded-cache"
                if getattr(item, "lengths", None) is not None:
                    return "length-limited-cache"
        return None

    def _call(self, instance, input_ids, cache=None, inputs_embeds=None, token_types=None, **kwargs):
        if instance is not self.model:
            return self._original_call(instance, input_ids, cache=cache,
                                       inputs_embeds=inputs_embeds, token_types=token_types, **kwargs)
        self._assert_owner()
        if self.frame is not None:
            raise RuntimeError("Reentrant model calls are unsupported by cone prototype")
        reason = self._fallback_reason(input_ids, cache, inputs_embeds, token_types, kwargs)
        previous = self.config.ced_prefill
        use_cone = reason is None
        self.config.ced_prefill = use_cone
        if use_cone:
            start = cache[0].size() if cache is not None else 0
            self.frame = {"start": start, "end": start + input_ids.shape[1],
                          "chunk_tokens": input_ids.shape[1], "next_layer": self.mid,
                          "layers": []}
            self.counts["cone_calls"] += 1
        else:
            self.counts["full_calls"] += 1
            self.counts["fallback:" + reason] += 1
        try:
            result = self._original_call(instance, input_ids, cache=cache,
                                         inputs_embeds=inputs_embeds, token_types=token_types, **kwargs)
            if use_cone and self.frame["next_layer"] != self.config.n_layers:
                raise RuntimeError("Cone forward did not visit every decoder layer")
            if self.record_events:
                event = {"mode": "cone" if use_cone else "full", "reason": reason,
                         "input_tokens": input_ids.shape[1]}
                if use_cone:
                    event.update(self.frame)
                self.events.append(event)
            return result
        finally:
            self.frame = None
            self.config.ced_prefill = previous

    def _block(self, block, h, pre, cache, shared, start, image_mask, ced_tail=None):
        layer = self.blocks.get(id(block))
        if layer is None:
            return self._original_block(block, h, pre, cache, shared, start, image_mask, ced_tail=ced_tail)
        self._assert_owner()
        if self.frame is None or layer < self.mid:
            if ced_tail is not None:
                # This cannot arise through the guarded LanguageModel call.
                # Refuse rather than accidentally execute approximate replay.
                raise RuntimeError("Unexpected CED block outside guarded cone forward")
            return self._original_block(block, h, pre, cache, shared, start, image_mask, ced_tail=None)
        frame = self.frame
        if ced_tail is None or layer != frame["next_layer"]:
            raise RuntimeError("Unexpected decoder call ordering or CED dispatch")
        # Native _forward passes the full chunk start at midpoint; for every
        # later decoder layer it assumes a fixed-W tail and passes N-W.
        endpoint = int(start) + (int(h.shape[1]) if layer == self.mid else int(self.config.window_size))
        if endpoint != frame["end"] or h.shape[:2] != pre.shape[:2]:
            raise RuntimeError("Cone absolute positions or mHC rows diverged")
        input_rows = int(h.shape[1])
        tail = min(required_rows(self.config, layer), input_rows)
        true_start = endpoint - input_rows
        if tail < required_rows(self.config, layer):
            raise RuntimeError("Insufficient decoder dependency rows")
        # These arrays carry *query* rows. Global KV/index keys must never be
        # cropped. At midpoint source queries are recomputed; later layers
        # reuse/recompute only the matching suffix of this query state.
        for name in ("idx", "candidates"):
            value = shared.get(name)
            if value is not None:
                if value.ndim != 3 or value.shape[0] != 1 or value.shape[1] < tail:
                    raise RuntimeError("Invalid shared query shape for " + name)
                shared[name] = value[:, -tail:]
        output = self._original_block(
            block, h, pre, cache, shared, true_start, image_mask, ced_tail=tail
        )
        if output[0].shape[1] != tail or output[1].shape[1] != tail:
            raise RuntimeError("Unexpected cone block output width")
        if self.record_events:
            frame["layers"].append({"layer": layer, "input_rows": input_rows,
                                    "query_rows": tail, "input_start": true_start,
                                    "query_start": endpoint-tail, "endpoint": endpoint})
        frame["next_layer"] += 1
        if layer == self.config.n_layers - 1:
            # This block's leading W-1 outputs lack earlier local inputs.
            # Only its last W rows are valid and part of the public cache-only
            # return contract. Crop before native norm/head and hidden-capture
            # concatenation, so DSpark priming also receives exactly W valid
            # target rows rather than a mixture of valid/boundary outputs.
            output = (output[0][:, -self.config.window_size:],
                      output[1][:, -self.config.window_size:])
        return output

    def __enter__(self):
        if self.active or not _INSTALL_LOCK.acquire(blocking=False):
            raise RuntimeError("Only one cone experiment may be installed at a time")
        try:
            self.owner_thread = threading.get_ident()
            self._had_own_call = "__call__" in language.LanguageModel.__dict__
            self._saved_own_call = language.LanguageModel.__dict__.get("__call__")
            self._original_call = language.LanguageModel.__call__
            self._original_block = language.Block.__call__
            self._original_ced = self.config.ced_prefill
            self.config.ced_prefill = False
            experiment = self
            def call(instance, *args, **kwargs):
                return experiment._call(instance, *args, **kwargs)
            def block(instance, *args, **kwargs):
                return experiment._block(instance, *args, **kwargs)
            self._call_shim, self._block_shim = call, block
            language.LanguageModel.__call__ = call
            language.Block.__call__ = block
            self.active = True
            return self
        except BaseException:
            _INSTALL_LOCK.release()
            raise

    def __exit__(self, exc_type, exc, tb):
        self._assert_owner()
        try:
            if self._had_own_call:
                language.LanguageModel.__call__ = self._saved_own_call
            else:
                del language.LanguageModel.__call__
            language.Block.__call__ = self._original_block
            self.config.ced_prefill = self._original_ced
            self.active = False
        finally:
            _INSTALL_LOCK.release()


@contextmanager
def installed(model, *, allow_tiny_layout=False, record_events=True):
    """Yield a scoped ConeExperiment and always restore original class methods."""
    with ConeExperiment(model, allow_tiny_layout=allow_tiny_layout,
                        record_events=record_events) as experiment:
        yield experiment
