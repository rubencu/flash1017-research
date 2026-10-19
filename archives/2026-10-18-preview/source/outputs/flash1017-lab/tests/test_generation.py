"""Real tiny-weight DSpark execution, target verification and CSA2 rollback.

Shape/weight fixture adapted from oMLX's MIT-licensed V4.1 regression suite;
this exercises the installed runtime, not a mocked verification algorithm.
"""
import zlib

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten

from flash1017.policy import Experiment, PrefixPolicy
from omlx.patches.deepseek_v41.config import ModelConfig
from omlx.patches.deepseek_v41.language import LanguageModel


def tiny_model(attention_rounding=False):
    config = ModelConfig(
        vocab_size=64, dim=32, moe_inter_dim=32, n_layers=5, n_heads=2,
        q_lora_rank=32, head_dim=32, rope_head_dim=8, o_groups=2,
        o_lora_rank=16, index_n_heads=8, index_head_dim=32, index_topk=3,
        n_routed_experts=4, n_activated_experts=2, window_size=4,
        dtype="bf16", expert_dtype=None, max_seq_len=64, max_batch_size=2,
        hc_sinkhorn_iters=3, candidate_source_layer=3, candidate_block_size=2,
        candidate_topk_blocks=2, index_source_layers=(1, 3, 4),
        preserve_mtp=True, n_mtp_layers=3, dspark_block_size=3,
        dspark_noise_token_id=2, dspark_target_layer_ids=(2, 3, 4),
        dspark_n_routed_experts=2, dspark_n_activated_experts=1,
        dspark_markov_rank=32, compress_ratios=(0, 2, 2, 1, 1, 0, 0, 0),
        attention_output_quantization=attention_rounding,
    )
    model = LanguageModel(config)

    def values(name, shape):
        rng = np.random.default_rng(341 + zlib.crc32(name.encode()))
        data = rng.normal(0, .08, shape).astype(np.float32)
        if "norm.weight" in name:
            data += 1
        if "weights_proj.weight" in name:
            data = np.abs(data)
        return data

    weights = []
    for name, parameter in tree_flatten(model.parameters()):
        if ".experts." in name and parameter.ndim == 3:
            prefix, suffix = name.split(".experts.")
            data = np.stack([values(f"{prefix}.experts.{i}.{suffix}", parameter.shape[1:])
                             for i in range(parameter.shape[0])])
        else:
            data = values(name, parameter.shape)
        weights.append((name, mx.array(data)))
    model.load_weights(weights, strict=True)
    return model


@pytest.mark.parametrize("selection", ["observe", "zero", "one", "three", "confidence"])
@pytest.mark.parametrize("attention_rounding", [False, True])
def test_real_generation_and_repeatability(selection, attention_rounding):
    from mlx_lm.generate import BatchGenerator
    from omlx.patches.mlx_lm_mtp import batch_generator

    batch_generator.apply()
    model = tiny_model(attention_rounding)
    prompt = [3, 4, 5, 6, 7, 8, 9]
    def generate():
        generator = BatchGenerator(model, max_tokens=24, prefill_step_size=3,
                                   completion_batch_size=1, prefill_batch_size=1,
                                   sampler=lambda logits: mx.argmax(logits, -1))
        result = []
        try:
            generator.insert([prompt])
            for _ in range(96):
                _, responses = generator.next()
                result.extend(response.token for response in responses)
                if len(result) == 24:
                    break
        finally:
            generator.close()
        return result

    # Match prefill shapes: activation rounding can amplify the difference
    # between a whole-prompt GEMM and the serving engine's chunked prefill.
    model.configure_mtp(False, 3)
    autoregressive = generate()
    model.configure_mtp(True, 3)
    stock_mtp = generate()
    selections = {
        "observe": Experiment(),
        "zero": Experiment(draft_width=3, forced_depth=0),
        "one": Experiment(draft_width=3, forced_depth=1),
        "three": Experiment(draft_width=3, forced_depth=3),
        "confidence": Experiment(draft_width=3, policy=PrefixPolicy((10, 11, 12, 13))),
    }
    experiment = selections[selection]
    model._flash1017_experiment = experiment
    model.configure_mtp(True, 3)
    result = generate()
    if selection == "zero":
        assert result == autoregressive
    if selection == "observe":
        assert result == stock_mtp, "Instrumentation alone must not change text"
    # GEMM/GEMV rounding can cross quantization thresholds and change greedy
    # text, with and without the reference attention correction. This fixture
    # exposes it. Do NOT claim bit identity across different verify shapes.
    # Each frozen policy must still be repeatable on the same execution path.
    assert generate() == result
    assert len(result) == 24 and all(0 <= token < 64 for token in result)
    assert experiment.events, "The instrumented verification path must execute"
    events = [event for event in experiment.events if not event["copied"]]
    assert all(event["confidence_logits"] for event in events)
    if experiment.forced_depth is not None:
        assert all(event["verified"] == experiment.forced_depth for event in events)
    assert all(0 <= event["accepted"] <= event["verified"] for event in events)
