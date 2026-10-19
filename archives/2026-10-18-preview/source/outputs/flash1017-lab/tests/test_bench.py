from types import SimpleNamespace

from flash1017.bench import run_one
from test_generation import tiny_model


def test_benchmark_uses_real_prefill_and_emission_with_resource_metrics():
    from omlx.patches.mlx_lm_mtp import batch_generator
    batch_generator.apply()
    language = tiny_model(True)
    model = SimpleNamespace(language_model=language, named_modules=language.named_modules)
    tokenizer = SimpleNamespace(eos_token_id=63, decode=lambda tokens, **kw: str(tokens))
    processor = SimpleNamespace(tokenizer=tokenizer)
    profile = {"name": "tiny", "attention_rounding": True, "mtp": True,
               "depth": 3, "instrument": True, "draft_width": 3,
               "forced_depth": 1, "prefill_step": 3}
    prompt = {"id": "tiny", "tokens": [3, 4, 5, 6, 7, 8, 9],
              "max_tokens": 16, "force_length": True}
    result = run_one(model, processor, prompt, profile, 0, 1017)
    assert result["generated_tokens"] == 16
    assert result["input_tokens"] == 7
    assert result["events"]
    assert 0 < result["prefill_seconds"] <= result["ttft_seconds"] <= result["total_seconds"]
    assert result["decode_tokens_per_second"] > 0
    assert result["resources"]["process_footprint_peak_bytes"] > 0
    assert all(event["verified"] == 1 for event in result["events"] if not event["copied"])
