#!/usr/bin/env python3
"""Fixed-input paired benchmark of unchanged main and the prefetch admission change."""
import argparse
import gc
import hashlib
import importlib.util
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--arm', choices=['baseline', 'candidate'], required=True)
    parser.add_argument('--block', type=int, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.runtime.resolve()))
    # Reuse only the established resource meter; no configure or run helpers.
    sys.path.append(str(ROOT / 'outputs/flash1017-lab'))
    import mlx.core as mx
    import psutil
    import omlx
    from flash1017.bench import Meter
    from mlx_lm.generate import BatchGenerator
    from omlx.model_settings import ModelSettings
    from omlx.utils.model_loading import maybe_apply_pre_load_patches
    from omlx.patches.deepseek_v41.loading import load
    from omlx.patches.deepseek_v41 import loading
    from omlx.patches.mlx_lm_mtp import batch_generator, set_mtp_active
    from omlx.utils.sampling import make_sampler

    assert Path(omlx.__file__).resolve().parent == args.runtime.resolve() / 'omlx'
    plan_bytes = (HERE / 'plan.json').read_bytes()
    plan = json.loads(plan_bytes)
    output = HERE / f'{args.arm}-{args.block}.jsonl'
    if output.exists():
        raise FileExistsError(output)
    def record(data):
        with output.open('a') as file:
            file.write(json.dumps(data, default=str) + '\n')
        print(json.dumps({k:v for k,v in data.items() if k in ('type','id','seconds','total_seconds','ttft_seconds','decode_tps','error')}, default=str), flush=True)

    if psutil.virtual_memory().available < 210 * 2**30:
        raise MemoryError('Less than 210 GiB available before loading')
    logging.basicConfig(level=logging.INFO)
    maybe_apply_pre_load_patches(plan['model'], ModelSettings(mtp_enabled=True, mtp_adaptive_max_depth=3, deepseek_v41_engram_ssd_offload=True), for_vlm=True)
    set_mtp_active(True)
    batch_generator.apply()
    device = mx.device_info()
    limit = min(device['max_recommended_working_set_size'], device['memory_size'] - 32 * 2**30)
    mx.set_wired_limit(limit)
    mx.set_memory_limit(limit)
    mx.set_cache_limit(8 * 2**30)
    original_load = loading._load_shard
    def guarded_load(path):
        if psutil.virtual_memory().available < 12 * 2**30:
            raise MemoryError('Insufficient headroom during loading')
        return original_load(path)
    loading._load_shard = guarded_load
    record({'type':'manifest','arm':args.arm,'block':args.block,'runtime':str(args.runtime.resolve()),'base':plan['base'],'plan_sha256':hashlib.sha256(plan_bytes).hexdigest(),'device':device})
    started = time.perf_counter()
    model, processor = load(plan['model'], engram_ssd_offload=True)
    record({'type':'load','seconds':time.perf_counter()-started})
    language = model.language_model
    language.configure_mtp(True, 3)
    language._omlx_mtp_depth_fixed = False
    from collections import Counter
    from threading import current_thread
    from omlx.patches.deepseek_v41 import storage
    assert storage.PREFETCH_BYTES == (16 if args.arm == 'baseline' else 64) * 2**20
    io_counts = Counter()
    submit = storage.EngramPrefetch.submit
    read = storage.DiskEngramEmbedding._read_rows
    def observe_submit(self, embed, ids):
        size = embed.selected_bytes(ids.size) + ids.size * 8
        result = submit(self, embed, ids)
        io_counts['submit_calls'] += 1
        io_counts['admitted' if self._pending is not None else 'rejected'] += 1
        io_counts['max_selected_plus_ids'] = max(io_counts['max_selected_plus_ids'], size)
        return result
    def observe_read(self, host):
        size = self.selected_bytes(host.size) + host.nbytes
        prefix = 'background' if current_thread().name.startswith('v41-engram') else 'foreground'
        io_counts[prefix + '_calls'] += 1
        io_counts[prefix + '_selected_plus_ids'] += size
        if size > 16 * 2**20: io_counts[prefix + '_over16_calls'] += 1
        return read(self, host)
    storage.EngramPrefetch.submit = observe_submit
    storage.DiskEngramEmbedding._read_rows = observe_read
    def run(prompt, count, warmup=False):
        tokens = prompt['tokens']
        step = prompt['prefill_step']
        io_counts.clear()
        gc.collect()
        mx.clear_cache()
        mx.synchronize()
        mx.reset_peak_memory()
        mx.random.seed(plan['seed'])
        generated, arrivals, stats = [], [], []
        log_stats = batch_generator._log_mtp_stats
        def capture(uid, value, reason):
            stats.append(asdict(value) | {'finish_reason':reason})
            log_stats(uid, value, reason)
        batch_generator._log_mtp_stats = capture
        cache = language.make_cache()
        generator = BatchGenerator(language, max_tokens=count, stop_tokens=None, sampler=make_sampler(temp=0.0, top_p=1.0, top_k=0, min_p=0), prefill_step_size=step, completion_batch_size=1, prefill_batch_size=1)
        meter = Meter().start()
        begun = time.perf_counter()
        try:
            with mx.stream(generator.stream):
                for start in range(0,len(tokens)-1,step):
                    chunk = tokens[start:min(start+step,len(tokens)-1)]
                    language._omlx_prefill(mx.array([chunk]), cache=cache, skip_lm_head=True)
                    mx.eval([item.state for item in cache])
            prefilled = time.perf_counter()
            generator.insert([tokens[-1:]], caches=[cache], all_tokens=[tokens[:-1]])
            while len(generated) < count:
                _, responses = generator.next()
                now = time.perf_counter()
                for response in responses:
                    generated.append(int(response.token)); arrivals.append(now-begun)
                if now-begun > 600:
                    raise TimeoutError('Per-request bound exceeded')
                if not responses and not len(generator._generation_batch) and not generator._unprocessed_sequences:
                    raise RuntimeError('Generation exhausted early')
            mx.synchronize(generator.stream)
            elapsed = time.perf_counter()-begun
            resources = meter.stop()
        finally:
            generator.close()
            batch_generator._log_mtp_stats = log_stats
            if not meter.done.is_set(): meter.stop()
        assert len(generated) == count and stats
        record({'type':'warmup' if warmup else 'run','id':prompt['id'],'input_tokens':len(tokens),'input_sha256':hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),'token_ids':generated,'total_seconds':elapsed,'prefill_seconds':prefilled-begun,'ttft_seconds':arrivals[0],'decode_tps':(count-1)/(arrivals[-1]-arrivals[0]),'prefill_step':step,'io':dict(io_counts),'resources':resources,'mtp_stats':stats})
    try:
        warm = {'id':'excluded-warmup','tokens':plan['prompts'][0]['tokens'][:plan['warmup_tokens']],'prefill_step':2048}
        run(warm,plan['warmup_decode'],True)
        prompts = plan['prompts'] if args.block == 0 else list(reversed(plan['prompts']))
        for prompt in prompts: run(prompt,plan['decode_tokens'])
    finally:
        model.close()
        record({'type':'closed','pending':language._engram_prefetch._pending is not None})


if __name__ == '__main__':
    main()
