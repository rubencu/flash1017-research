"""Initialize this Mac's large-model residency before loading any weights.

The ordinary oMLX server does this through its memory manager. Direct loader
users must initialize it too: MLX's default wired limit is zero. A per-shard
headroom check stops this lab's own load before consuming its OS reserve.
No system-wide settings are changed.
"""
import json
from pathlib import Path
import time

import mlx.core as mx
import psutil


def prepare_memory(trace_path=None, cache_limit_bytes=8*2**30):
    from omlx.patches.deepseek_v41 import loading
    from omlx.utils.proc_memory import get_phys_footprint
    device=mx.device_info()
    reserve=32*2**30
    limit=min(device['max_recommended_working_set_size'],device['memory_size']-reserve)
    if limit<210*2**30:
        raise MemoryError('This 1017 lab requires enough memory for the verified 205 GiB checkpoint and its working space')
    mx.synchronize()
    previous=mx.set_wired_limit(limit)
    mx.set_memory_limit(limit)  # MLX allocation guideline, not a hard OS cap.
    mx.set_cache_limit(cache_limit_bytes)
    original=getattr(loading,'_flash1017_original_load_shard',loading._load_shard)
    trace=Path(trace_path) if trace_path else None
    floor=12*2**30

    def check(stage,path):
        vm=psutil.virtual_memory()
        row={'time':time.time(),'stage':stage,'shard':Path(path).name,
             'available_bytes':vm.available,'system_wired_bytes':vm.wired,
             'process_footprint_bytes':get_phys_footprint(),
             'mlx_active_bytes':mx.get_active_memory(),'swap_bytes':psutil.swap_memory().used}
        if trace:
            trace.parent.mkdir(parents=True,exist_ok=True)
            with trace.open('a') as file:file.write(json.dumps(row)+'\n')
        if vm.available<floor:
            raise MemoryError('Stopped checkpoint loading to preserve at least 12 GiB of system headroom')

    def guarded(path):
        check('before',path)
        values=original(path)
        check('after',path)
        return values

    loading._flash1017_original_load_shard=original
    loading._load_shard=guarded
    return {'wired_limit_bytes':limit,'previous_wired_limit_bytes':previous,
            'allocator_guideline_bytes':limit,'cache_limit_bytes':cache_limit_bytes,
            'load_available_floor_bytes':floor,'physical_memory_reserve_bytes':reserve}
