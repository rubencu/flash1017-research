# SPDX-License-Identifier: Apache-2.0
"""Unmodified resource meter used by the recorded experiment."""
import ctypes
import os
import threading
import time
import mlx.core as mx
import psutil

def process_io():
    # Reuse oMLX's macOS ABI definition, including Metal-backed footprint.
    from omlx.utils.proc_memory import _RusageInfoV4, _proc_pid_rusage, _RUSAGE_INFO_V4
    if _proc_pid_rusage is None:
        return {}
    info = _RusageInfoV4()
    if _proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(info)):
        return {}
    return {"disk_read_bytes": int(info.ri_diskio_bytesread),
            "disk_write_bytes": int(info.ri_diskio_byteswritten)}


class Meter:
    """Sample the process ledger and system state; report the sampling interval."""
    def __init__(self):
        self.done = threading.Event()
        self.samples = []
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _snapshot(self):
        from omlx.utils.proc_memory import get_phys_footprint
        vm = psutil.virtual_memory()
        self.samples.append({"time": time.perf_counter(),
                             "footprint_bytes": get_phys_footprint(),
                             "system_used_bytes": vm.used,
                             "available_bytes": vm.available,
                             "swap_used_bytes": psutil.swap_memory().used,
                             "mlx_cache_bytes": mx.get_cache_memory()})

    def _sample(self):
        while not self.done.wait(.2):
            self._snapshot()

    def start(self):
        self._snapshot()
        self.io_before = process_io()
        self.thread.start()
        return self

    def stop(self):
        self.done.set()
        self.thread.join()
        self._snapshot()
        io = process_io()
        return {
            "sampling_interval_seconds": .2,
            "process_footprint_peak_bytes": max(s["footprint_bytes"] for s in self.samples),
            "system_used_peak_bytes": max(s["system_used_bytes"] for s in self.samples),
            "system_available_min_bytes": min(s["available_bytes"] for s in self.samples),
            "swap_start_bytes": self.samples[0]["swap_used_bytes"],
            "swap_peak_bytes": max(s["swap_used_bytes"] for s in self.samples),
            "swap_end_bytes": self.samples[-1]["swap_used_bytes"],
            "mlx_cache_peak_bytes": max(s["mlx_cache_bytes"] for s in self.samples),
            "mlx_active_bytes": mx.get_active_memory(),
            "mlx_peak_bytes": mx.get_peak_memory(),
            **{f"{key}_delta": value - self.io_before.get(key, value)
               for key, value in io.items()},
        }

