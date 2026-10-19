# Installable oMLX fork: build and serving validation

This record tests whether the Flash1017 fork installs and serves
DeepSeek-V4.1-Flash-1017 through oMLX's normal server path. It covers package
`0.7.1rc1+flash1017.1`, tested on October 18, 2026. It is functional validation;
the [latency experiments](../../docs/RESULTS.md) used a different prototype build.

## Artifact and environment

Runtime commit: `ad2f74058056e4cc7c0a13ddc76f96602e8ce092`, branch
`flash1017-preview`, based on public oMLX `v0.7.1rc1` at `7a5a3d75`.

Wheel: `omlx-0.7.1rc1+flash1017.1-cp312-cp312-macosx_15_0_arm64.whl`

SHA-256: `b5c333adc078c28d0950e925af55d5da628d0915e0586bc6ac62f97d35400554`

The wheel was installed with its declared dependencies in a new Python 3.12
environment. Imports and checks ran outside the runtime checkout. `pip check`
reported no broken requirements. All five packaged native modules imported.
The test machine was an M5 Ultra Mac Studio with 80 GPU cores and 256 GiB
unified memory. The model was `mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp`
at revision `78456115d301eecf9fadcdcc81c0cf6f8b2c8eab`, with unchanged weights.
The packaged native binaries come from a public oMLX release and differ from
the installed application's binaries used for the prototype latency study.

## Checks and results

- **Full default suite:** 18,054 passed, 12 failed, 460 skipped, 85 deselected,
  and 20 subtests passed. All twelve remaining failures are the same
  `test_dspark_attention_matches_stock_m1_fallback` cases reproduced on the
  unchanged public base with the same dependencies and native assets. This is
  not an entirely green suite; the exact failing node IDs are retained.
- **Installed-package ownership/lifecycle tests:** 33 passed.
- **Real Metal projection tests:** six cases passed, including hidden width
  5,120 and a vocabulary tail. Rows 2–5 exactly matched the normal batched
  kernel. Row 6 exactly matched six original one-row reductions; it was not
  compared as equivalent to the old six-row GEMM fallback.
- **Source continuity:** the parsed code structure (abstract syntax tree) of 24
  arithmetic functions/methods matches the prototype snapshot. The third-party
  HotRows file is byte-identical to its credited source. This checks preservation
  of those code units, not whole-runtime equivalence.
- **Checkpoint continuity:** all 61 file fingerprints match the saved receipt
  from full-weight hash verification. This check did not rehash all model weights.
- **Research archive integrity:** all 337 files match the snapshot's manifest.

## Real installed-server smoke check

The [harness](../../tools/smoke_installed_runtime.py) launched the real installed
`omlx flash1017` command and sent HTTP requests. It did not replace inference
functions. Observation of runtime counters occurred at idle boundaries.

Four requests completed: a short arithmetic question, an 8,314-token code
summary, an 8,319-token prefix continuation, and a short recovery request after
a cancelled stream. The continuation reported **8,312 cached prompt tokens**.
The arithmetic answer was “Two plus two equals four.” The recovery answer was
“Paris”. These are functional checks, not a model-quality evaluation.

Both an oversized prompt and a small prompt with an oversized output budget
returned HTTP 400. The latter carried `context_length_exceeded`. A stream was
closed after content arrived, and the following request succeeded. Normal
server shutdown closed the runtime controller and released its model ownership.

The recorded configuration was one active request, an 8,192-token maximum
prefill chunk, a 131,072-token total admission cap, and a separate prefix-cache
namespace. Counters demonstrate replay/shape repair and all three DSpark
priming stages, confidence-policy activity, a 64 MiB prefetch budget, and two
256 MiB retained row caches. The live check did not exercise a successful
128K request or million-token serving.

To provide enough memory, the host's other loaded model was temporarily
unloaded while idle and restored after the test. The test server then stopped.
The restoration receipt is included for audit of the harness's cleanup.

## Evidence and reproduction

See the installation receipt, build provenance, full test summary, smoke JSON
and compressed logs in this directory. `log-provenance.json` records original
log hashes and the path-prefix redaction applied before compression.
`MANIFEST.json` binds the files here; it is separate from the immutable
historical archive manifest.

Build/install commands belong in the runtime fork's `docs/flash1017-install.md`.
For the serving harness, unload other large models first and run the installed
environment's Python with `-I -B`, supplying `--model`, an empty `--base-path`,
`--corpus` pointing to the archived CPython corpus, `--output` pointing to a
new local result file, and an unused `--port`. The loader requires at least
210 GiB available before loading. The test starts and stops its own server.

These checks support a functional preview on the identified machine and
checkpoint. They do not measure a speedup for this wheel or establish equivalent
answer quality, support for other models or performance on another machine.
Timings included by normal HTTP usage reporting are incidental observations
from the serving check, not a controlled performance comparison.
