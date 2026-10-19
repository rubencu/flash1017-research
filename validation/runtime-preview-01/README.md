# Installable fork validation, October 18

Runtime commit: `ad2f74058056e4cc7c0a13ddc76f96602e8ce092`, branch
`flash1017-preview`, based on public oMLX `v0.7.1rc1` at `7a5a3d75`.

Wheel: `omlx-0.7.1rc1+flash1017.1-cp312-cp312-macosx_15_0_arm64.whl`

SHA-256: `b5c333adc078c28d0950e925af55d5da628d0915e0586bc6ac62f97d35400554`

The wheel was installed with its declared dependencies in a new Python 3.12
environment. Imports and checks ran outside the runtime checkout. `pip check`
reported no broken requirements. All five packaged native modules imported.
The model and test machine were the same M5 Ultra/256 GiB setup as the archived
study, but the public native binaries differ from that historical application.

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
- **Port preservation:** 24 retained arithmetic functions/methods have identical
  ASTs to their frozen sources. The third-party HotRows file remains unchanged.
- **Checkpoint continuity:** all 61 saved file fingerprints still match the
  earlier full-hash verification. This is not a new full-weight hash pass.
- **Historical archive integrity:** all 337 files still match the frozen record.

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

The original oMLX server stayed running; its idle Qwen model was temporarily
unloaded to make room and then reloaded after this test. No original server
settings were changed.

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

This record establishes a bounded functional preview. It does **not** establish
the old 20.06% speedup for this wheel, equivalent answer quality, general model
support, or performance on another machine. Timings included by normal HTTP
usage reporting are incidental smoke-test observations, not benchmark results.
