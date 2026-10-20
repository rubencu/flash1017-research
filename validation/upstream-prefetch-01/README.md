# DeepSeek V4.1 1017: prefetch admission for wide prompt chunks

This experiment isolates one change: raising the maximum packed Engram read
accepted by the existing prefetch worker from 16 to 64 MiB. At an 8,192-token
prefill step, four matched requests finished 13.6–20.9% faster. The equal-case
geometric mean reduction was 16.0%. **Default 2,048-token prefill already fits
the old limit; this is not a default-setting speedup.**

The runtime diff changes one constant and adds a comment and two regression
cases. It preserves row selection, decoding arithmetic, the single pending
future, synchronous fallback above the cap, and drain/close behavior. There
is no persistent row cache or experimental generation policy in either arm.

## Source and configuration

- Baseline: oMLX main `73322ed188a25b9ce2e490d3137d810f7d50128d`.
- Candidate: `058cf8473b7d49b078ebf43125f42854253e8fb7`; [exact patch](candidate.patch).
- Machine: M5 Ultra Mac Studio, 80 GPU cores, 256 GiB unified memory.
- Checkpoint: `mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp`, revision
  `78456115d301eecf9fadcdcc81c0cf6f8b2c8eab`.
- Python 3.12.15, MLX 0.32.4, mlx-lm `94cdcae13b266c337bcaca09b97b9c5a9c0e2cde`,
  mlx-vlm `9790231827a890eade1a626643a6bc94a1317c72`.
- Engram SSD offload, resident experts, greedy sampling, native adaptive MTP
  capped at three, one request at a time, 128 output tokens, 8 GiB MLX cache limit.
- Optional compiled oMLX extensions absent in both checkouts; normal MLX Metal
  execution active. No kernel source changes.

Each arm explicitly imports its own source checkout and asserts the package
location. The environment's editable-distribution metadata is not the runtime
selection mechanism. See the manifests in the four raw result files.

## Design and results

Four fresh processes ran in baseline/candidate/candidate/baseline order. Each
had an excluded 512-input-token, 16-output-token warmup. Within a process the
three measured workloads shared file-cache state but had fresh KV state.
The second block reversed workload order. OS file caches were not purged.

The control uses 16,384 input tokens with 2,048-token chunks. The code and
prose treatments each use 32,768 input tokens with 8,192-token chunks. These
are direct native inference timings, not HTTP or client-visible-text timings.
Prefill uses the native `_omlx_prefill` path; generation uses the upstream MTP
batch generator. All input and output token IDs are retained.

| Fixed geometry | Pairs | Total time | First-token time | Decode throughput |
|---|---:|---:|---:|---:|
| 2,048-token control | 2 | 0.68% higher | 0.74% higher | 0.15% lower |
| 8,192-token treatment | 4 | 16.01% lower | 16.64% lower | 0.35% lower |

Ratios are geometric means of matched pairs, with equal weight per case.
All 6 pairs, covering 768 generated token positions, matched exactly.

| Workload | Block | Baseline seconds | Candidate seconds | Time reduction |
|---|---:|---:|---:|---:|
| Code, 16K, step 2048 | 0 | 26.520 | 26.668 | −0.56% |
| Code, 32K, step 8192 | 0 | 52.250 | 44.556 | 14.72% |
| Prose, 32K, step 8192 | 0 | 132.269 | 114.314 | 13.57% |
| Code, 16K, step 2048 | 1 | 25.937 | 26.146 | −0.81% |
| Code, 32K, step 8192 | 1 | 84.596 | 72.177 | 14.68% |
| Prose, 32K, step 8192 | 1 | 91.320 | 72.254 | 20.88% |

At 24 IDs per token and 160 packed bytes per row, each table needs 7.875 MiB
including IDs for 2,048 tokens, or 31.5 MiB for 8,192 tokens. Instrumentation
confirmed that all control reads were already prefetched. Each wide request
had eight synchronous gathers above 16 MiB on baseline; all eight ran on the
prefetch worker on candidate. Both arms used the same observation wrappers.

No measured request increased system swap usage. The wide-pair difference in
sampled peak process footprint ranged from −300 to +60 MiB; absolute peaks
were about 218 GiB. The cap covers selected packed bytes and IDs, not all
temporary allocations or total process memory. Sampling every 200 ms can miss
short peaks. The worker retains only one pending read per model.

The broad spread in absolute times makes matched comparisons important. Two
process blocks are not a confidence interval or a statistical significance
claim. These observations cover this checkpoint, device and configuration;
they do not establish benefits for default chunks, other quantizations,
concurrent requests, cluster serving, or optional compiled extensions.
The default chunk size is unchanged by the runtime patch.

## Reproduce

Use the two source revisions above in separate checkouts, with one shared
Python 3.12 environment. The candidate's `make dev-no-kernels` installs the
upstream dependency pins used here. Compare versions with [environment.json](environment.json).
Obtain the exact checkpoint revision and unload other large local models;
the runner requires at least 210 GiB available before loading.

From this directory, set `BASELINE_CHECKOUT`, `CANDIDATE_CHECKOUT`, and
`MODEL_DIR` to the corresponding absolute paths, then run:

```bash
python benchmark.py --runtime "$BASELINE_CHECKOUT" --model "$MODEL_DIR" --output-dir rerun --arm baseline --block 0
python benchmark.py --runtime "$CANDIDATE_CHECKOUT" --model "$MODEL_DIR" --output-dir rerun --arm candidate --block 0
python benchmark.py --runtime "$CANDIDATE_CHECKOUT" --model "$MODEL_DIR" --output-dir rerun --arm candidate --block 1
python benchmark.py --runtime "$BASELINE_CHECKOUT" --model "$MODEL_DIR" --output-dir rerun --arm baseline --block 1
python analyze.py --results rerun
```

The portable runner adds model/output path arguments and imports the resource
meter from a local file. Its generation function and the meter were checked
by Python syntax-tree comparison against the measured implementation; both
are unchanged. The original runner is [archived](raw/measured-benchmark.py).
To recalculate the recorded results without loading a model, run
`python analyze.py`. The analyzer checks plan/input hashes, exact token counts,
prefetch engagement and completed cleanup before calculating ratios.

## Regression validation

- 39 storage/offload tests passed, including packed/affine reads, cancellation,
  mismatched-request fallback and close after resource errors.
- The new 8,192-token admission test fails on untouched baseline and passes on
  candidate. The new over-cap test rejects 16,645 tokens and verifies ordinary
  synchronous packed-row results.
- DeepSeek suites: 752 passed, 70 skipped.
- Default suite: 18,028 passed, 690 skipped, two failures. Both failures in
  `test_glm5_next_mtp.py::test_split_k_batched_decode_matches_rows_alone`
  reproduced on untouched baseline. Black, Ruff and whitespace checks passed.

Full logs and XML results are in [test-logs.tar.gz](test-logs.tar.gz), with a
[SHA-256 sidecar](test-logs.tar.gz.sha256). See [validation.json](validation.json)
and [raw results](raw/). The startup attempt that lacked free memory stopped
before model loading or timing; it is not a measured observation.

## Input provenance

Inputs reuse public CPython and Pride and Prejudice token fixtures. The wide
workloads take the first 32,768 IDs of 65,536-token fixtures; their trailing
question/generation header is therefore absent. They measure fixed-prefix
execution and output equivalence, not semantic task scores. Details are in
[prompt-provenance.json](prompt-provenance.json) and [plan.json](plan.json).

The original source files, source licenses and Gutenberg notices accompany
this repository under
[the fixture archive](../../archives/2026-10-18-preview/evidence/overall/fixtures/).
CPython is credited to the Python Software Foundation and contributors;
Pride and Prejudice to Jane Austen, with the Gutenberg preparation credits
retained in that archive. Preserve those sources/notices when redistributing
the token fixtures. Experiment code is covered by the repository's Apache-2.0
license; third-party texts retain their own terms.
