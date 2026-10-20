# Installed v2: 7.9% higher generation throughput on eight workloads

The installed `0.7.1rc1+flash1017.2` fork completed a paired HTTP comparison
against public oMLX `v0.7.1rc1` on one **M5 Ultra Mac Studio, 80 GPU cores,
256 GiB unified memory**. Across eight workloads, geometric-mean generation
throughput was **7.9% higher** and client time to first visible text was
**4.5% lower**. All 16 workload/block pairs matched answer text, finish reason,
prompt and completion token counts. The separate
[618-request quality comparison](../quality-v2-01/README.md) also matched every
answer and finish reason, with zero observed task regressions.

Generation throughput improved in **16/16 pairs**. First-text latency improved
in **11/16 pairs**, with substantial variation between blocks on some cases.
These are bounded measurements from eight cases and two temporal blocks on
one Mac, not statistical significance, confidence intervals or other-hardware
claims. The measured runtime commit is
`9280d25a1322c39f2f6713dc22b099a6e9dd41d1`; later documentation commits do not
change that identity.

The [passing report](qualified/pair-summary-final-01.json) records **16,193
passed integrity checks**, 32 measured requests and four excluded warmups.
The [supervisor state](qualified/state.json) confirms completion and restoration
of the original server's loaded model. Nothing was published by this workflow.

## Results and denominators

Ratios above 1 favor v2. Token-rate ratios use candidate/baseline; latency
ratios use baseline/candidate. A latency ratio of 1.047192 means **4.507% less
latency**, calculated as `100 × (1 − 1 / ratio)`, not 4.719% less latency.

| Workload group | Cases / paired observations | Generation throughput | First-text latency | Prompt-processing throughput |
|---|---:|---:|---:|---:|
| All eight, mixed synthetic/natural | 8 / 16 | **+7.91%** (1.079092×) | **−4.51%** (1.047192×) | +4.71% (1.047080×) |
| Natural-source cases only | 6 / 12 | **+8.01%** (1.080061×) | **−5.52%** (1.058433×) | +5.85% (1.058469×) |
| Synthetic ≈1K | 2 / 4 | +7.62% | −1.40% | +1.36% |
| Natural ≈16K | 4 / 8 | +7.77% | −4.42% | +4.63% |
| Natural 64K | 2 / 4 | +8.47% | −7.68% | +8.33% |

Only **three cases ended normally within 512 tokens**: *Moby-Dick* ≈16K and
*Pride and Prejudice* at 16K and 64K. Their **six paired observations** had
7.60% lower geometric-mean completed-answer latency (ratio 1.082215). The other
five cases hit the 512-token budget in both arms. Their timings remain in
first-text and throughput results, but are not completed-answer latency.
Warmups are excluded from every performance aggregate.

| Case | Prompt tokens | Generation ratio, two-block geometric mean | First-text ratio, block 0 | First-text ratio, block 1 |
|---|---:|---:|---:|---:|
| Synthetic code | 1,024 | 1.0855 | 1.0316 | 1.0208 |
| Synthetic prose | 1,024 | 1.0669 | 0.9954 | 1.0093 |
| CPython | 16,384 | 1.0884 | 0.9488 | 1.0886 |
| curl | 16,384 | 1.0900 | 1.0023 | 1.0030 |
| *Moby-Dick* | 16,383 | 1.0593 | 1.0888 | 0.9639 |
| *Pride and Prejudice* | 16,384 | 1.0735 | 1.3219 | 0.9969 |
| CPython | 65,536 | 1.0874 | 1.1208 | 1.0259 |
| *Pride and Prejudice* | 65,536 | 1.0820 | 0.9371 | 1.2777 |

All eight per-case first-text geometric means favored v2, but the block-level
reversals make that smaller gain less stable than the generation result. For
example, the 64K novel case was about 6.7% slower in block 0 and 21.7% faster
in block 1. Two blocks cannot establish a precise distribution of latency.

## What was measured

Both builds used the unchanged `mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp`
checkpoint, revision `78456115d301eecf9fadcdcc81c0cf6f8b2c8eab`, matching Python,
110 dependency versions and 10 native assets. The baseline is public release
commit `7a5a3d75ce7463300bdfa5fedccd34348f60ff46`, not upstream `main`.
V2 retains weight reuse for head rows 2–5, raw Engram row caching and 64 MiB
prefetch admission. Both use stock attention behavior, native adaptive MTP
with maximum depth 3 and prompt chunks configured at no more than 2,048 tokens.
Replay and confidence scheduling are disabled in the v2 profile.

Each of four fresh processes loaded the model, ran one excluded representative
16K warmup, then all eight cases. Block 0 ran baseline then candidate; block 1
reversed both arm and case order. Generation used temperature 0, thinking
disabled, ordinary EOS and a 512-token maximum. Prefix reuse was disabled in
normal settings and all responses reported zero cached prompt tokens. V2's
separate Engram row cache retained its state after warmup and between cases.
Operating-system page caches were not purged. This measures **warm loaded-model
performance**, excluding model loading, rather than cold-start performance.

Client first-text time starts immediately before HTTP send and stops on the
first nonempty content event. Role-only, empty and usage-only events do not
count. Native prompt/generation rates are separate server-reported, rounded
usage values. Completion timing ends at the stream's final marker and applies
only to matched normal-stop answers. The frozen analyzer requires every pair;
there is no selective removal of slower or length-stopped cases.

Aggregation first takes the geometric mean of the two block ratios for each
case, then the equal-case geometric mean within each declared group. The 32
requests are not treated as 32 independent workloads. These content-based
latency fixtures are not answer-quality benchmarks; the separate quality
suite supplies the bounded quality evidence.

The HTTP fixtures use the historical corpus excerpts and questions, rerendered
through the pinned chat encoder. Two streams change under normal tokenization:
CPython 64K has different token IDs, and *Moby-Dick* is 16,383 tokens. The new
saved HTTP requests and rendered tokens are authoritative. The draft-named
fixture files retain their preparation names; their exact hashes were frozen
in the final plan before timing. This does not reuse the old timings.

## Limits, memory and the interrupted attempt

Both harnesses requested 224 GiB allocator/wired memory and an 8 GiB allocator
cache. Actual wired policy differs: stock startup raises its wired limit to
about 243.2 GiB; the v2 profile resets it to 224 GiB before load. This comparison
therefore measures those native wheel/profile behaviors, not identical live
wired limits. The plan and receipts preserve this qualification.

Read-only MLX counters show the candidate's cumulative allocator peak differed
from baseline by 81,611,036 bytes in block 0 and zero in block 1. These peaks
include loading and warmup; they are not per-request peaks, whole-process RSS
or total system memory. They are not comparable to the historical prototype's
7.75 GiB sampled-footprint change.

An earlier attempt stopped after two processes and 16 measured requests when
the original server changed during a headroom wait. Its
[failed report](interrupted/pair-summary-final-01.json), raw outputs and
[state](interrupted/state.json) remain unchanged. Its aggregate is null. The
qualified attempt reran the entire four-process comparison under the same
plan and supervisor bytes; it did not select cases from the earlier attempt.
The interruption was not a model-quality failure.

The historical prototype's **20.06%** first-token reduction used different
code, native binaries, settings and direct-inference timing. It remains a
separate historical result and is not this wheel's speedup.

## Reproduce from saved files

Run these commands from the research repository with Python 3.9 or newer.
They perform CPU-only file/arithmetic checks and execute no model or generated
answer code. The checker retains its historical name but accepts this bundle's
manifest schema:

```sh
python3 -I -B tools/verify_quality_evidence.py --folder validation/performance-v2-01
python3 -I -B validation/performance-v2-01/reproduce.py \
  --mode measurements --output /tmp/flash1017-performance-measurements.json
```

The second command recomputes all 16 paired ratios and group aggregates from
the raw response files, verifies paired output equality and compares the
calculation with the saved report. It does **not** repeat the wheel/source or
runtime-identity checks.

To replay the complete frozen audit, supply these two exact wheel files:

| Wheel | SHA-256 |
|---|---|
| `omlx-0.7.1rc1-cp312-cp312-macosx_15_0_universal2.whl` | `b3e12950a607991a1feb833b1bdc0b92fe41aeb73656ca5373db03d0d87b71c7` |
| `omlx-0.7.1rc1+flash1017.2-cp312-cp312-macosx_15_0_arm64.whl` | `0fbdb77ca86ce35b4dca0275b534c1fafde7d70f0f7f6a89d5f964821f7f7f06` |

```sh
python3 -I -B validation/performance-v2-01/reproduce.py \
  --mode full \
  --baseline-wheel /path/to/omlx-0.7.1rc1-cp312-cp312-macosx_15_0_universal2.whl \
  --candidate-wheel /path/to/omlx-0.7.1rc1+flash1017.2-cp312-cp312-macosx_15_0_arm64.whl \
  --output /tmp/flash1017-performance-full.json
```

This stages unchanged evidence in a temporary directory using its recorded
workspace-relative paths, then invokes the exact frozen analyzer. It checks
wheel contents against saved source/native manifests and requires the complete
result to equal the shipped passing report. No wheel is installed. No private
settings, environments or model weights are needed. Wheel binaries are kept
out of the research repository; use the hash-matching runtime release artifacts
when available. A locally rebuilt wheel with a different hash is not a substitute.

Add `--attempt interrupted` to replay the earlier incomplete attempt. Its
expected exit status is 1, with a failed report and no aggregates; missing
second-block files are deliberately not invented. Error paths differ because
of temporary staging. This preserves the failed comparison rather than turning
it into a pass.

## Provenance and file map

| Location | Contents |
|---|---|
| `qualified/` | Final frozen plan, reviewed configuration, supervisor, restoration state, passing report and all four raw process receipts/response streams. |
| `interrupted/` | Earlier unchanged plan/supervisor, reviewed configuration, state, failed report, two completed process records and supervisor fixture tests. |
| `preparation/` | Exact request/payload/rendered-token fixtures, provenance, original draft plan, environment reference and preparation scripts. |
| `frozen-source/` | Exact executed runner, quality-harness support, analyzer, and CPU test sources. |
| `quality-prerequisite/` | Exact passing quality report, protocol, analysis correction and finalization receipt bound by the timing plan. |
| `COPY-PROVENANCE.json`, `MANIFEST.json` | Original source locations, exact copied bytes and complete bundle identity. |
| `reproduce.py` | Packaging helper for the two offline verification levels described above. |

The executed runner is SHA
`35689e6d76464ac132fe945068fe51a91bcb68b8b603da5dcf5b7f08131fb20d`;
its quality support is SHA
`1ad06c439ae008abb9d35ce81f2ed772fd329a1f862353855ccda56dc494b372`.
The analyzer is SHA
`febc26646dcf2198343396a3a0c69ed8e2a502a89939338f358eb52064404ed6`.
The final plan is SHA
`df65cf6313ab49bbc3a4ac9477c7f2a8ab7cd98b3d868b25d9941db3e0021540`.

Original records retain laboratory paths and statuses. The copy map locates the
same bytes in this bundle without rewriting those observations. Raw
secret-bearing settings, caches, environments and weights are excluded.
The analyzer checks saved safe settings and checkpoint metadata/tokenizer
hashes; it does not independently attest to live settings, rehash model shards
or reproduce GPU execution. Checksums are not an external signature.

The fixtures contain excerpts from CPython, curl and two Project Gutenberg
books. [CORPUS-NOTICE.txt](CORPUS-NOTICE.txt) supplies attribution. Original
sources and complete licenses are retained in the
[research archive](../../archives/2026-10-18-preview/evidence/overall/fixtures/).
Redistribute these excerpts with that archive; this folder alone is not the
complete licensed corpus package.
