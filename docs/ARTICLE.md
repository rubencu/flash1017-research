# Faster Flash-1017 inference on a 256 GiB Mac, with matched outputs

Sharing repeated weight reads and caching lookup-table rows increased
**DeepSeek-V4.1-Flash-1017 generation throughput by 7.9%** on a 256 GiB M5 Ultra
Mac Studio. Across eight workloads, client time to first visible text fell
**4.5%**. In a separate quality evaluation, the optimized package matched the
public oMLX baseline's **answer text and finish reason on all 618 requests**,
with zero observed task regressions.

The model already runs in oMLX, an inference server built on Apple's MLX
framework. This work changes its execution using the existing
[mlx-community oQ2.5e-MTP checkpoint](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp/tree/78456115d301eecf9fadcdcc81c0cf6f8b2c8eab),
without changing weights or quantization. The measured package is
`0.7.1rc1+flash1017.2`, commit `9280d25a1322c39f2f6713dc22b099a6e9dd41d1`.
The reference is public oMLX `v0.7.1rc1` with native speculative decoding enabled.
All results below apply to these builds, this checkpoint and this machine.

## Reducing repeated memory work

Inference has two main phases. **Prefill** processes the input prompt;
**generation** produces output tokens. oMLX's multi-token prediction (MTP)
path uses DeepSeek's DSpark drafter to propose tokens for the target model to
verify together. Both measured builds use native adaptive MTP with maximum
depth 3. Verifying several token positions creates an opportunity to reuse
weights that would otherwise be read repeatedly.

The vocabulary projection maps each token position's activations to vocabulary
scores. Its BF16 weight matrix is shared across those positions. The optimized
kernel loads weights once for reuse across two to five verification rows,
while preserving the stock FP32 accumulation and reduction order for each row.
Single-row inputs and inputs with six or more rows use the stock paths. This
reduces repeated weight traffic without changing the represented weights.

Engram, the model's lookup-table component, presents a different opportunity.
In this configuration, oMLX reads the requested rows from disk. A **512 MiB
cache**, divided between two tables, keeps raw row bytes for reuse. Complete
row-ID tags and bounds checks identify valid entries; inverse indexing restores
the requested row order. The cache changes where the bytes come from, without
adding a dequantization or numerical transformation.

The background Engram reader also admits up to **64 MiB of prefetch work**,
compared with 16 MiB in the public release. It retains the same row identity
checks and native decode path. The benchmark measures these three changes
together; it does not assign a separate percentage gain to each one.

The measured profile keeps stock attention behavior and prompt chunks
configured at no more than 2,048 tokens. Native memory-pressure logic may
reduce those chunks. Cache capacity refers to retained rows and excludes
temporary allocations. The correctness argument assumes immutable weights
and proper draining of model-owned resources on unload; it is scoped to the
identified profile and supported kernel shapes.

## Performance through the normal HTTP interface

The test machine was an **M5 Ultra Mac Studio with 80 GPU cores and 256 GiB
unified memory**. Both builds used the same checkpoint, Python, dependency
versions and native assets. Eight fixed prompts cover two synthetic ≈1K cases,
four natural ≈16K cases and two 64K cases, using CPython, curl, *Moby-Dick* and
*Pride and Prejudice*. These are execution workloads rather than scored
capability tasks.

Each build ran in two fresh processes. Every process loaded the model, ran one
excluded 16K warmup, then all eight cases. The second block reversed build and
case order: **32 measured requests, four warmups and 16 matched pairs** in total.
Prefix reuse was disabled, while the separate Engram row cache retained its
history after warmup. Operating-system page caches were not purged. This
measures warm loaded-model behavior and excludes model loading.

Requests used public streaming HTTP, temperature 0, thinking disabled,
ordinary end-of-sequence stopping and a 512-token maximum. Client first-text
latency includes transport and ends at the first nonempty content event.
Prompt and generation throughput use the native server's usage counters.

| Workload group | Generation throughput | First-text latency |
|---|---:|---:|
| All eight, mixed synthetic/natural | **7.9% higher** | **4.5% lower** |
| Six natural-source cases | **8.0% higher** | **5.5% lower** |
| Two synthetic ≈1K cases | 7.6% higher | 1.4% lower |
| Four natural ≈16K cases | 7.8% higher | 4.4% lower |
| Two natural 64K cases | 8.5% higher | 7.7% lower |

Each case receives the geometric mean of its two paired block ratios, then
contributes equal weight to its group. All 16 pairs matched text, finish reason
and prompt/completion counts. Throughput improved in every pair. First-text
latency improved in 11/16 pairs and varied substantially for some cases: the
64K novel case was about 6.7% slower in one block and 21.7% faster in the other.
The throughput gain is therefore the more consistent observation.

Only **three workloads finished normally within 512 tokens**: *Moby-Dick* ≈16K
and *Pride and Prejudice* at 16K and 64K. Their six pairs showed **7.6% less
completed-answer latency**. The other five workloads hit the budget in both
builds. Their throughput and first-text measurements remain included, but
their timed intervals are not described as completed answers.

Two blocks on one machine do not establish population significance, tail
latency or performance on other hardware. Requested memory guidelines matched,
but effective wired policies differed: stock startup raised its limit to
about 243.2 GiB, while the optimized profile set 224 GiB before load. The
comparison includes those native profile behaviors. [Methods and results](RESULTS.md)
provides every case, the ratio formulas and the measurement limitations.

## Checking answer preservation

The quality comparison used **618 fixed requests** through ordinary HTTP chat,
with greedy generation, thinking disabled and normal stopping. Coding and
math requests allowed 2,048 output tokens; synthetic retrieval controls allowed
64. Baseline responses came from a saved public-release generation run and
were reused by exact file hash. Baseline inference was not repeated for this
comparison. Coding answers from both builds were freshly graded using the
same frozen, seed-controlled evaluator.

| Evaluation | Public baseline passed | Optimized package passed | Regressions |
|---|---:|---:|---:|
| HumanEval+ v0.1.10 | 148/164 | 148/164 | 0 |
| MBPP+ v0.2.0 | 299/378 | 299/378 | 0 |
| Fixed GSM8K test subset | 61/64 | 61/64 | 0 |
| Synthetic 16K/64K retrieval controls | 11/12 | 11/12 | 0 |

Every answer text and finish reason matched exactly. The acceptance rule,
frozen before optimized-package generation, required every pair, no generation
errors, zero task regressions and no additional output-budget stops. Four
baseline truncations occurred identically in the optimized build and remain
in the evaluation. Matching responses preserves the baseline's observed
successes and failures; it does not mean every answer is correct.

Coding uses the full pinned EvalPlus releases with the unchanged sanitizer
and special oracles through a local isolated macOS adapter. Canonical solutions
passed **540/542** tasks: two documented upstream behaviors reject
`HumanEval/32` and `Mbpp/630`. Both stay in both denominators, without repaired
model answers. Workers control hash seeds and the shared Python/NumPy random
generators, but cannot guarantee determinism for every possible program.
The math result covers a fixed 64-question GSM8K subset. The retrieval cases
are synthetic engineering controls, not a standard long-context benchmark.

Preserving per-row arithmetic and exact cached bytes provides an implementation
argument; the 618 observed matches provide a finite behavioral test. Neither
proves equality for every prompt, sampling setting or hardware configuration.
The [quality methods](QUALITY.md) explain task provenance, grading and the
remaining limits in detail.

## Inspecting and reproducing the result

The [performance evidence](../validation/performance-v2-01/README.md) contains
raw request/response records, both block ratios, frozen sources and offline
reproduction commands. The [quality evidence](../validation/quality-v2-01/README.md)
contains all 618 requests and responses, grades and analysis provenance.
Detailed [evidence provenance](EVIDENCE-PROVENANCE.md) records source identities,
execution records and analysis history.

Saved performance arithmetic can be reproduced without a model or GPU.
Supplying the two exact wheel files also replays the source/native identity
audit without installing them or using private settings. Repeating inference
requires the pinned checkpoint, suitable hardware and the declared process/cache
sequence. The [companion oMLX fork](https://github.com/rubencu/omlx) is the runtime
repository; this repository provides the research evidence and analysis.

These optimizations build on established techniques and community implementations.
The contribution is their integration into this execution path and the measured
verification of the resulting package. [Attribution](ATTRIBUTION.md) identifies
the reused work and contribution boundaries.
