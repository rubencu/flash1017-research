# Flash-1017 on a 256 GiB Mac: faster generation with matched outputs

**Draft for review.** The installed v2 fork delivered **7.9% higher generation
throughput and 4.5% less time to first visible text** across eight workloads on
one M5 Ultra Mac Studio. In a separate quality comparison, all **618 answer
texts and finish reasons matched** public oMLX, with zero observed task
regressions. These are results for a specific build, checkpoint and workload;
they do not establish universal equivalence or performance on other machines.

DeepSeek-V4.1-Flash-1017 already runs in oMLX, an inference server built on
Apple's MLX framework. This project improves how the existing
[mlx-community oQ2.5e-MTP checkpoint](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp/tree/78456115d301eecf9fadcdcc81c0cf6f8b2c8eab)
executes, without changing its weights or quantization. The measured package
is `0.7.1rc1+flash1017.2`, runtime commit
`9280d25a1322c39f2f6713dc22b099a6e9dd41d1`. The reference is public oMLX
`v0.7.1rc1` with native speculative decoding enabled, not upstream `main`.

| Current installed v2 result | Observed change |
|---|---|
| Generation throughput, eight mixed workloads | **+7.9%**, improved in 16/16 paired observations |
| First visible text, the same eight workloads | **−4.5% latency**, improved in 11/16 pairs |
| Six natural-source workloads | **+8.0% generation throughput; −5.5% first-text latency** |
| Separate quality evaluation | **618/618 identical response texts and finish reasons; zero regressions** |

The generation gain was consistent across the two process blocks. First-text
latency varied considerably for some cases, so its smaller aggregate gain
needs more caution. The
[performance evidence and offline reproduction guide](../validation/performance-v2-01/README.md)
contains every measurement, both block ratios and the exact analysis sources.

## What the current runtime changes

Two parts of execution matter: **prefill** processes the input prompt, and
**generation** produces output tokens. oMLX's multi-token prediction (MTP)
path uses DeepSeek's DSpark drafter to propose tokens for the target model to
verify together. Both builds retain native adaptive MTP with maximum depth 3.

V2 combines three retained optimizations:

- **Vocabulary-weight reuse.** Head rows 2–5 share BF16 weight reads across
  token positions while retaining the original FP32 accumulation and reduction
  order. Row 1 and rows 6 or greater use the stock paths.
- **Raw row caching.** A 512 MiB retained cache, divided between two Engram
  lookup tables, reuses bytes already read from disk. Complete row-ID tags,
  bounds checks and inverse indexing preserve the requested rows and order.
- **Wider prefetch admission.** The background Engram reader admits up to
  64 MiB rather than 16 MiB, retaining the same row identity checks and native
  decode path.

The production profile keeps stock attention behavior and configured prompt
chunks of at most 2,048 tokens. It disables the experimental confidence
scheduler and dependency replay. The cache allowance excludes temporary
allocations, and native memory-pressure logic may reduce prompt chunk sizes.
Correctness arguments assume immutable weights and proper draining of model-owned
resources on unload. They apply to this pinned profile, not every optional
setting in the fork.

The contribution is the integrated implementation, numerical and lifecycle
repairs, and measured verification of this model/runtime combination. It builds
on existing techniques and community implementations; the
[attribution record](ATTRIBUTION.md) identifies those sources.

## How the installed-wheel performance test works

The test used an **M5 Ultra Mac Studio with 80 GPU cores and 256 GiB unified
memory**. Both arms used the same checkpoint, Python, dependency versions and
native assets. Eight fixed prompts cover two synthetic ≈1K cases, four natural
≈16K cases and two 64K cases, using CPython, curl, *Moby-Dick* and *Pride and
Prejudice*. These are execution workloads, not answer-quality benchmarks.

Each arm ran in two fresh processes. Every process loaded the model, ran one
excluded 16K warmup, then all eight cases. The second block reversed arm and
case order. That produced **32 measured requests, four warmups and 16 matched
pairs**. Prefix reuse was disabled; the separate Engram row cache retained its
history after warmup. Operating-system page caches were not purged. This is
warm loaded-model performance, excluding model loading.

Requests used public streaming HTTP, temperature 0, thinking disabled,
ordinary EOS and a 512-token output budget. Client first-text latency includes
transport and stops on the first nonempty content event. Prompt and generation
throughput come from native server usage counters. Each case first receives
the geometric mean of its two block ratios; group results give each case equal
weight. A latency ratio of 1.047192 means 4.507% less latency, not 4.719% less.

All 16 pairs matched text, finish reason and prompt/completion counts. Only
**three cases finished normally**: *Moby-Dick* ≈16K and *Pride and Prejudice*
at 16K and 64K. Their six pairs showed **7.6% less completed-answer latency**.
The other five cases hit 512 tokens in both arms; their throughput and
first-text results remain included, without calling those intervals completed
answers.

Two blocks on one machine do not support confidence intervals, tail-latency
claims or population significance. For the 64K novel case, first-text latency
was about 6.7% worse in one block and 21.7% better in the other. Generation
throughput improved in every pair, making it the more consistent observation.

Requested memory guidelines matched, but live wired policies differed: stock
startup raised its limit to about 243.2 GiB and v2 reset it to 224 GiB before
load. The test compares those native profiles, rather than asserting identical
live limits. Cumulative MLX allocator peaks include load and warmup; they do
not measure total system memory or reproduce the older prototype's footprint
metric. [Methods and results](RESULTS.md) gives the detailed qualifications.

The complete comparison passed 16,193 integrity checks, and the original
server's loaded model was restored. A prior attempt stopped after 16 requests
when that server changed during a headroom wait. Its failed report and raw
observations are preserved with no aggregate; the successful attempt reran
all four processes under the unchanged plan.

## Why matching outputs was an explicit requirement

The first installed build, v1, failed its quality gate. Both it and the public
baseline passed 447/542 coding tasks, but the equal totals concealed five
regressions and five gains. Synthetic long-context controls fell from 11/12 to
10/12, with two regressions and one gain. Improvements could not offset the
**seven baseline-pass to candidate-fail tasks**. Only 43/618 answer texts and
finish reasons matched. The [v1 evidence](../validation/quality-v1-01/README.md)
retains those failures rather than presenting equal totals as preserved quality.

That finding prompted the conservative v2 profile and a new acceptance plan,
frozen before v2 generation. It requires every pair, exact text/finish parity,
zero task regressions and no candidate-only output-budget stops. V2 was compared
with the exact original baseline response file, reused by hash. Baseline
inference was not repeated; coding answers in both arms were freshly graded
under the same amended seed-controlled evaluator.

| Evaluation | Baseline passed | V2 passed | Regressions |
|---|---:|---:|---:|
| HumanEval+ v0.1.10 | 148/164 | 148/164 | 0 |
| MBPP+ v0.2.0 | 299/378 | 299/378 | 0 |
| Fixed GSM8K test subset | 61/64 | 61/64 | 0 |
| Synthetic 16K/64K retrieval controls | 11/12 | 11/12 | 0 |

All **618 texts and finish reasons matched**. Four original baseline budget
stops occurred identically in v2 and remain included. The earlier absolute
zero-truncation rule still fails for the baseline itself and is preserved
separately; the amended conservation rule passed.

Coding uses the full pinned EvalPlus releases, unchanged sanitizer and oracles,
and a local isolated macOS adapter. Canonical solutions passed 540/542 because
of two documented upstream behaviors on `HumanEval/32` and `Mbpp/630`; both
stay in both denominators. The seeded amendment fixes the original worker's
ignored hash seed, without claiming control over all possible program
nondeterminism. The 64 math questions are a fixed GSM8K subset; retrieval cases
are synthetic engineering checks, not a standard long-context benchmark.

The first v2 report misread adaptive MTP as fixed depth. An analysis-only
correction fixed that check without changing requests, outputs, grading or
acceptance rules. Both reports and exact sources are preserved. The
[quality guide](QUALITY.md) explains these limits and the final report's
15,900 integrity checks. The numerical implementation argument and finite
observed parity support this bounded result, not equality for every possible
prompt, sampling setting or machine.

## Historical prototype results are separate

An earlier prototype combined a wider numerical and scheduling profile:
reference attention rounding, five-token confidence scheduling, 8K prompt
chunks, dependency replay, head reuse through six rows and row caching. It
reduced geometric-mean first-token time **19.554 → 15.631 seconds, or 20.06%**,
against the installed application's stock 2K configuration. That test excluded
HTTP transport, used different native binaries and settings, and produced
different output streams. Sampled peak footprint rose 7.75 GiB and a 64K code
case slowed by 2.32%. The percentage does not describe the current wheel.

The prototype exposed shape-dependent floating-point effects and draft-state
failures. Smaller tensors can select different kernels or reduction orders;
reference rounding is itself a numerical choice. Replay required preserving
attention boundaries and correctly staging the draft token/cache state.
Those repairs and unsuccessful experiments remain inspectable in the archive.

Two component studies had different references: vocabulary-weight reuse gave
7.34% higher generation throughput with 10,240 matching token positions against
a reference already using attention rounding; replay plus row caching reduced
prompt-processing time 40.33% with 1,536 matching positions against a modified
reference. These results cannot be added to one another or transferred to v2.
The [methods document](RESULTS.md) and
[historical archive](../archives/2026-10-18-preview/README.md) retain the full
configurations, plots, failures and analysis.

## What is available for review

The companion [rubencu/omlx fork](https://github.com/rubencu/omlx) holds runtime
source. This research repository contains the article, raw evidence, frozen
protocols, analysis code and attribution. The measured preview branch and
wheel, and the proposed research remote, await publication review. Historical
[`RUNTIME-LINK.json`](../RUNTIME-LINK.json) identifies v1; the
[v2 identity](../validation/quality-v2-01/BUILD-IDENTITY.json) names the measured
build, independently of subsequent documentation commits.

The [offline performance commands](../validation/performance-v2-01/README.md#reproduce-from-saved-files)
recompute all ratios without a model. Supplying the two exact wheels also
replays the frozen source/native identity audit, without installing them or
using private settings. Repeating inference requires the pinned checkpoint,
sufficient hardware and the declared process/cache sequence.
