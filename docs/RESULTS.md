# Methods and results

The research question is how much a set of oMLX execution changes improves
latency for **DeepSeek-V4.1-Flash-1017**, and which numerical, memory and serving
constraints accompany that improvement. This document defines each comparison
before reporting its result.

All performance results below concern the **benchmark prototype**, whose source
and observations are preserved in
[`archives/2026-10-18-preview/`](../archives/2026-10-18-preview/README.md). The
**installable fork**, package `0.7.1rc1+flash1017.1`, integrates the retained
changes with public release binaries. Its installation and serving tests are
covered in the final section; it has not received a performance comparison.

## Hardware, model and terms

The machine was an **M5 Ultra Mac Studio with 80 GPU cores and 256 GiB unified
memory**. The checkpoint was
[`mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp`](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp/tree/78456115d301eecf9fadcdcc81c0cf6f8b2c8eab),
revision `78456115d301eecf9fadcdcc81c0cf6f8b2c8eab`. Its weights and quantization
were unchanged.

**Prefill** is the processing of the input prompt before generation.
**Time to first token (TTFT)** runs from the start of that processing to delivery
of the first output token. **Speculative decoding** proposes several tokens with
a drafter and verifies them with the target model. Here, oMLX's multi-token
prediction (MTP) path uses DeepSeek's DSpark drafter. **Engram** refers to the
model's lookup tables; oMLX reads the relevant table rows from disk in this
configuration. The HotRows cache retains selected rows in memory for reuse.

## Overall comparison: the complete configuration versus oMLX 0.7.1rc1

This experiment tests the combined effect of code and configuration changes.
All three configurations used the same checkpoint, hardware and embedded
Python/dependency environment.

| Configuration | Speculative decoding | Maximum prompt chunk | Engram prefetch budget | Additional execution changes |
|---|---|---:|---:|---|
| **Stock 2K** (`native2048` in raw data) | Native adaptive MTP, maximum depth 3 | 2,048 tokens | 16 MiB | None; installed oMLX 0.7.1rc1 |
| **Stock 8K** (`native8192`) | Native adaptive MTP, maximum depth 3 | 8,192 tokens | 16 MiB | None; the same installed runtime |
| **Complete prototype** (`complete`) | Five-token proposals, confidence-based selection of the prefix to verify | 8,192 tokens | 64 MiB | BF16 vocabulary-weight reuse through six rows; reference attention-output rounding in target and draft; dependency replay with numerical/state repairs; HotRows, 256 MiB per table and 512 MiB total retained cache; ownership and cleanup adapters |

BF16 is the 16-bit floating-point format used by the vocabulary projection.
Dependency replay computes only the decoder suffixes required by the model's
dependency structure. The [article](ARTICLE.md) explains these changes and why
shrinking tensor shapes can change floating-point results.

The reference is a specific installed release with speculative decoding
enabled. Upstream `main` was not measured. Comparing the complete configuration
with Stock 2K gives a combined result; it does not assign a speedup to each patch.
Stock 8K measures what happens when only the prompt-chunk setting is widened.

### Workloads and procedure

The suite is a **custom latency benchmark**. Its inputs come from two synthetic
code/prose prompts and public source material: CPython, curl, *Moby-Dick* and
*Pride and Prejudice*. Those sources supply workload content; the experiment
does not score coding ability or reading comprehension. No standardized
answer-quality benchmark was run.

There are eight fixed cases: two 1K synthetic inputs, four 16K natural-source
inputs and two 64K natural-source inputs. Exact token IDs, execution order and
settings were saved in the [benchmark plan](../archives/2026-10-18-preview/evidence/overall/plan-01.json).
Each configuration ran all eight cases in two fresh processes, producing
**48 measured requests**. Six warmup requests were excluded. The second block
reversed both configuration and case order, with seeds matched across
configurations within each block.

Each request asked for 512 output tokens with stopping disabled, temperature 1,
top-p 1 and thinking disabled. Conversation state was fresh for each request.
Models stayed loaded within each process and operating-system caches were not
purged. The complete prototype's row cache was cleared after warmup, then
retained across that process's eight measured requests. All configurations used
an 8 GiB MLX allocator cache and a 224 GiB wired/allocation guideline; these are
runtime settings, not a hard operating-system memory cap.

Timing begins before prefill and excludes model loading and HTTP transport.
Aggregate times are geometric means with equal weight per case. Plot ranges
show the two process-block means, not confidence intervals. Two blocks on one
machine provide limited evidence about run-to-run variation and none about
other hardware.

### Results

| Metric | Stock 2K | Stock 8K | Complete prototype |
|---|---:|---:|---:|
| Time to first token, geometric mean | 19.554 s | 24.293 s | **15.631 s** |
| Prompt-processing time, geometric mean | 19.190 s | 23.843 s | **15.319 s** |
| Sampled peak process footprint | 208.35 GiB | 215.77 GiB | **216.10 GiB** |

Against Stock 2K, the complete prototype took **20.06% less time to first token**
and **20.17% less prompt-processing time**. Widening chunks alone made the
aggregate slower. Stock 2K is therefore the stronger reference for the headline.

The 16K group's first-token time fell 32.19%; the 1K group's fell only 0.60%.
The 64K CPython case regressed **2.32%**, from 102.073 to 104.445 seconds.
Sampled peak footprint increased **7.75 GiB**. That difference includes the
changed prompt-chunk geometry and cannot all be attributed to the 512 MiB row
cache. Sampling can miss brief memory peaks.

### Why there is no fixed-512 answer-completion claim

All 48 requests completed, but **nine emitted an end-of-sequence (EOS) token
before output token 512: three in each configuration**. The generator continued
because stopping was disabled. The experiment's acceptance rule required the
512-token interval to precede EOS, so that endpoint failed its acceptance rule.
Timing forced continuation after EOS does not measure useful answer completion.

Prefill and first-token measurements precede EOS and satisfy their separate
checks. A supplementary analysis found **20.21% less time to token 128**, using
the saved delivery timestamps in all 48 runs. Every such prefix preceded EOS.
This endpoint was chosen after observing the EOS results, so it is **post hoc**.
It is not a fresh benchmark with a 128-token output limit.

Output streams differ between the complete and stock configurations. These
timings establish neither equivalent answer quality nor faster completion of
equivalent tasks. See the [full report](../archives/2026-10-18-preview/evidence/overall/review/report.txt)
for individual cases, resource observations and every EOS occurrence, and the
[analysis contract](../archives/2026-10-18-preview/analysis/README.txt) for the
rules used to validate and aggregate the records.

## Component comparison: vocabulary-weight reuse

This study asks whether sharing BF16 vocabulary-weight loads across verification
rows improves generation throughput while preserving the selected arithmetic.
Both configurations enabled reference attention-output rounding and used fixed
speculative depth 2. The changed configuration additionally enabled the
weight-reuse kernel. In the saved records they are named `reference-f2` and
`reference-f2-head`.

Across **20 matched pairs**—code and prose inputs at 1K and 16K, five repetitions
each—the paired geometric-mean generation-throughput ratio was **1.07345**, or
about **7.34% higher throughput**. All **10,240 compared output-token positions**
matched. The [saved summary](../archives/2026-10-18-preview/evidence/historical/results/heldout-summary-01.json)
and [configuration plan](../archives/2026-10-18-preview/evidence/historical/plans/heldout.json)
identify the observations and settings.

This result isolates the kernel within a runtime that has reference rounding
enabled. It does not measure the entire package against stock oMLX. Separate
numerical checks found that rows 2–5 match the existing specialized kernel;
row 6 matches six one-row reductions but differs from the original six-row
matrix-multiply fallback. The depth-2 comparison does not establish a row-6
performance or numerical-equivalence claim.

## Component comparison: dependency replay and row caching

This study asks what replay plus HotRows adds when the other optimizations are
held fixed. Both configurations use reference attention rounding, vocabulary
weight reuse through six rows, the confidence policy with draft width 5,
8,192-token prompt chunks and a 64 MiB Engram prefetch budget.

The reference computes the full decoder path with native decoder replay
disabled and bypasses HotRows. Its recorded name is **`native64`**: “64” denotes
the prefetch budget in MiB, not the prompt length. The candidate, **`fastpath64`**,
enables the repaired dependency replay and the two 256 MiB HotRows caches.

Across **12 measured pairs**, geometric-mean prompt-processing time fell
**68.828 → 41.073 seconds, a 40.33% reduction**. All **1,536 compared output-token
positions** matched, along with the recorded functional speculative-decoding
traces. The cases cover CPython, curl and the two novels at 16K and 32K, plus
one longer case per source: 64K for three sources and 128K for *Moby-Dick*.
One warmup pair was excluded. Six measured pairs ran reference first and six
candidate first. Conversation state was fresh, OS caches were retained, and
the candidate row cache persisted from warmup through the measured sequence;
its allocation remained present while reference runs bypassed it.

See the [analysis record](../archives/2026-10-18-preview/evidence/historical/results/serving-v3-performance-01-analysis.json)
and [execution record](../archives/2026-10-18-preview/evidence/historical/results/serving-v3-performance-01.qualification.json).
This comparison uses different workloads, cache history and a substantially
modified reference. It measures the incremental effect of replay and caching.
The **20.06%, 7.34% and 40.33% figures have different references and metrics**;
adding or multiplying them would not estimate a measured overall improvement.

## Answer-quality evidence

The complete packaged configuration has not received a paired task-quality
comparison against its unchanged release base. In an eight-task coding check,
stock and the confidence/head configuration both passed 8/8; a configuration
that enabled reference rounding at the same fixed depth as stock passed 7/8.
Those tests did not cover the final combined fork. [Answer-quality evidence and
validation requirements](QUALITY.md) records the exact configurations, the
failed task and the additional checks needed. Component token equality and
software-test counts cannot substitute for that missing comparison.

## Serving limits and unsuccessful experiments

The prototype's optimized serving profile admitted one active request, queued
additional requests and capped total context at 131,072 tokens. Offline checks
at longer lengths compare finite model states or outputs; they do not demonstrate
that a chat server reliably handles those lengths. A separate long-context HTTP
test stopped after its recovery response contained an unexpected newline. The
cause was not established, and the remaining cases were not run. Optimized
million-token chat was therefore not validated.

Selected larger-cache experiments, numerical mismatches, small quality checks
and unsuccessful serving checks are indexed in the
[development experiment records](../archives/2026-10-18-preview/evidence/historical/README.md).
These records include incomplete or superseded implementations. Large activation
arrays and some supporting traces are omitted, as documented in that index.

## What was tested in the installable fork

The distributed build uses oMLX source based on public `v0.7.1rc1`, with the
retained changes integrated into normal package modules. Its public native
binaries are not byte-identical to those in the installed application used for
the performance experiments. Source continuity and a successful installation
cannot establish identical performance across these builds.

For package `0.7.1rc1+flash1017.1`, a fresh installation passed dependency checks
and served real requests, including an 8,319-token prompt, prefix-cache reuse,
cancellation/recovery and context-limit rejection. The default test suite had
**18,054 passes and 12 failures**; the same 12 failures reproduced on the
unchanged public release base. The [validation record](../validation/runtime-preview-01/README.md)
identifies the exact commit, wheel, checks and limitations.

That evidence supports a functional experimental distribution on the test
machine. A speedup for the installable build, equivalent answer quality and
performance on other machines remain unmeasured.
