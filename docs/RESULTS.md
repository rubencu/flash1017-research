# Methods and results

The optimized oMLX package delivered **7.9% higher generation throughput and
4.5% lower client first-text latency** across eight DeepSeek-V4.1-Flash-1017
workloads on one M5 Ultra Mac Studio. All 16 workload/block pairs matched
response text, finish reason and token counts. A separate 618-request quality
evaluation matched every answer text and finish reason with zero observed
task regressions.

This document defines the builds, workload, timing and aggregation behind
those results. The [article](ARTICLE.md) explains the execution changes;
[Answer quality](QUALITY.md) gives the full grading method.

## Hardware, model and software

| Component | Identity |
|---|---|
| Hardware | Apple M5 Ultra Mac Studio, 80 GPU cores, 256 GiB unified memory |
| Public-release baseline | oMLX `v0.7.1rc1`, commit `7a5a3d75ce7463300bdfa5fedccd34348f60ff46` |
| Optimized package | `0.7.1rc1+flash1017.2`, commit `9280d25a1322c39f2f6713dc22b099a6e9dd41d1` |
| Model | `mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp` |
| Model revision | `78456115d301eecf9fadcdcc81c0cf6f8b2c8eab` |
| Runtime profile | `flash1017-v2` |

The [checkpoint](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp/tree/78456115d301eecf9fadcdcc81c0cf6f8b2c8eab)
weights and quantization were unchanged. Both installed builds used matching
Python, 110 dependency versions and 10 native assets. Exact wheel hashes,
installed source manifests, checkpoint metadata and tokenizer identities are
recorded in the [performance protocol](../validation/performance-v2-01/qualified/protocol.json)
and receipts. The reference is the specified public release; upstream `main`
and other model revisions were not measured.

| Execution setting | Public baseline | Optimized package |
|---|---|---|
| Attention behavior | Stock | Stock |
| Multi-token prediction | Native adaptive, maximum depth 3 | Native adaptive, maximum depth 3 |
| Configured prompt-chunk maximum | 2,048 tokens | 2,048 tokens |
| Vocabulary projection | Stock | Weight reads shared across rows 2–5, preserving per-row arithmetic |
| Engram row cache | Stock | 256 MiB per table, 512 MiB total retained raw bytes |
| Engram prefetch admission | 16 MiB | 64 MiB |

The measurements cover the combined package. They do not isolate the speedup
of an individual change. Native memory-pressure logic can reduce the configured
chunk maximum, and the retained row-cache allowance excludes temporary allocations.

## Workload and execution order

The custom latency suite uses synthetic code/prose and public source content
from CPython, curl, *Moby-Dick* and *Pride and Prejudice*. These inputs exercise
execution; they do not score programming ability or reading comprehension.

| Workload | Actual prompt tokens | Finish reason in both builds |
|---|---:|---|
| Synthetic code | 1,024 | Output limit, 512 tokens |
| Synthetic prose | 1,024 | Output limit, 512 tokens |
| CPython | 16,384 | Output limit, 512 tokens |
| curl | 16,384 | Output limit, 512 tokens |
| *Moby-Dick* | 16,383 | Normal stop |
| *Pride and Prejudice* | 16,384 | Normal stop |
| CPython | 65,536 | Output limit, 512 tokens |
| *Pride and Prejudice* | 65,536 | Normal stop |

Exact HTTP payloads and rendered token arrays were frozen before timing.
Corpus sources, selection and tokenization hashes are in the
[fixture provenance](../validation/performance-v2-01/preparation/fixture-provenance-draft-01.json).
The approximately 16K group includes the 16,383-token *Moby-Dick* input.

Each build ran in two fresh processes. Every process loaded the model, ran
one excluded representative 16K warmup, then all eight measured cases. Block 0
ran baseline then optimized; block 1 reversed both build and case order. This
produced **32 measured requests, four excluded warmups and 16 matched pairs**.
Concurrency was one request. Generation used temperature 0, top-p 1, top-k 0,
seed 431017, thinking disabled, ordinary end-of-sequence stopping and a
512-token maximum; warmups allowed 64 output tokens.

Prefix reuse was disabled through normal settings. Every response reported
zero cached prompt tokens. The separate Engram row cache retained its history
after warmup and through each process's measured sequence. Models stayed
loaded and operating-system file/page caches were not purged. This measures
warm loaded-model performance, excluding model loading. Counterbalancing
reduces order effects but does not eliminate them.

## Timing and aggregation

**Client first-text latency** starts immediately before sending the HTTP request
and ends when a streaming event delivers nonempty content. Role-only, empty
and usage-only events do not count. Transport is included.

**Prompt-processing throughput** and **generation throughput** use native
server-reported token rates. Those counters are rounded by the server and are
kept separate from client measurements. **Completed-answer latency** runs from
HTTP send to the final stream marker, and is reported only when both builds
produce identical text with a normal stop.

Each ratio compares the same workload and block. Rate ratios use
`optimized / baseline`; latency ratios use `baseline / optimized`, so a value
above 1 favors the optimized package. The geometric mean of the two block
ratios gives each case's result, then cases receive equal weight in a group.
The 32 requests are not treated as 32 independent workloads.

Rate increase is `100 × (ratio − 1)`. Latency reduction is
`100 × (1 − 1 / ratio)`. The all-eight latency ratio 1.0471920448284588
therefore corresponds to **4.507% less latency**.

## Performance results

| Group | Cases / pairs | Generation throughput | First-text latency | Prompt-processing throughput |
|---|---:|---:|---:|---:|
| All eight, mixed | 8 / 16 | **+7.91%** | **−4.51%** | +4.71% |
| Natural-source cases | 6 / 12 | **+8.01%** | **−5.52%** | +5.85% |
| Synthetic ≈1K | 2 / 4 | +7.62% | −1.40% | +1.36% |
| Natural ≈16K | 4 / 8 | +7.77% | −4.42% | +4.63% |
| Natural 64K | 2 / 4 | +8.47% | −7.68% | +8.33% |

The exact all-eight generation ratio is **1.0790916711693925**. All 16 pairs
matched text, finish reason and prompt/completion counts; outputs also matched
across blocks within each build. Every planned measurement is included.

| Case | Generation ratio, two-block geometric mean | First-text ratio, block 0 | First-text ratio, block 1 |
|---|---:|---:|---:|
| Synthetic code ≈1K | 1.0855 | 1.0316 | 1.0208 |
| Synthetic prose ≈1K | 1.0669 | 0.9954 | 1.0093 |
| CPython 16K | 1.0884 | 0.9488 | 1.0886 |
| curl 16K | 1.0900 | 1.0023 | 1.0030 |
| *Moby-Dick* ≈16K | 1.0593 | 1.0888 | 0.9639 |
| *Pride and Prejudice* 16K | 1.0735 | 1.3219 | 0.9969 |
| CPython 64K | 1.0874 | 1.1208 | 1.0259 |
| *Pride and Prejudice* 64K | 1.0820 | 0.9371 | 1.2777 |

Generation throughput improved in **16/16 pairs**. First-text latency improved
in **11/16**, with substantial block variation on some cases. The 64K novel
case was about 6.7% slower in one block and 21.7% faster in the other. All eight
per-case first-text geometric means favored the optimized package, but the
smaller latency gain is less consistent than the generation gain.

The three normal-stop workloads contribute **six pairs** to completed-answer
latency: *Moby-Dick* ≈16K and *Pride and Prejudice* at 16K and 64K. Their
geometric-mean latency was **7.60% lower**, ratio **1.0822150476315349**. The
other five workloads hit the output budget in both builds and remain in the
first-text/throughput tables. They do not contribute to completed-answer latency.

## Memory and scope

Both builds requested a 224 GiB startup allocator/wired guideline and an 8 GiB
allocator cache. Their effective wired policies differed: stock startup raised
its limit to about 243.2 GiB; the optimized profile set 224 GiB before loading.
This comparison includes those native profile behaviors and does not isolate
execution changes under identical live wired limits.

Read-only MLX counters showed cumulative allocator peaks differing by
81,611,036 bytes in block 0 and zero in block 1. Those peaks include loading
and warmup. They are not per-request peaks, whole-process RSS or total system
memory measurements.

Two temporal blocks on one machine provide a bounded comparison. They do not
support confidence intervals, p95/p99 latency, population significance or
extrapolation to other hardware. The workload covers single-request execution
at the listed prompt sizes; it does not measure concurrent-serving throughput
or longer contexts. Timing output equality makes these requests comparable;
the separate quality evaluation provides broader but still finite behavioral
evidence.

## Quality result

The quality evaluation retained all **618 paired requests and grades**, with
no generation errors. Every answer text and finish reason matched.

| Evaluation | Public baseline | Optimized package | Task regressions |
|---|---:|---:|---:|
| HumanEval+ v0.1.10 | 148/164 | 148/164 | 0 |
| MBPP+ v0.2.0 | 299/378 | 299/378 | 0 |
| Fixed GSM8K test subset | 61/64 | 61/64 | 0 |
| Synthetic 16K/64K retrieval controls | 11/12 | 11/12 | 0 |

Four shared output-budget stops remain included. The public baseline's saved
generation was reused by exact hash; both builds' coding answers were freshly
graded with the same seed-controlled local macOS adapter. Canonical validation
passed 540/542 because two pinned upstream evaluator behaviors also reject
canonical solutions. Both tasks remain in both denominators.
[Answer quality](QUALITY.md) explains the tasks, scoring, acceptance rule and
why the result establishes observed parity rather than universal equivalence.

## Evidence and reproduction

The [performance bundle](../validation/performance-v2-01/README.md) contains
the frozen protocol, all raw process receipts and responses, exact sources and
CPU-only reproduction commands. Saved arithmetic can be reproduced without
model weights; the complete source/native audit additionally requires the two
exact wheels. The [quality bundle](../validation/quality-v2-01/README.md) contains
all requests, outputs, grades and evaluator controls.

Detailed [evidence provenance](EVIDENCE-PROVENANCE.md) records the source and
audit history. Recalculating saved observations verifies their interpretation;
independent inference replication also requires the specified checkpoint,
hardware, environment and process/cache sequence.
