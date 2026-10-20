# Answer quality: 618 exact paired responses in the revised build

**The complete installed v2 fork reproduced the baseline's answer text and
finish reason on all 618 tested requests, with zero observed task regressions.**
Its frozen acceptance plan passed. This establishes observed response and
task-score preservation for the tested checkpoint, settings and machine. It
does not establish universal quality equivalence. The quality test alone does
not measure a performance improvement; the separate timing result is below.

The qualified local build is `0.7.1rc1+flash1017.2`, commit
`9280d25a1322c39f2f6713dc22b099a6e9dd41d1`, using the `flash1017-v2` profile.
The first installed build, v1, failed with seven paired task regressions; those
results remain intact. The [v2 evidence guide](../validation/quality-v2-01/README.md)
contains the raw responses, grades, protocols and full provenance. Its
[final report](../validation/quality-v2-01/pair-summary-final-02.json) records
15,900 passed integrity checks, no generation errors and no missing outcomes.

## What was compared

The baseline was unchanged public oMLX `v0.7.1rc1`, commit
`7a5a3d75ce7463300bdfa5fedccd34348f60ff46`, with native adaptive multi-token
prediction (MTP) enabled. This is a specific release comparison, not upstream
`main`. The same original baseline response file was reused by hash in the v2
protocol. V2 inference was new; baseline inference was not repeated. Coding
answers from both arms were freshly graded under the same amended evaluator.

Both used the pinned
[mlx-community oQ2.5e-MTP checkpoint](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp/tree/78456115d301eecf9fadcdcc81c0cf6f8b2c8eab),
matching native assets and dependency versions, on an M5 Ultra Mac Studio with
80 GPU cores and 256 GiB unified memory. Requests used the normal HTTP chat
interface, one greedy answer per task, temperature 0, top-p 1, top-k 0, seed
431017, thinking disabled and ordinary end-of-sequence stopping. Output budgets
were 2,048 tokens for coding/math and 64 for long-context controls. All 618
requests were fixed before model generation.

The harness requested a 224 GiB memory/wired guideline in both arms, but this
did not prove identical live wired limits. Stock server startup raised its
wired limit to about 243.2 GiB; the candidate profile reset it to 224 GiB before
loading. Saved common requested values must not be represented as matching
effective memory limits. The original baseline receipt also does not hash
every installed Python source file; the v2 installation record separately
checks 752 installed package files against its wheel. The evidence guide
describes these provenance limits and the omission of private settings.

## V2 results and the acceptance rule

| Evaluation | Tasks | Baseline passed | V2 passed | Regressions | Exact text and finish reason |
|---|---:|---:|---:|---:|---:|
| HumanEval+ v0.1.10 | 164 | 148 | 148 | 0 | 164 |
| MBPP+ v0.2.0 | 378 | 299 | 299 | 0 | 378 |
| Frozen GSM8K test subset | 64 | 61 | 61 | 0 | 64 |
| Synthetic long-context controls | 12 | 11 | 11 | 0 | 12 |

Every scored verdict matched. Both builds missed the same three GSM8K
questions and `long-65536-late-2hop`; all shared coding failures remain in the
denominator. This is preservation of the observed baseline behavior, not a
claim that every answer was correct.

The v2 plan required all 618 pairs, no generation errors, zero baseline-pass
to candidate-fail tasks, no candidate-only length stops and exact answer text
plus finish reason on every request. Gains could not offset regressions.
Both builds hit the output budget on `Mbpp/56`, `Mbpp/260`, `Mbpp/430` and
`Mbpp/468`. These four identical truncations remain included and are not
described as completed answers.

The original v1 rule required absolute zero truncation, which the unchanged
baseline itself failed. The v2 conservation rule was declared separately and
frozen before v2 generation. The original rule's failure is retained in the
v1 record and the v2 report's `original_acceptance` field. Passing the amended
rule does not erase the original failed rule or the v1 model regressions.

The first v2 analysis report failed one integrity check because its reporter
interpreted draft width as a fixed MTP depth, despite the frozen protocol's
explicit adaptive setting. The
[analysis correction](../validation/quality-v2-01/analysis-correction-02.json)
checks the declared adaptive ceiling and mode. It changes no requests,
generation, grades or acceptance rules. Both reports and both frozen reporter
versions are retained. The corrected report is the passing result above.

## Where the tests come from

Coding covers the complete pinned HumanEval+ and MBPP+ releases through EvalPlus
v0.3.1, commit `e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2`. A task passes only
when both its original and additional tests pass. The local macOS adapter uses
the unchanged upstream sanitizer and oracles. This is not an unmodified
EvalPlus command-line run or a leaderboard submission. Dataset, evaluator and
adapter hashes and licenses are preserved with the
[original coding protocol](../validation/quality-v2-01/evalplus/PROTOCOL-original.md)
and [seeded amendment](../validation/quality-v2-01/evalplus/AMENDMENT.md).

The GSM8K supplement contains 64 questions from the official test split,
selected by a fixed hash rule before outputs. Answers use the frozen exact
numeric-answer format. This is a subset result, not a full GSM8K score. The
12 long-context cases are authored one-hop and two-hop retrieval controls at
approximately 16K and 64K tokens, with facts at early, middle and late positions.
They are engineering checks, not RULER or another standard long-context
capability benchmark. The
[supplement protocol](../validation/quality-v2-01/supplement/frozen-02/PROTOCOL.json)
defines selection, tokenization and scoring.

## Grader validation and limitations

Generated programs and canonical references ran in fresh macOS sandboxed
workers with resource monitoring and deadlines. The supervisor did not execute
answer code. Fresh isolation probes and 12 deliberately wrong or nonterminating
controls passed their checks before the seeded comparison. This is a tested
local evaluation arrangement, not an adversarially hardened judging service.

The full canonical sweep passed **540/542** tasks. Two failures are documented
behavior in the pinned upstream evaluator:

- `HumanEval/32`: the special `find_zero` oracle skips a progress update, then
  the official completeness check rejects the result.
- `Mbpp/630`: the sanitizer removes a generator helper containing `yield` but
  no `return`. A separately labeled original synthetic control confirms the cause.

Both remain in the full denominators for both arms. No model answer was repaired
and no sanitizer or oracle was changed. These limitations do not establish
model errors on those two tasks.

The original v1 worker used Python `-I`, which ignores `PYTHONHASHSEED`; its
verdicts have a documented repeatability limitation. V2 uses the separately
identified `flash1017-evalplus-macos-seeded-02` protocol. Its workers run with
`-s -P -B`, honor `PYTHONHASHSEED=0` and seed the shared Python/NumPy generators
before each execution, while preserving isolation restrictions. Both arms were
freshly graded with this same frozen adapter. This does not control all possible
randomness in generated programs. The original source and grades are unchanged,
and exact raw-answer parity is separate evidence from executable test results.

## Why the v2 runtime differs from v1

The v2 profile restores stock attention behavior, configured prompt chunks of
at most 2,048 tokens and native adaptive MTP with maximum depth 3. Dependency
replay and confidence scheduling are disabled in this production profile.
It retains vocabulary-weight reuse, raw row caching and wider prefetch.

The implementation argument is narrow: head rows 2–5 share weight reads while
retaining the original FP32 accumulation and reduction order; row 1 and rows
6 or greater use stock paths. Row caching retains raw bytes, compares complete
row-ID tags, checks bounds and restores request order. The wider prefetch budget
changes read admission with the same ID checks and native decode. These arguments
assume immutable weights and correct lifecycle draining. The 512 MiB retained
row-cache allowance excludes temporary allocations; stock memory-pressure logic
can reduce the configured 2,048-token chunk maximum. The claims apply to the
pinned v2 profile, not optional nonstock numerical settings elsewhere in the fork.

The implementation review and 618 exact observed responses support the bounded
claim together. Finite tests cannot prove equivalence for arbitrary prompts,
stochastic decoding, contexts, model revisions or hardware.

## The original v1 failure remains part of the record

The complete installed v1 wheel, `0.7.1rc1+flash1017.1`, commit
`ad2f74058056e4cc7c0a13ddc76f96602e8ce092`, completed all 618 requests without
generation errors and passed evidence-integrity checks, but failed its quality
gate. The [v1 bundle](../validation/quality-v1-01/README.md) retains every outcome.

| Evaluation | Baseline passed | V1 passed | Regressions | Improvements |
|---|---:|---:|---:|---:|
| HumanEval+ | 148/164 | 148/164 | 1 | 1 |
| MBPP+ | 299/378 | 299/378 | 4 | 4 |
| GSM8K subset | 61/64 | 61/64 | 0 | 0 |
| Synthetic long-context controls | 11/12 | 10/12 | 2 | 1 |

Equal coding totals of 447/542 concealed five regressions: `HumanEval/93`,
`Mbpp/103`, `Mbpp/265`, `Mbpp/301` and `Mbpp/626`. The long-context regressions
were `long-16384-early-2hop` and `long-65536-middle-2hop`: the candidate returned
an intermediate key instead of the requested final value. These were correctly
formatted wrong answers. Five coding gains and one long-context gain did not
offset the seven losses. Only 43/618 pairs had identical text and finish reason.

The original zero-truncation rule also failed: four baseline length stops and
three v1 stops, including candidate-only `Mbpp/125`. These records and the
original evaluator remain distinct from the v2 result.

Earlier eight-task development fixtures and bounded component studies did not
test the complete installed wheel. The 10,240 matched token positions in the
head-kernel study and 1,536 in the replay/cache study used references already
containing other changes. [Methods and results](RESULTS.md) identifies their
scope; they do not override the later complete-build results.

## The separate performance result

The same installed v2 wheel subsequently completed its HTTP performance
qualification: **7.9% higher generation throughput and 4.5% less time to first
visible text** across eight workloads and two process blocks. All 16 timing
pairs matched text, finish reason and prompt/completion counts. Generation
improved in every pair; first-text latency improved in 11/16 and varied across
blocks. Only three normal-stop cases, six pairs, qualify for completed-answer
latency; budget-limited requests remain in throughput and first-text metrics.

The [performance bundle](../validation/performance-v2-01/README.md) contains the
complete result, exact identities and offline analysis. Its earlier interrupted
16-request attempt remains failed with no aggregate; the full comparison was
rerun unchanged. The historical prototype's 20.06% result is separate. Timing
comparability does not establish general quality beyond the independently
reported finite quality suite.
