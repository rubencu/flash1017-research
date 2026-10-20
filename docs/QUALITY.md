# Answer quality: 618 exact paired responses

The optimized oMLX package reproduced the public-release baseline's **answer
text and finish reason on all 618 tested requests**, with **zero observed task
regressions**. All requests and grades are retained, with no generation errors.
This establishes observed response and task-score preservation for the tested
checkpoint, settings and machine. It does not prove equivalence for arbitrary
prompts, stochastic decoding or other hardware.

The measured package is `0.7.1rc1+flash1017.2`, commit
`9280d25a1322c39f2f6713dc22b099a6e9dd41d1`, using profile `flash1017-v2`.
The [evidence bundle](../validation/quality-v2-01/README.md) contains all raw
requests and responses, grades, exact identities and analysis sources.

## Comparison and generation settings

The baseline is public oMLX `v0.7.1rc1`, commit
`7a5a3d75ce7463300bdfa5fedccd34348f60ff46`, with native adaptive multi-token
prediction enabled. This is a comparison with that public release, not
upstream `main`. Baseline responses came from a saved generation run and were
reused by exact file hash; baseline inference was not repeated for this
comparison. Both builds' coding answers were freshly graded under the same
frozen evaluator.

Both used the unchanged
[mlx-community oQ2.5e-MTP checkpoint](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp/tree/78456115d301eecf9fadcdcc81c0cf6f8b2c8eab),
revision `78456115d301eecf9fadcdcc81c0cf6f8b2c8eab`, matching native assets and
dependency versions, on an M5 Ultra Mac Studio with 80 GPU cores and 256 GiB
unified memory. All 618 requests were fixed before model generation.

Requests used the ordinary HTTP chat interface, one greedy answer per task,
temperature 0, top-p 1, top-k 0, seed 431017, thinking disabled and normal
end-of-sequence stopping. Output limits were **2,048 tokens for coding/math**
and **64 tokens for synthetic retrieval controls**.

The harness requested a 224 GiB memory/wired guideline in both builds, but
effective wired limits differed: stock startup raised its limit to about
243.2 GiB, while the optimized profile set 224 GiB before loading. The result
covers those measured profiles rather than identical live memory policies.

## Results and acceptance

| Evaluation | Tasks | Baseline passed | Optimized passed | Regressions | Exact text and finish reason |
|---|---:|---:|---:|---:|---:|
| HumanEval+ v0.1.10 | 164 | 148 | 148 | 0 | 164 |
| MBPP+ v0.2.0 | 378 | 299 | 299 | 0 | 378 |
| Fixed GSM8K test subset | 64 | 61 | 61 | 0 | 64 |
| Synthetic long-context controls | 12 | 11 | 11 | 0 | 12 |

Every scored verdict matched, including the failures. Both builds missed the
same three GSM8K questions and `long-65536-late-2hop`; all coding failures
remain in the denominator. Preserving the baseline's behavior does not imply
that every answer is correct.

The acceptance rule was frozen before optimized-package generation. It requires
all 618 pairs, no generation errors, zero baseline-pass to optimized-fail tasks,
no additional output-budget stops, and exact answer-text plus finish-reason
parity on every request. An improvement on one task cannot offset a regression
on another. The package passed this rule.

Both builds reached the output limit on **`Mbpp/56`, `Mbpp/260`, `Mbpp/430` and
`Mbpp/468`**. These four identical truncations remain included. They are not
described as completed answers, and the comparison does not require every
baseline answer to be correct or complete. The full
[paired report](../validation/quality-v2-01/pair-summary-final-02.json) contains
all outcomes and 15,900 passed evidence-integrity checks.

## Task sources and scoring

Coding covers the complete pinned HumanEval+ and MBPP+ releases through EvalPlus
v0.3.1, commit `e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2`. A pass requires
both the original and extended tests. The local macOS adapter retains the
upstream sanitizer and special oracles. It is not an unmodified upstream
command-line run or a leaderboard submission. Dataset hashes, licenses and
exact adapter sources are preserved in the evidence bundle.

The GSM8K supplement contains 64 questions from the official test split,
selected by a fixed hash rule before model outputs. Responses are scored with
the frozen exact numeric-answer format. This is a subset result, not a score
on all of GSM8K.

The 12 authored retrieval controls place facts early, midway and late in
approximately 16K and 64K contexts. One-hop cases request a stored value;
two-hop cases require following an intermediate key to its value. They are
synthetic engineering checks, not RULER or another standard long-context
capability benchmark. Their [protocol and scorers](../validation/quality-v2-01/supplement/frozen-02/PROTOCOL.json)
define task selection, tokenization and exact answer checks.

## Executable grader validation

Generated programs and canonical references ran in fresh macOS sandboxed
workers with resource monitoring and deadlines. The supervisor did not execute
answer code. Isolation probes and 12 deliberately wrong or nonterminating
controls passed their checks before model grading.

The evaluator protocol, `flash1017-evalplus-macos-seeded-02`, launches workers
with `-s -P -B`, honors `PYTHONHASHSEED=0` and seeds the shared Python and NumPy
generators before each execution. Both builds use this same frozen adapter.
These measures do not control independently created random generators,
explicit reseeding or every source of program nondeterminism. Exact raw-answer
parity is separate evidence from executable test outcomes.

The full canonical-solution sweep passed **540/542** tasks. Two documented
behaviors in the pinned upstream evaluator also reject canonical solutions:

- **`HumanEval/32`:** the special `find_zero` oracle skips a progress update,
  then the official completeness check rejects the result.
- **`Mbpp/630`:** the sanitizer removes a generator helper containing `yield`
  but no `return`.

Both tasks remain in both full denominators. No model answer, sanitizer or
oracle was repaired. These two scoring limitations do not establish model
errors on those tasks. The evidence contains canonical outcomes, the validation
receipt, negative controls and a synthetic control for the generator-helper case.

## What supports the preservation claim

The runtime's vocabulary kernel shares weight reads across rows 2–5 while
retaining the stock FP32 accumulation and reduction order. Other row counts
use stock paths. Row caching stores raw bytes, checks full IDs and bounds,
and restores requested order. Wider prefetch changes read admission while
retaining the same identity checks and native decode. These implementation
arguments assume immutable weights and correct lifecycle draining.

The observed 618 exact matches complement that argument for the identified
profile. They cannot prove equivalence for every request, sampling setting,
model revision or machine. The math subset and synthetic retrieval controls
also cover less than a broad capability evaluation.

Receipts verify saved request hashes, versions, native assets and dependencies.
The optimized installation separately checks 752 package files against its
wheel; the baseline generation receipt does not hash every installed Python
file. Private settings are excluded from the distributed evidence, so their
recorded hash audit cannot be independently repeated from the bundle alone.
Checksums establish saved-file identity, not external attestation or a fresh
model run. [Evidence provenance](EVIDENCE-PROVENANCE.md) documents the complete
audit record without changing the observations above.

The [separate performance comparison](RESULTS.md) measures throughput and
latency for the same package. Its matched timing outputs make those workloads
comparable; they do not enlarge the quality claim beyond the evaluated suite.
