# Answer quality: the first installed build failed

**The complete v1 fork did not preserve the baseline's task outcomes.** Its
paired evaluation is finished: all 618 requests have saved outputs and grades,
the evidence-integrity checks passed, and the frozen quality gate failed.
There were **seven baseline-pass to candidate-fail tasks**. Improvements on
other tasks do not cancel those regressions.

The current local candidate is a different build, `0.7.1rc1+flash1017.2`, at
commit `9280d25a1322c39f2f6713dc22b099a6e9dd41d1`. It uses a more conservative
profile intended to retain stock arithmetic. Its 618-task qualification is
**pending**, and it has no measured performance result. The historical
prototype's 20.06% first-token latency reduction applies to neither installed
build without a new measurement.

## What was compared

The baseline was the unchanged public oMLX `v0.7.1rc1` release, commit
`7a5a3d75ce7463300bdfa5fedccd34348f60ff46`, with native adaptive multi-token
prediction (MTP) enabled. The candidate was the complete installed v1 wheel,
version `0.7.1rc1+flash1017.1`, commit
`ad2f74058056e4cc7c0a13ddc76f96602e8ce092`.

Both used the same pinned
[mlx-community oQ2.5e-MTP checkpoint](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp/tree/78456115d301eecf9fadcdcc81c0cf6f8b2c8eab),
native assets and dependency environment on the 256 GiB M5 Ultra machine.
Requests went through the normal HTTP chat interface. The frozen settings used
one greedy answer per task, temperature 0, top-p 1, seed 431017, thinking
disabled, ordinary end-of-sequence stopping, and fixed output budgets: 2,048
tokens for coding/math and 64 for the long-context controls.

The harness requested a 224 GiB memory/wired guideline in both arms, but this
was not proof of identical live wired limits. Stock server startup raised its
wired limit to about 243.2 GiB; the candidate profile reset it to 224 GiB before
loading. The saved receipts' common requested value must not be presented as
a verified match of effective memory limits.

The [complete evidence bundle](../validation/quality-v1-01/README.md) contains
the protocol, requests, raw answers, exact build identities, grading records
and [paired report](../validation/quality-v1-01/pair-summary-final-02.json).
This compares a specific public release, not upstream `main`.

## Results and paired regressions

| Evaluation | Tasks | Baseline passed | V1 passed | Baseline pass → v1 fail | Baseline fail → v1 pass |
|---|---:|---:|---:|---:|---:|
| HumanEval+ v0.1.10 | 164 | 148 | 148 | 1 | 1 |
| MBPP+ v0.2.0 | 378 | 299 | 299 | 4 | 4 |
| Frozen GSM8K test subset | 64 | 61 | 61 | 0 | 0 |
| Synthetic long-context controls | 12 | 11 | 10 | 2 | 1 |

The coding total was **447/542 in both arms**, but five previously passing
coding tasks failed in v1: `HumanEval/93`, `Mbpp/103`, `Mbpp/265`, `Mbpp/301`
and `Mbpp/626`. Five different tasks improved. An unchanged aggregate score
does not establish preserved behavior or satisfy the zero-regression rule.

The two long-context regressions were `long-16384-early-2hop` and
`long-65536-middle-2hop`. In each case the candidate returned the intermediate
lookup key instead of following it to the requested value. These were wrong
answers in the required format, not formatting-only failures or truncations.
The candidate improved on `long-65536-late-2hop`; that gain does not erase the
two losses. Both arms missed the same three GSM8K questions.

All 618 responses were retained in each arm, with no HTTP generation errors.
Only **43/618 pairs had identical answer text and finish reason**. Different
wording need not be wrong, but v1 did not simply reproduce every tested
baseline answer faster.

The original gate also required zero length stops. The baseline hit its output
budget on four coding tasks and v1 on three; `Mbpp/125` was a candidate-only
length stop. This prerequisite failed independently of the seven task
regressions. The failed rule and every affected task remain in the report.

## Where the tests come from

Coding uses the full pinned HumanEval+ and MBPP+ releases through EvalPlus
v0.3.1, commit `e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2`. A task passes only
when both its original and additional tests pass. This is a local macOS
adapter using the unchanged upstream sanitizer and oracles; it is not the
unmodified EvalPlus command-line program or a leaderboard submission. Dataset,
evaluator and adapter hashes are retained in the
[coding protocol](../validation/quality-v1-01/evalplus/PROTOCOL.md).

The GSM8K supplement contains 64 questions from the official test split,
selected by a fixed hash rule before model outputs. Answers are scored with
the frozen exact numeric-answer format. It is a subset result, not a score on
the entire GSM8K benchmark. The 12 long-context cases are authored one-hop and
two-hop retrieval controls at approximately 16K and 64K tokens, with facts at
early, middle and late positions. They are engineering checks, not RULER or a
standard long-context capability benchmark. The
[supplement protocol](../validation/quality-v1-01/supplement/frozen-02/PROTOCOL.json)
defines the selection, tokenization and scorers.

## Grader validation and limitations

Generated programs and canonical reference solutions ran in fresh macOS
sandboxed workers with resource monitoring and deadlines. The supervisor did
not execute answer code. Isolation probes and 12 deliberately wrong or
nonterminating controls passed their rejection checks. This protects the host
within the tested rules; it is not an adversarially hardened judging service.
The exact isolation and memory-limit qualifications are in the coding protocol.

The full canonical-solution check passed **540/542** tasks. Two failures are
documented behavior in the pinned upstream evaluator:

- `HumanEval/32`: the special `find_zero` oracle skips a progress update, then
  the official completeness check rejects the result.
- `Mbpp/630`: the official sanitizer removes a generator helper containing
  `yield` but no `return`. A separate synthetic control confirms the cause.

Both tasks stay in the full denominators for both arms. The sanitizer and
oracles were not changed, and no model answer was repaired. These failures
are limitations of this scoring protocol, not evidence of model errors on
those two tasks.

The original worker used Python `-I`, which ignores `PYTHONHASHSEED`. Its
per-process string hashing was therefore not deterministic, as the saved
[startup observation](../validation/quality-v1-01/evalplus/hash-seed-observation.json)
shows. The coding verdicts are the actual original executions, with zero
recorded infrastructure errors or outer phase timeouts; repeatability is
limited by this interpreter behavior and any randomness in generated code.
A later amendment uses a separately identified seed-controlled worker and
fresh grading. It does not overwrite this original protocol, change its
scores, or turn its failed gate into a pass.

## The revised build must establish its own result

The v2 profile restores stock 2,048-token prompt chunks, native adaptive MTP
with maximum depth 3, and stock attention rounding. It limits vocabulary-weight
reuse to the existing specialized row range and keeps dependency replay and
confidence scheduling out of the production profile. Retained row caching,
prefetch and kernel changes still require verification in the combined build.
Calling the profile conservative is a design description, not a result.

The implementation argument is narrow: head rows 2–5 share weight reads while
retaining the original FP32 accumulation and reduction order; row 1 and rows
6 or greater use stock paths. Row caching retains raw bytes, compares complete
row-ID tags, checks bounds and restores request order. The wider prefetch
budget changes read admission, with the same ID checks and native decode.
These arguments assume immutable weights and correct lifecycle draining.
The 512 MiB row-cache allowance excludes temporary allocations, and stock
memory-pressure logic can reduce the configured 2,048-token chunk maximum.
They apply to the pinned v2 profile, not optional nonstock numerical settings
elsewhere in the fork, and do not prove universal equivalence.

The separately declared v2 acceptance plan requires all 618 pairs, verified
evidence, zero baseline-pass to candidate-fail tasks, no candidate-only length
stops, and exact answer-text plus finish-reason parity on every request.
Existing baseline truncations remain disclosed; the original absolute
zero-truncation gate remains failed. The revised qualification is pending;
startup failures or partial progress are not completed results.

Only after that gate passes will a new installed-wheel HTTP performance run
measure first visible text, prompt-processing throughput and generation
throughput. Completed-answer latency will be compared only for identical
nontruncated answers. No historical speedup will be assigned to v2.

Finite tests cannot establish quality for every possible request. A passing
future result would support exact observed parity and task-score preservation
on the specified suite, together with the implementation's numerical argument.
It would not prove universal equivalence.

## How the older checks fit

Before the installed-wheel evaluation, eight authored coding tasks produced
8/8 for a stock fixed-depth-2 configuration and 7/8 after enabling reference
attention rounding. The failure was `longest_unique`, whose generated program
moved a sliding-window boundary backwards. Later component configurations
passed those eight tasks. They remain development fixtures, but they did not
evaluate the complete installed build.

Likewise, 10,240 matching output-token positions in the head-kernel study and
1,536 in the replay/cache study compare against references already containing
other changes. Their bounded numerical evidence and the large software-test
counts do not supersede the complete v1 quality failure. [Methods and
results](RESULTS.md) identifies each historical comparison.
