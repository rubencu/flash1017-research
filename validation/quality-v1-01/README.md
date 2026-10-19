# Installed v1 quality evaluation: failed

This bundle records the completed comparison of the first installed Flash1017
fork against its unchanged public oMLX release base. **The evidence is complete
and its integrity checks passed; the quality acceptance test failed.** All 618
requests in each arm have saved outputs and paired task grades. Seven tasks
passed in the baseline and failed in the candidate. Gains on other tasks do
not offset these regressions.

| Suite | Baseline passes | Candidate passes | Regressions | Improvements |
|---|---:|---:|---:|---:|
| HumanEval+ | 148/164 | 148/164 | 1 | 1 |
| MBPP+ | 299/378 | 299/378 | 4 | 4 |
| GSM8K frozen test subset | 61/64 | 61/64 | 0 | 0 |
| Synthetic long-context controls | 11/12 | 10/12 | 2 | 1 |

The equal coding total, 447/542, conceals five changed task failures and five
gains. The long-context controls add two regressions and one gain. These are
different kinds of tests, so this document does not combine them into one
overall capability score. The controls are authored retrieval checks, not a
standard long-context benchmark.

## Builds and generation

- **Baseline:** public oMLX `0.7.1rc1`, source
  `7a5a3d75ce7463300bdfa5fedccd34348f60ff46`; wheel SHA256
  `b3e12950a607991a1feb833b1bdc0b92fe41aeb73656ca5373db03d0d87b71c7`.
- **Candidate:** `0.7.1rc1+flash1017.1`, source
  `ad2f74058056e4cc7c0a13ddc76f96602e8ce092`; wheel SHA256
  `b5c333adc078c28d0950e925af55d5da628d0915e0586bc6ac62f97d35400554`.
- **Checkpoint:** `mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp`, revision
  `78456115d301eecf9fadcdcc81c0cf6f8b2c8eab`, unchanged in both arms.

Generation used the normal installed HTTP server on an M5 Ultra machine with
256 GiB memory, matching dependencies/native assets, one greedy response per
task, temperature 0, seed 431017 and thinking disabled. Normal end-of-sequence
stopping was enabled. Coding and math had a 2,048-token output budget; the
long-context controls had 64. The complete request list and frozen parameters
are in [protocol-02.json](protocol-02.json) and
[requests-01.jsonl](requests-01.jsonl).

The receipts record a common requested 224 GiB memory/wired guideline, not
verified identical live limits: stock server startup raised its wired limit
to about 243.2 GiB, while the candidate profile reset it to 224 GiB before load.

There were no recorded generation errors. The baseline reached the output
budget on `Mbpp/56`, `Mbpp/260`, `Mbpp/430` and `Mbpp/468`; the candidate did so
on `Mbpp/125`, `Mbpp/260` and `Mbpp/430`. The original protocol allowed no
truncations, so that prerequisite failed in addition to the task regressions.
Only 43/618 pairs had identical answer text and finish reason.

The seven observed regressions were `HumanEval/93`, `Mbpp/103`, `Mbpp/265`,
`Mbpp/301`, `Mbpp/626`, `long-16384-early-2hop` and
`long-65536-middle-2hop`. The long-context failures returned an intermediate
key instead of the final value. They were correctly formatted wrong answers.
See [pair-summary-final-02.json](pair-summary-final-02.json) for every paired
outcome, gain, length stop and acceptance violation.

## Read the evidence

| Files | Purpose |
|---|---|
| `protocol-02.json`, `requests-01.jsonl` | Frozen build identities, sampling, task inventory and original acceptance rules. |
| `baseline-02/`, `candidate-02/` | Installed-runtime receipts and all raw responses. |
| `pair-summary-final-02.json` | Completed 618-pair integrity audit and failed acceptance result. |
| `evalplus/*grades*.jsonl` and metadata | Full coding outcomes and exact input/evaluator identities; canonical solutions are scored separately. |
| `evalplus/PROTOCOL.md`, `evalplus/adapter-source/`, `evalplus/datasets/`, `evalplus/licenses/` | Pinned public benchmark releases, local macOS execution adapter and attribution. |
| `evalplus/validation-receipt.json`, controls and probe | Canonical validation, deliberate failure controls and host-isolation checks. |
| `supplement/frozen-02/` | Preselected GSM8K questions, synthetic controls, private answer keys, scorer and frozen preparation source. |
| `supplement-scores-02.json` and arm supplement files | Supplemental outcomes and exact raw-response subsets. |
| `run_installed_quality.frozen-02.py`, `analysis/` | Generation and reporting source snapshots for inspecting how these records were produced. |

The `frozen-02` names identify preparation revisions, not a second model
candidate. This directory records the **v1** runtime. The historical
[`RUNTIME-LINK.json`](../../RUNTIME-LINK.json) also identifies that v1 build.
Neither file identifies the later v2 candidate.

## Scoring limits are part of this result

The coding adapter uses unchanged EvalPlus v0.3.1 sanitizer/oracles with the
complete HumanEval+ v0.1.10 and MBPP+ v0.2.0 releases. Each generated program
and canonical solution runs in an isolated macOS worker; the supervisor does
not execute answer code. The saved protocol explains the resource limits and
the limits of host isolation. This was a local adapter, not the unmodified
upstream CLI or a leaderboard submission.

Canonical validation passed **540/542**. The pinned `HumanEval/32` special
oracle skips a progress update that its final completeness check requires.
The pinned sanitizer removes the generator helper in `Mbpp/630` because it
contains `yield` but no `return`. Both tasks remain in both denominators and
the upstream behavior is unchanged. The synthetic positive control for the
generator is not a repaired model answer. These two failures do not establish
that the model itself answered those tasks incorrectly.

The original worker used `-I`, so Python ignored `PYTHONHASHSEED`; the
[startup observation](evalplus/hash-seed-observation.json) confirms randomized
per-process string hashing. These are the original observed verdicts, with
zero recorded infrastructure errors or outer phase timeouts, but deterministic
regrading is not guaranteed. A later, separately identified seed-controlled
protocol regrades the unchanged baseline and revised v2 candidate. Its results
do not replace the original grades or change this failed acceptance decision.

The original coding and validation receipts bind
[`evalplus/isolation-probe-v2.json`](evalplus/isolation-probe-v2.json), SHA256
`f8cef84da390b842f5542cc90f1985c63aaaa8d5ef19fae632cfe0856e3092ac`.
[`evalplus/isolation-probe-final.json`](evalplus/isolation-probe-final.json) is
another passed probe execution, with different bytes; it is not a filename
alias for the receipt-bound result. Both are retained. The probe's `v2`
filename is an isolation-test revision, not runtime v2.

## Verify the saved bundle

From the research repository, without model weights or a GPU:

```sh
python3 -I -B tools/verify_quality_evidence.py --folder validation/quality-v1-01
```

This checks the complete local file inventory, sizes and SHA256 hashes against
[MANIFEST.json](MANIFEST.json). A successful checksum verification means the
saved evidence is intact; it does not mean the model passed its quality gate,
rerun generated code, or independently reproduce the experiment.

The bundle preserves local execution paths and original source identities.
It does not include model weights, virtual environments, server authentication
state, the full upstream EvalPlus source checkout or every diagnostic attempt.
The copied coding protocol gives the pinned source/dependency setup for a
fresh isolated execution. Regrading can produce different verdicts under the
original interpreter behavior and must be labeled as a new execution.

The [current quality status](../../docs/QUALITY.md) describes the later
stock-arithmetic candidate and its pending qualification. The prototype's
historical 20.06% first-token speedup is not a measured result for this v1 wheel
or the revised build. Nothing in this bundle has been published.
