# Installed v2 quality evaluation: 618 exact paired responses

The installed `0.7.1rc1+flash1017.2` fork reproduced the public oMLX
`v0.7.1rc1` baseline's **answer text and finish reason on all 618 fixed greedy
requests**, with **zero observed task regressions**. The corrected paired
report passed the acceptance plan frozen before v2 generation. This is a
bounded quality result for the identified checkpoint, software and machine.
It does not establish universal equivalence. The subsequent
[installed-wheel performance comparison](../performance-v2-01/README.md)
separately measured 7.9% higher generation throughput and 4.5% lower first-text
latency across eight workloads on this machine.

The [final report](pair-summary-final-02.json) records 15,900 passed integrity
checks, all 618 paired grades, no generation errors and no omitted outcomes.
The earlier v1 build's seven regressions and failed gate remain in the
[separate v1 evidence bundle](../quality-v1-01/README.md).

## Builds, requests and results

| Item | Baseline | V2 candidate |
|---|---|---|
| Package | `0.7.1rc1` | `0.7.1rc1+flash1017.2` |
| Commit | `7a5a3d75ce7463300bdfa5fedccd34348f60ff46` | `9280d25a1322c39f2f6713dc22b099a6e9dd41d1` |
| MTP | Native adaptive, maximum depth 3 | Native adaptive, maximum depth 3 |
| Configured prompt-chunk maximum | 2,048 tokens | 2,048 tokens |
| Engram prefetch admission | 16 MiB | 64 MiB |

Both used the unchanged `mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp`
checkpoint at revision `78456115d301eecf9fadcdcc81c0cf6f8b2c8eab`, on an M5
Ultra Mac Studio with 80 GPU cores and 256 GiB unified memory. This compares a
specific public release, not upstream `main`. [BUILD-IDENTITY.json](BUILD-IDENTITY.json)
records both wheel hashes, the model revision and the v2 profile. The wheel and
model weights are not included in this evidence bundle.

Generation used ordinary HTTP chat requests, temperature 0, top-p 1, top-k 0,
seed 431017, thinking disabled and normal end-of-sequence stopping. Coding and
math allowed 2,048 output tokens; the retrieval controls allowed 64. All requests
were frozen before model outputs. The v2 plan reused the exact original
baseline response file and receipt by hash; **baseline inference was not run
again for this comparison**. The baseline's coding answers and the new v2
answers were freshly graded under the same amended evaluator.

| Evaluation | Tasks | Baseline passed | V2 passed | Regressions | Identical text and finish reason |
|---|---:|---:|---:|---:|---:|
| HumanEval+ v0.1.10 | 164 | 148 | 148 | 0 | 164 |
| MBPP+ v0.2.0 | 378 | 299 | 299 | 0 | 378 |
| Fixed GSM8K test subset | 64 | 61 | 61 | 0 | 64 |
| Synthetic long-context controls | 12 | 11 | 11 | 0 | 12 |

There were also zero improvements: every scored task preserved its verdict.
The GSM8K failures were `gsm8k-test-0403`, `gsm8k-test-0454` and
`gsm8k-test-0823`; both builds failed `long-65536-late-2hop`. Shared failures
remain failures. Passing this comparison does not mean passing every task.

## Acceptance, truncation and the analysis correction

The [frozen v2 protocol](protocol.json) requires every pair, zero generation
errors, zero baseline-pass to candidate-fail tasks, no candidate-only length
stops and exact text plus finish-reason parity on all 618 requests. No quality
gain can offset a regression. The four baseline output-budget stops on
`Mbpp/56`, `Mbpp/260`, `Mbpp/430` and `Mbpp/468` also occurred identically in v2.
They were retained, not dropped or counted as completed answers.

The earlier protocol required absolute zero truncation, which the unchanged
baseline itself failed. That rule remains failed in both the historical v1
record and the v2 report's `original_acceptance` field. The separately declared
v2 conservation rule was frozen before v2 outputs; its pass does not rewrite
the original acceptance decision.

The first v2 report, [pair-summary-final-01.json](pair-summary-final-01.json),
failed one integrity check because its reporter interpreted draft width as a
fixed MTP depth. The frozen plan explicitly required adaptive MTP with maximum
depth 3 and `mtp_depth_fixed=false`. The
[analysis correction](analysis-correction-02.json) checks that declared mode
and ceiling. It changed only the reporter: the plan, requests, outputs, grades
and acceptance rules remained byte-for-byte unchanged. No inference or grading
was repeated for this correction. Both reporter versions, the failed report,
the corrected report and the correction's CPU fixture tests are preserved.

The corrected report binds reporter SHA
`87825e2b4aff2ce85af88e269136505a67a949181bdabf4a0a6adb34ad6492a5`
in `analysis-source-02/`. It is the exact source used for that report, even if
the repository's current general-purpose tool subsequently changes.

## Where the tasks and scores come from

Coding covers the complete pinned HumanEval+ and MBPP+ releases using EvalPlus
v0.3.1, commit `e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2`. A pass requires
both original and extended tests. The local macOS adapter retains the upstream
sanitizer and special oracles; this is not an unmodified upstream command-line
run or a leaderboard submission. Dataset releases, compressed bytes, hashes,
licenses, adapter source and dependency versions are included.

The [seeded evaluator amendment](evalplus/AMENDMENT.md), protocol
`flash1017-evalplus-macos-seeded-02`, replaces child-worker `-I` with
`-s -P -B` so `PYTHONHASHSEED=0` takes effect, and seeds the shared Python and
NumPy generators before each execution. Fresh sandboxed-worker probes, 12
negative controls and a full canonical sweep preceded model grading. The
[validation receipt](evalplus/validation-receipt.json) and both completion
receipts bind the actual executions. These seeds do not control independently
created random generators, explicit reseeding or every source of nondeterminism.

Canonical answers passed **540/542** tasks. Two pinned upstream behaviors
remain disclosed and in both full denominators:

- `HumanEval/32`: the special `find_zero` oracle skips a progress update before
  the official completeness check.
- `Mbpp/630`: the sanitizer removes a generator helper with `yield` and no
  `return`. The original synthetic control is included with its historical
  label; it is not a repaired model answer.

No model answer, sanitizer or oracle was repaired. The original unseeded
evaluator and its v1 scores remain unchanged in the v1 bundle. The seeded
adapter SHA is `24dc62dc3cdd23052de53d049057a5a6747b02115ea8a57dce59353704898098`.

The supplement uses 64 official GSM8K test questions selected by a fixed hash
rule, scored with an exact numeric-answer format. It is not a full GSM8K score.
The 12 authored one-hop and two-hop retrieval cases place facts early, midway
and late in approximately 16K/64K contexts. These are synthetic engineering
controls, not a standard long-context benchmark. Their
[frozen protocol](supplement/frozen-02/PROTOCOL.json), inputs, answer keys and
scorers are included unchanged.

## Evidence map and limits

| Files | Purpose |
|---|---|
| `protocol.json`, `protocol-02.json`, `requests-01.jsonl` | Frozen v2 plan, original baseline plan and all 618 requests. |
| `baseline-02/`, `candidate-01/` | Full saved response streams and generation receipts. |
| `harness.py`, `run_installed_quality.frozen-02.py` | V2 and original generation harnesses; request construction and configuration functions have matching ASTs. |
| `installation.json`, `BUILD-IDENTITY.json` | Candidate installation checks and explicit runtime/model identities. |
| `evalplus/*-grades.jsonl` and their metadata | Full canonical, baseline and candidate verdicts, extracted programs, input hashes and evaluator provenance. |
| `evalplus/candidate-responses-frozen.jsonl`, `candidate-response-freeze.json` | Exact response copy used by the seeded candidate grader and its capture receipt. |
| `evalplus/AMENDMENT.md`, `PROTOCOL-original.md`, `adapter-source/`, probes and controls | Exact original protocol, amended execution rules, frozen adapters and validation evidence. |
| `supplement/`, `*-supplement-01.jsonl`, `supplement-scores-01.json` | Frozen math/retrieval tasks, extracted responses and all paired scores. |
| `analysis-source/`, `analysis-source-02/`, both final reports, `analysis-correction-02.json` | Original and corrected analysis, with the failed intermediate report retained. |
| `COPY-PROVENANCE.json`, `MANIFEST.json` | Original workspace locations, unchanged copied-file hashes and complete bundle inventory. |

Original records retain their execution-time paths. The table and copy map
locate those same bytes here; embedded laboratory paths are not portable
commands. The full pinned EvalPlus checkout, Python environments, model
weights, caches and secret-bearing settings are not distributed in this bundle.

The report verifies saved request hashes, versions, native assets and dependency
agreement. Candidate installation separately checked 752 package files against
the wheel. Original baseline receipts do not hash every installed Python file.
The report's settings reconstruction uses hashes only here: private settings
were excluded, so the bundle does not independently repeat that portion of the
original audit. A file-integrity check is not an external attestation.

Both harnesses requested a 224 GiB memory/wired guideline, but stock startup
raised its live wired limit to about 243.2 GiB and the candidate profile reset
it to 224 GiB before loading. Common requested values do not prove identical
live memory limits. This does not alter the observed answer comparison.

The quality result itself provides no timing claim. A subsequent HTTP
performance attempt was interrupted after 16 of 32 measured requests when the
original server reloaded another model. That failed comparison remains intact
with no aggregate. The full replacement run completed all 32 requests and
qualified its 16 matched pairs. Its
[separate evidence bundle](../performance-v2-01/README.md) records the measured
7.9% generation-throughput gain and 4.5% first-text latency reduction, including
block variability and output-budget limits. Historical prototype percentages
do not transfer to this build.

## Verify saved files without running answers

From the research repository:

```sh
python3 -I -B tools/verify_quality_evidence.py --folder validation/quality-v2-01
python3 -I -B validation/quality-v2-01/analysis-source-02/test_summarize_installed_quality.py
```

The first command checks the manifest; the second runs CPU-only analysis
fixture tests. Neither runs model inference or generated answer code. Repeating
model inference or executing the coding tests requires the pinned runtime,
checkpoint or isolated evaluator described by the preserved protocols. Exact
observed parity on this finite suite does not prove equality for arbitrary
prompts, sampling settings, contexts or hardware.
