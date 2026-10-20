# Answer quality: 618 exact paired responses

Package `0.7.1rc1+flash1017.2` reproduced public oMLX `v0.7.1rc1`'s **answer
text and finish reason on all 618 fixed greedy requests**, with **zero observed
task regressions**. Every scored task kept its baseline verdict. This is an
output-preservation result for the identified checkpoint and test suite, not
a claim that every answer is correct or that arbitrary requests are equivalent.

The [paired report](pair-summary-final-02.json) contains all responses' results.
The [performance study](../performance-v2-01/README.md) measures the same wheel
separately: 7.9% higher generation throughput and 4.5% lower first-text latency
across eight workloads.

## Builds and results

| Item | Baseline | Fork |
|---|---|---|
| Package | `0.7.1rc1` | `0.7.1rc1+flash1017.2` |
| Commit | `7a5a3d75ce7463300bdfa5fedccd34348f60ff46` | `9280d25a1322c39f2f6713dc22b099a6e9dd41d1` |
| Multi-token prediction | Native adaptive, maximum depth 3 | Native adaptive, maximum depth 3 |
| Configured prompt-chunk maximum | 2,048 tokens | 2,048 tokens |
| Engram prefetch admission | 16 MiB | 64 MiB |

Both used the unchanged `mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp`
checkpoint at revision `78456115d301eecf9fadcdcc81c0cf6f8b2c8eab`, on an M5
Ultra Mac Studio with 80 GPU cores and 256 GiB unified memory.
[BUILD-IDENTITY.json](BUILD-IDENTITY.json) records the wheel hashes and model
identity. The baseline is the named public release, not upstream `main`.

| Evaluation | Tasks | Baseline passed | Fork passed | Regressions | Identical responses |
|---|---:|---:|---:|---:|---:|
| HumanEval+ v0.1.10 | 164 | 148 | 148 | 0 | 164 |
| MBPP+ v0.2.0 | 378 | 299 | 299 | 0 | 378 |
| Fixed GSM8K test subset | 64 | 61 | 61 | 0 | 64 |
| Synthetic long-context controls | 12 | 11 | 11 | 0 | 12 |

## Generation and acceptance

Requests used ordinary HTTP chat, temperature 0, top-p 1, top-k 0, seed 431017,
thinking disabled and normal end-of-sequence stopping. Coding and maths
allowed 2,048 output tokens; retrieval controls allowed 64. Requests and the
acceptance plan were fixed before the fork's outputs were generated.

The [protocol](protocol.json) reuses a complete baseline response file and
receipt verified by hash; **baseline inference was not repeated**. Both sets
of coding answers were freshly graded with the same seeded evaluator.
Acceptance requires every response pair, zero generation errors, zero
baseline-pass to fork-fail tasks, no additional output-budget stops, and exact
text plus finish-reason parity on every request. Improvements cannot offset
regressions.

Four coding responses reached the output budget identically in both builds:
`Mbpp/56`, `Mbpp/260`, `Mbpp/430` and `Mbpp/468`. They remain in the results
and are not counted as completed answers. The separate absolute no-truncation
rule retained in the report remains failed. The
[acceptance and analysis provenance](../../docs/EVIDENCE-PROVENANCE.md#quality-protocol-and-analysis)
explains the report's acceptance fields and source identities.

## Task sources and scoring

Coding covers the complete pinned HumanEval+ and MBPP+ releases with EvalPlus
v0.3.1, commit `e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2`. A pass requires
both original and extended tests. A local, isolated macOS adapter retains the
upstream sanitizer and special oracles. This is not an unmodified upstream
CLI execution or leaderboard submission.

Before grading, the evaluator passed sandbox probes and 12 negative controls.
Its canonical-answer sweep passed **540/542** tasks. Two pinned evaluator
behaviors remain in both model-score denominators:

- `HumanEval/32`: the special `find_zero` oracle skips a progress update before
  the official completeness check.
- `Mbpp/630`: the sanitizer removes a generator helper with `yield` and no
  `return`.

Neither model answers nor evaluator oracles were repaired. Workers use a fixed
Python hash seed and seed the shared Python and NumPy random generators.
Independently created generators, explicit reseeding and other nondeterminism
are outside those controls. The [evaluator specification](evalplus/AMENDMENT.md)
and [validation receipt](evalplus/validation-receipt.json) identify the exact
execution setup.

The maths supplement is 64 official GSM8K test questions selected by a fixed
hash rule and scored by exact numeric answer. It is not a full GSM8K score.
The 12 authored one-hop and two-hop retrieval controls place facts early,
midway and late in approximately 16K/64K contexts. They are engineering
controls rather than a standardized long-context benchmark. Their
[protocol](supplement/frozen-02/PROTOCOL.json), inputs, answer keys and scorers
are included.

## Inspect and verify the evidence

| Evidence | Location |
|---|---|
| Requests and protocol | `requests-01.jsonl`, `protocol.json` |
| Raw baseline and fork responses | `baseline-02/`, `candidate-01/` |
| Coding grades, extracted programs and evaluator source | `evalplus/` |
| Maths and retrieval tasks and scores | `supplement/`, `supplement-scores-01.json` |
| Paired report and its exact analysis source | `pair-summary-final-02.json`, `analysis-source-02/` |
| Package verification and build identities | `installation.json`, `BUILD-IDENTITY.json` |
| Copied-file provenance and checksums | `COPY-PROVENANCE.json`, `MANIFEST.json` |

From the research repository:

```sh
python3 -I -B tools/verify_quality_evidence.py --folder validation/quality-v2-01
python3 -I -B validation/quality-v2-01/analysis-source-02/test_summarize_installed_quality.py
```

These commands verify saved files and run CPU-only analysis fixtures. They do
not run inference or execute model-generated programs. Repeating those steps
requires the pinned model/runtime or isolated evaluator described by the
protocols. The full EvalPlus checkout, Python environments, model weights,
caches and private server settings are not included.

The report checks saved request hashes, versions, native assets and dependency
agreement. Installation separately checked 752 candidate package files against
the wheel; baseline generation receipts do not hash every installed Python
file. Private settings were excluded, so the bundle cannot independently
repeat that portion of the original settings audit. Checksums do not establish
independent replication.

Both harnesses requested 224 GiB memory/wired limits. Stock startup raised its
live wired limit to about 243.2 GiB, while the fork set it to 224 GiB before
loading. The result therefore preserves the builds' respective memory policies;
it does not establish identical live memory limits.
