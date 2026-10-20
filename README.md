# DeepSeek-V4.1-Flash-1017 on Apple Silicon: measured gains with matched outputs

This project investigates how to reduce the time spent processing prompts and
generating tokens for **DeepSeek-V4.1-Flash-1017 in oMLX**, an inference server
built on Apple's MLX framework. It uses the existing
[mlx-community oQ2.5e-MTP checkpoint](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp/tree/78456115d301eecf9fadcdcc81c0cf6f8b2c8eab)
without changing its weights or quantization.

**Current result: 7.9% higher generation throughput and 4.5% less time to first
visible text across eight workloads on one 256 GiB M5 Ultra Mac Studio.** The
installed v2 fork also matched the baseline's answer text and finish reason on
all 618 quality requests, with zero observed task regressions.
Package `0.7.1rc1+flash1017.2`, commit
`9280d25a1322c39f2f6713dc22b099a6e9dd41d1`, passed its frozen quality plan
against public oMLX `v0.7.1rc1`. Both scored 148/164 on HumanEval+, 299/378 on
MBPP+, 61/64 on a fixed GSM8K subset and 11/12 on synthetic long-context
controls. Four identical output-budget stops remain included. This establishes
observed parity on the specified suite, not universal equivalence.
The [complete v2 evidence bundle](validation/quality-v2-01/README.md) contains
the requests, raw outputs, fresh seeded grades and analysis provenance.

| Installed v2 versus public oMLX `v0.7.1rc1` | Observed result |
|---|---|
| Generation throughput, eight mixed workloads | **7.9% higher**; improved in all 16 workload/block pairs |
| Client first-visible-text latency, same workloads | **4.5% lower**; improved in 11/16 pairs, with noticeable block variability |
| Six natural-source workloads | **8.0% higher generation throughput; 5.5% lower first-text latency** |
| Separate quality comparison | **618/618 identical answer texts and finish reasons; zero task regressions** |

The [performance evidence and offline reproduction guide](validation/performance-v2-01/README.md)
contains all 32 measured HTTP requests, four excluded warmups and the frozen
analysis. Results use equal-case geometric means over two process blocks.
Only three workloads finished normally within 512 tokens; completed-answer
latency applies to their six pairs. Other requests retain their throughput
and first-text measurements. These are bounded results on one machine, with
no confidence-interval or general hardware claim.

The first installed build, v1, had seven paired task regressions. Its
[failed result](validation/quality-v1-01/README.md) is preserved. V2 restores
stock arithmetic and draft-depth control while retaining weight reuse, row
caching and wider prefetch.
[Answer-quality evidence](docs/QUALITY.md) explains the revisions, acceptance
rules and evaluator limits.

## Start here

- **[Installed v2 performance](validation/performance-v2-01/README.md):** current
  results, all paired observations, limitations and CPU-only reproduction.
- **[Research article](docs/ARTICLE.md):** the execution changes, the numerical
  problems they exposed, and the complete comparison.
- **[Methods and results](docs/RESULTS.md):** exact configurations, workload
  sources, individual component studies, failed checks and limits on the claims.
- **[Installed v1 quality result](validation/quality-v1-01/README.md):** all 618
  paired outcomes, seven regressions and the failed quality gate.
- **[Installed v2 quality result](validation/quality-v2-01/README.md):** 618 exact
  paired responses, zero regressions and the passing bounded acceptance rule.
- **[Historical v1 functional validation](validation/runtime-preview-01/README.md):**
  installation checks, software-test failures and real HTTP requests.
- **[Answer-quality evidence](docs/QUALITY.md):** benchmark provenance, scoring
  limits and the revised build's completed qualification.
- **[Attribution](docs/ATTRIBUTION.md):** reused implementations, related work
  and this project's contributions.

## Benchmark code and installable software

The runtime fork and research repository preserve distinct versions and evidence:

| Artifact | What it contains | What was tested |
|---|---|---|
| **Benchmark prototype** | The source snapshot and experiment records in [`archives/2026-10-18-preview/`](archives/2026-10-18-preview/README.md), executed with the installed oMLX application's embedded environment. | The 20.06% overall latency result and the component comparisons described in [Methods and results](docs/RESULTS.md). |
| **Historical installed v1 fork** | Package `0.7.1rc1+flash1017.1`, commit `ad2f74058056e4cc7c0a13ddc76f96602e8ce092`. | Clean installation and serving checks; completed 618-task quality evaluation **failed**. No installed-wheel performance comparison. |
| **Current local v2 fork** | Maintained runtime source in `rubencu/omlx`, branch `flash1017-preview`; package `0.7.1rc1+flash1017.2`, measured commit `9280d25a1322c39f2f6713dc22b099a6e9dd41d1`. | All 618 quality responses match; zero task regressions. Qualified HTTP comparison: 7.9% higher generation throughput and 4.5% lower first-text latency over eight workloads. |

The fork integrates execution changes into ordinary package modules and bundles
native binaries from a public oMLX release. Those binaries
differ from the installed application's binaries used in the latency study.
Consequently, the prototype's measured speedup cannot be assigned to either
installed build. The historical prototype's 20.06% first-token reduction is
documented separately in [Methods and results](docs/RESULTS.md); the current
wheel's measurements above stand on their own.
[`RUNTIME-LINK.json`](RUNTIME-LINK.json) is preserved as the
historical **v1** installation-validation identity; it does not identify the
current v2 build. Its identity and bounded quality result are in
[`validation/quality-v2-01/BUILD-IDENTITY.json`](validation/quality-v2-01/BUILD-IDENTITY.json).

To run the model or change the runtime, use the companion oMLX fork. To examine
the method, recalculate results or propose analysis corrections, use this
research repository. [Repository layout and artifact identities](docs/REPOSITORIES.md)
explains where each kind of contribution belongs.

**Publication status:** these additions are prepared locally for review. The
preview branch and wheel have not been published to the existing
[rubencu/omlx fork](https://github.com/rubencu/omlx), and the proposed
`rubencu/flash1017-research` remote has not been created. The local runtime
checkout is the sibling `../omlx`; its `docs/flash1017-install.md` contains the
installation instructions. [Release review](docs/REVIEW.md) and the
[publication plan](docs/PUBLICATION-PLAN.md) are maintainer documents.

## Inspect and recalculate the evidence

With Python 3.9 or newer, from this repository:

```sh
python3 -I -B tools/verify_research.py
python3 -I -B tools/verify_runtime_preview.py
python3 -I -B tools/verify_quality_evidence.py --folder validation/quality-v1-01
python3 -I -B tools/verify_quality_evidence.py --folder validation/quality-v2-01
python3 -I -B tools/verify_quality_evidence.py --folder validation/performance-v2-01
python3 -I -B validation/performance-v2-01/reproduce.py --mode measurements --output /tmp/flash1017-performance-measurements.json
python3 -I -B archives/2026-10-18-preview/analysis/analyze.py
python3 -I -B -m unittest discover -s archives/2026-10-18-preview/analysis/tests -v
```

These commands check saved files against their manifests, recalculate the
overall comparison from recorded observations, and test the analyzer. They
require no model weights or GPU and do not run inference or contact a server.
Checksum verification establishes saved-file integrity, independently of the
recorded quality decision: v1 failed, while v2 passed its amended bounded rule.
It does not execute answers again or independently replicate model inference.
The source patch can also be checked offline using the system `patch` utility:

```sh
python3 -I -B archives/2026-10-18-preview/source/verify_source_kit.py
```

Recalculating saved results is one level of reproduction. Running the benchmark
again also requires the pinned checkpoint, hardware with sufficient memory,
the execution environment and the recorded request/cache sequence. See the
[environment and source requirements](archives/2026-10-18-preview/DEPENDENCIES.md).
The prototype's application-dependent launcher is not a standalone installer.

The archive contains raw observations, exact overall-comparison input tokens,
analysis code, source snapshots, selected unsuccessful experiments and the
notes written during development. Its internal filenames and notes preserve
laboratory terminology; the article and methods document provide the reader's
guide. [`ARCHIVE-INDEX.json`](ARCHIVE-INDEX.json) and
[`ARCHIVE-SHA256SUMS`](ARCHIVE-SHA256SUMS) bind its 337 files. Checksums detect
changes; they are not external replication or a publisher signature.

## License and credits

Original additions and repository documentation use [Apache-2.0](LICENSE).
Existing per-file MIT/Apache notices and fixture licenses remain applicable;
see [NOTICE](NOTICE) and [attribution](docs/ATTRIBUTION.md). Model weights are
not included. The contribution is integration, numerical and state repairs,
and measurements of a specific configuration, built on the credited model,
runtime and optimization techniques.
