# DeepSeek-V4.1-Flash-1017 on Apple Silicon

This project investigates how to reduce the time spent processing prompts and
generating tokens for **DeepSeek-V4.1-Flash-1017 in oMLX**, an inference server
built on Apple's MLX framework. It uses the existing
[mlx-community oQ2.5e-MTP checkpoint](https://huggingface.co/mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp/tree/78456115d301eecf9fadcdcc81c0cf6f8b2c8eab)
without changing its weights or quantization.

**Current result: the first installed fork failed its quality gate.** The
completed 618-task comparison against public oMLX `v0.7.1rc1` found seven
baseline-pass to candidate-fail tasks. Both builds passed 447/542 coding tasks,
but those equal totals concealed five regressions and five gains. The math
subset passed 61/64 in both arms; synthetic long-context controls fell from
11/12 to 10/12, with two regressions and one gain. The
[complete evidence bundle](validation/quality-v1-01/README.md) preserves every
outcome and the failed acceptance decision.

A revised local build, `0.7.1rc1+flash1017.2` at commit
`9280d25a1322c39f2f6713dc22b099a6e9dd41d1`, uses a conservative profile intended
to preserve stock arithmetic. Its complete quality qualification is pending;
it has no measured speedup. [Answer-quality evidence](docs/QUALITY.md) explains
the original failure, evaluator limits and the revised acceptance requirements.

Separately, a historical prototype on one **M5 Ultra Mac Studio with 80 GPU
cores and 256 GiB unified memory** reduced geometric-mean **time to first token
by 20.06%: 19.554 → 15.631 seconds** across eight workloads. That comparison
used the installed application's embedded environment, loaded models and native
speculative decoding in the reference; it excluded model loading and HTTP
transport. Sampled peak memory rose **7.75 GiB**, and one 64K code workload
became **2.32% slower**. This is an archived prototype result, not a measured
speedup for either installed fork build.

![Time to first token for the benchmark prototype and two oMLX configurations](archives/2026-10-18-preview/evidence/overall/review/figures/first-token.png)

## Start here

- **[Research article](docs/ARTICLE.md):** the execution changes, the numerical
  problems they exposed, and the complete comparison.
- **[Methods and results](docs/RESULTS.md):** exact configurations, workload
  sources, individual component studies, failed checks and limits on the claims.
- **[Installed v1 quality result](validation/quality-v1-01/README.md):** all 618
  paired outcomes, seven regressions and the failed quality gate.
- **[Historical v1 functional validation](validation/runtime-preview-01/README.md):**
  installation checks, software-test failures and real HTTP requests.
- **[Answer-quality evidence](docs/QUALITY.md):** benchmark provenance, scoring
  limits and the revised build's pending qualification.
- **[Attribution](docs/ATTRIBUTION.md):** reused implementations, related work
  and this project's contributions.

## Benchmark code and installable software

The runtime fork and research repository preserve distinct versions and evidence:

| Artifact | What it contains | What was tested |
|---|---|---|
| **Benchmark prototype** | The source snapshot and experiment records in [`archives/2026-10-18-preview/`](archives/2026-10-18-preview/README.md), executed with the installed oMLX application's embedded environment. | The 20.06% overall latency result and the component comparisons described in [Methods and results](docs/RESULTS.md). |
| **Historical installed v1 fork** | Package `0.7.1rc1+flash1017.1`, commit `ad2f74058056e4cc7c0a13ddc76f96602e8ce092`. | Clean installation and serving checks; completed 618-task quality evaluation **failed**. No installed-wheel performance comparison. |
| **Current local v2 fork** | Maintained runtime source in `rubencu/omlx`, branch `flash1017-preview`; package `0.7.1rc1+flash1017.2`, commit `9280d25a1322c39f2f6713dc22b099a6e9dd41d1`. | Stock-arithmetic profile under qualification. Full quality and installed-wheel performance results pending. |

The fork integrates execution changes into ordinary package modules and bundles
native binaries from a public oMLX release. Those binaries
differ from the installed application's binaries used in the latency study.
Consequently, the prototype's measured speedup cannot be assigned to either
installed build. [`RUNTIME-LINK.json`](RUNTIME-LINK.json) is preserved as the
historical **v1** installation-validation identity; it does not identify the
current v2 candidate or certify quality preservation.

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
python3 -I -B archives/2026-10-18-preview/analysis/analyze.py
python3 -I -B -m unittest discover -s archives/2026-10-18-preview/analysis/tests -v
```

These commands check saved files against their manifests, recalculate the
overall comparison from recorded observations, and test the analyzer. They
require no model weights or GPU and do not run inference or contact a server.
An intact quality bundle still records a failed model-quality gate; checksum
verification is not a passing benchmark result or a fresh execution of answers.
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
