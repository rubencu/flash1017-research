# DeepSeek-V4.1-Flash-1017 on a 256 GB Mac Studio

**Research preview — local publication-review copy.** This package contains the complete local experimental source, an overall comparison with installed oMLX, and the evidence behind it. The saved-results analysis is portable. The experimental inference runtime still depends on the original installed application's environment; this is **not a verified standalone installer**.

On one M5 Ultra Mac Studio with 80 GPU cores and 256 GiB memory, the complete configuration reduced geometric-mean time to first token from **19.554 s to 15.631 s: 20.06% less time** across our fixed suite. The baseline was untouched installed **oMLX 0.7.1rc1 with native adaptive MTP enabled**, using the same `mlx-community/DeepSeek-V4.1-Flash-1017-oQ2.5e-mtp` checkpoint. Current upstream main was not benchmarked.

![Time to first token, including workload variation](evidence/overall/review/figures/first-token.png)

The comparison contains eight workload definitions, two process blocks, three configurations and 48 measured requests, plus six excluded warmups. Models remained loaded; OS caches were not purged. Timings exclude loading and HTTP transport. The headline includes both code and configuration changes and does not assign the gain to any single patch.

**Read the tradeoffs with the result:** sampled peak process footprint increased from 208.35 to 216.10 GiB; the 64K CPython case's first-token time regressed 2.32%. The original fixed-512 useful-answer qualification failed because nine requests reached EOS early, three in each configuration. First-token measurements qualify separately. The 20.21% reduction in observed time to token 128 is explicitly a post-hoc analysis of the original runs. Generated outputs differ, so answer-quality equivalence and faster equivalent task completion are unestablished.

## Start here

- [Technical article](docs/ARTICLE.md): the result, numerical failure analysis and repairs.
- [Publication review](docs/REVIEW.md): the proposed destinations, exact contents and remaining decisions.
- [Complete comparison report](evidence/overall/review/report.txt): method, every case, both stock controls and limitations.
- [Source/dependency guide](DEPENDENCIES.md): all accepted changes, where the implementation lives, and what remains external.
- [Recalculate the saved results](analysis/README.txt): CPU-only analysis; no model download, MLX or GPU required.
- [Release notes](docs/RELEASE-NOTES.md), [upstream contribution plan](docs/UPSTREAM.md) and [attribution](docs/ATTRIBUTION.md).
- [First upstream patch packet](upstream/bf16-rows2-5/README.md): concrete BF16 rows 2–5 diff and draft PR; hardware validation remains pending.
- [Local verification records](validation/README.md): relocated analysis, 36 CPU tests and source/patch checks.

From the extracted package directory, with Python 3.9 or later:

```sh
python3 verify_package.py
python3 analysis/analyze.py
```

The first command checks the packaged file manifest. The second recalculates the observed results and checks the saved reference summaries. Neither executes the model. Checksums establish internal consistency, not a digital signature, external replication or proof of numerical correctness.

## What is included

The source snapshot includes the earlier changes as well as the later replay/cache work:

| Change | Scope |
|---|---|
| BF16 output-head weight reuse | Local implementation through six rows; rows 2–5 retain the old specialized arithmetic. Six rows replace a numerically different fallback. |
| Reference attention rounding | Optional target/draft boundary alignment; no demonstrated general quality gain. |
| Confidence-prefix scheduling | DSpark-based causal stopping, fixed machine costs and frozen calibration. |
| Wider prefill and prefetch budget | 8192-token chunks and a 64 MiB budget; existing oMLX mechanisms. |
| Dependency replay and numerical repairs | Reduced decoder work with selective arithmetic-shape handling and three-stage draft-context initialization. |
| Bounded HotRows integration | Reused cache algorithm with local model ownership, background-reader coordination and cleanup; 512 MiB total retained row-cache budget. |
| Launch, activation and validation support | Source/model identities, profiling, lifecycle helpers and the original benchmark execution code. |

Our most substantive technical work is the diagnosis and repair of specific numerical and draft-state differences introduced by reducing computation. Published algorithms and related implementations remain credited at the appropriate scope. The package contains no new model weights or quantization.

`source/` preserves the original lab-relative directory layout. It includes all accepted local runtime source, external helper definitions, fitted calibration, upstream modifications and their provenance. The copied prototype scripts retain their historical status and guards; read `DEPENDENCIES.md` before attempting to adapt them. The full installed app, native libraries, model weights and environment directories are external.

`evidence/overall/` contains the complete new comparison: exact token inputs in `plan-01.json`, all six raw process records, loading/resource observations, frozen plans, source receipts, original analyzers and reviewed figures. `evidence/historical/` contains selected component measurements and closed-study reports, with an explicit [scope and omissions guide](evidence/historical/README.md). Older experiments have different comparators and their percentages must not be combined.

`SOURCE-INVENTORY.json` and `EVIDENCE-INVENTORY.json` map original paths and digests to packaged files. Where necessary, personal filesystem prefixes are replaced with `${ORIGINAL_LAB_ROOT}` or `${USER_HOME}`. Original digest fields in historical files continue to describe the original bytes. Use the inventories and package manifest to verify the redistributed copies; path normalization does not update historical acceptance or source guards.

## Scope

The checkpoint revision is `78456115d301eecf9fadcdcc81c0cf6f8b2c8eab`. Measurements used MLX 0.32.4, the installed oMLX 0.7.1rc1 runtime and its pinned dependencies. The full local profile allowed one active request and 131072 total context tokens. This timing suite used inputs up to 65536 tokens. Longer offline state checks do not qualify optimized million-token serving; the separate long-HTTP gate failed and is retained in the historical record.

A clean installation and inference run of this archive have **not** been qualified. The public historical oMLX release supplies a possible reconstruction route, but its inspected wheel's native binary hashes differ from the measured installed build. A future installer needs a separately pinned environment and finite actual-artifact checks. A future current-main performance claim needs a current-main comparison.

## License and review status

Original additions and new release documentation are offered under Apache-2.0 in this proposed release. Existing MIT/Apache source notices and public-fixture licenses remain applicable; see [LICENSE](LICENSE), [NOTICE](NOTICE), [attribution](docs/ATTRIBUTION.md) and [fixture notices](evidence/overall/fixtures/NOTICE). No model weights or copied application distribution are included.

This is a local review artifact. The proposed public home is the existing `rubencu/omlx` fork, on a dedicated research branch and immutable release tag. No branch, release, article or upstream PR has been published by preparing this package. Public actions require the author's review of these exact contents.
